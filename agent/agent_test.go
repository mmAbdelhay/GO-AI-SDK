package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/agent"
	"github.com/mmabdelhay/go-ai-sdk/aitest"
	"github.com/mmabdelhay/go-ai-sdk/memory"
	"github.com/mmabdelhay/go-ai-sdk/tool"
)

type weatherInput struct {
	City string `json:"city"`
}

func weatherTool() tool.Tool {
	return tool.MustNew("get_weather", "Weather for a city",
		func(_ context.Context, in weatherInput) (string, error) {
			return "sunny in " + in.City, nil
		})
}

// toolCallResponse builds an assistant response requesting a tool call.
func toolCallResponse(id, name, input string) ai.Response {
	return ai.Response{
		Message: ai.Message{Role: ai.RoleAssistant, Content: []ai.Part{
			ai.ToolCall{ID: id, Name: name, Input: json.RawMessage(input)},
		}},
		StopReason: ai.StopToolUse,
		Usage:      ai.Usage{InputTokens: 10, OutputTokens: 5},
	}
}

func TestAgentMultiStep(t *testing.T) {
	model := &aitest.Model{Responses: []ai.Response{
		toolCallResponse("c1", "get_weather", `{"city":"Cairo"}`),
		aitest.TextResponse("It is sunny in Cairo."),
	}}
	a := agent.New(model, agent.WithTools(weatherTool()), agent.WithSystem("be helpful"))

	res, err := a.Run(context.Background(), "Weather in Cairo?")
	if err != nil {
		t.Fatal(err)
	}
	if res.Output != "It is sunny in Cairo." {
		t.Errorf("output = %q", res.Output)
	}
	if res.Iterations != 2 || res.ToolCalls != 1 {
		t.Errorf("iterations=%d toolCalls=%d", res.Iterations, res.ToolCalls)
	}

	// The second model call must have carried the tool result back.
	reqs := model.Requests()
	last := reqs[len(reqs)-1]
	found := false
	for _, m := range last.Messages {
		for _, p := range m.Content {
			if tr, ok := p.(ai.ToolResult); ok && tr.CallID == "c1" {
				if tr.Content[0].(ai.Text).Text != "sunny in Cairo" {
					t.Errorf("tool result content = %+v", tr)
				}
				found = true
			}
		}
	}
	if !found {
		t.Error("tool result was not fed back to the model")
	}
	// Tool defs must be declared on every request.
	if len(last.Tools) != 1 || last.Tools[0].Name != "get_weather" {
		t.Errorf("tools not declared: %+v", last.Tools)
	}
}

func TestAgentMaxIterations(t *testing.T) {
	// A model that always asks for a tool never terminates on its own.
	model := &aitest.Model{GenerateFunc: func(context.Context, ai.Request) (ai.Response, error) {
		return toolCallResponse("c", "get_weather", `{"city":"Loop"}`), nil
	}}
	a := agent.New(model, agent.WithTools(weatherTool()), agent.WithMaxIterations(3))

	_, err := a.Run(context.Background(), "loop forever")
	if !errors.Is(err, agent.ErrMaxIterations) {
		t.Fatalf("err = %v, want ErrMaxIterations", err)
	}
	var le *agent.LimitError
	if !errors.As(err, &le) {
		t.Fatal("not a LimitError")
	}
	if le.Result.Iterations != 3 {
		t.Errorf("partial result iterations = %d, want 3", le.Result.Iterations)
	}
}

func TestAgentTokenBudget(t *testing.T) {
	model := &aitest.Model{GenerateFunc: func(context.Context, ai.Request) (ai.Response, error) {
		return toolCallResponse("c", "get_weather", `{"city":"X"}`), nil
	}}
	a := agent.New(model, agent.WithTools(weatherTool()), agent.WithTokenBudget(20))

	_, err := a.Run(context.Background(), "spend tokens")
	if !errors.Is(err, agent.ErrTokenBudget) {
		t.Fatalf("err = %v, want ErrTokenBudget", err)
	}
}

func TestAgentApprovalDenied(t *testing.T) {
	model := &aitest.Model{Responses: []ai.Response{
		toolCallResponse("c1", "get_weather", `{"city":"Cairo"}`),
	}}
	denied := errors.New("not allowed")
	a := agent.New(model,
		agent.WithTools(weatherTool()),
		agent.WithApproval(func(_ context.Context, call ai.ToolCall) error {
			if call.Name == "get_weather" {
				return denied
			}
			return nil
		}))

	_, err := a.Run(context.Background(), "weather?")
	if !errors.Is(err, agent.ErrApprovalDenied) {
		t.Fatalf("err = %v, want ErrApprovalDenied", err)
	}
}

func TestAgentContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	model := &aitest.Model{GenerateFunc: func(context.Context, ai.Request) (ai.Response, error) {
		cancel() // cancel mid-run, after the first model call
		return toolCallResponse("c", "get_weather", `{"city":"X"}`), nil
	}}
	a := agent.New(model, agent.WithTools(weatherTool()))

	_, err := a.Run(ctx, "go")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}

func TestAgentMemoryAcrossRuns(t *testing.T) {
	store := memory.NewInMemory()
	calls := 0
	model := &aitest.Model{GenerateFunc: func(_ context.Context, req ai.Request) (ai.Response, error) {
		calls++
		if calls == 1 {
			return aitest.TextResponse("Nice to meet you, Ada."), nil
		}
		// Second run: prior conversation must be present in the request.
		for _, m := range req.Messages {
			for _, p := range m.Content {
				if txt, ok := p.(ai.Text); ok && txt.Text == "My name is Ada." {
					return aitest.TextResponse("Your name is Ada."), nil
				}
			}
		}
		return aitest.TextResponse("I do not know."), nil
	}}

	a := agent.New(model, agent.WithMemory(store, "conv-1"))
	if _, err := a.Run(context.Background(), "My name is Ada."); err != nil {
		t.Fatal(err)
	}
	res, err := a.Run(context.Background(), "What is my name?")
	if err != nil {
		t.Fatal(err)
	}
	if res.Output != "Your name is Ada." {
		t.Errorf("agent forgot: %q", res.Output)
	}
}

func TestAgentContextStrategyApplied(t *testing.T) {
	// A conversation exceeding the budget must degrade via the strategy, and
	// the drop must be observable.
	store := memory.NewInMemory()
	long := ""
	for range 200 {
		long += "waffle waffle waffle "
	}
	_ = store.Append(context.Background(), "c", ai.UserText(long), ai.AssistantText(long))

	dropped := 0
	model := &aitest.Model{GenerateFunc: func(_ context.Context, req ai.Request) (ai.Response, error) {
		if got := memory.EstimateTokens(req.Messages); got > 600 {
			return ai.Response{}, fmt.Errorf("context not truncated: ~%d tokens", got)
		}
		return aitest.TextResponse("ok"), nil
	}}
	a := agent.New(model,
		agent.WithMemory(store, "c"),
		agent.WithContextStrategy(memory.Truncate{OnDrop: func(n int) { dropped += n }}, 500))

	if _, err := a.Run(context.Background(), "hello"); err != nil {
		t.Fatal(err)
	}
	if dropped == 0 {
		t.Error("expected observable message drops")
	}
}
