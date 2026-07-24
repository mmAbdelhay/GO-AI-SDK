package agent_test

import (
	"context"
	"fmt"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/agent"
	"github.com/mmabdelhay/go-ai-sdk/aitest"
	"github.com/mmabdelhay/go-ai-sdk/tool"
)

// Example runs an agent with a tool against a scripted fake model, so it needs
// no network or API key. Swap the fake for a real provider client in production.
func Example() {
	echo := tool.MustNew("echo", "Echoes its input",
		func(_ context.Context, in struct {
			Text string `json:"text"`
		}) (string, error) {
			return in.Text, nil
		})

	model := &aitest.Model{Responses: []ai.Response{
		{
			Message: ai.Message{Role: ai.RoleAssistant, Content: []ai.Part{
				ai.ToolCall{ID: "c1", Name: "echo", Input: []byte(`{"text":"ping"}`)},
			}},
			StopReason: ai.StopToolUse,
		},
		aitest.TextResponse("The tool said: ping"),
	}}

	a := agent.New(model, agent.WithTools(echo), agent.WithMaxIterations(5))
	res, err := a.Run(context.Background(), "Use the echo tool.")
	if err != nil {
		panic(err)
	}
	fmt.Println(res.Output)
	fmt.Println("tool calls:", res.ToolCalls)
	// Output:
	// The tool said: ping
	// tool calls: 1
}
