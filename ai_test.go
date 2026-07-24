package ai_test

import (
	"context"
	"errors"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

func TestMessageConstructors(t *testing.T) {
	tests := []struct {
		name string
		msg  ai.Message
		role ai.Role
		text string
	}{
		{"system", ai.System("be terse"), ai.RoleSystem, "be terse"},
		{"user", ai.UserText("hi"), ai.RoleUser, "hi"},
		{"assistant", ai.AssistantText("hello"), ai.RoleAssistant, "hello"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.msg.Role != tt.role {
				t.Errorf("role = %q, want %q", tt.msg.Role, tt.role)
			}
			if got := (ai.Response{Message: tt.msg}).Text(); got != tt.text {
				t.Errorf("text = %q, want %q", got, tt.text)
			}
		})
	}
}

func TestResponseTextConcatenatesTextParts(t *testing.T) {
	resp := ai.Response{Message: ai.Message{
		Role: ai.RoleAssistant,
		Content: []ai.Part{
			ai.Text{Text: "Hello, "},
			ai.Image{URL: "http://example.com/x.png"}, // ignored
			ai.Text{Text: "world"},
		},
	}}
	if got, want := resp.Text(), "Hello, world"; got != want {
		t.Errorf("Text() = %q, want %q", got, want)
	}
}

func TestUsageTotalTokens(t *testing.T) {
	u := ai.Usage{InputTokens: 10, OutputTokens: 25}
	if got, want := u.TotalTokens(), 35; got != want {
		t.Errorf("TotalTokens() = %d, want %d", got, want)
	}
}

func TestOptionsBuildRequest(t *testing.T) {
	// Generate delegates to a fake so we can inspect the built Request.
	var got ai.Request
	fake := chatFunc(func(_ context.Context, req ai.Request) (ai.Response, error) {
		got = req
		return ai.Response{}, nil
	})

	_, err := ai.Generate(context.Background(), fake, "the prompt",
		ai.WithModel("m"),
		ai.WithSystem("sys"),
		ai.WithTemperature(0.5),
		ai.WithTopP(0.9),
		ai.WithMaxTokens(128),
		ai.WithStopSequences("STOP"),
		ai.WithMessages(ai.UserText("earlier")),
	)
	if err != nil {
		t.Fatal(err)
	}

	if got.Model != "m" || got.System != "sys" || got.MaxTokens != 128 {
		t.Errorf("scalar options not applied: %+v", got)
	}
	if got.Temperature == nil || *got.Temperature != 0.5 {
		t.Errorf("temperature not applied: %v", got.Temperature)
	}
	if got.TopP == nil || *got.TopP != 0.9 {
		t.Errorf("top_p not applied: %v", got.TopP)
	}
	if len(got.StopSequences) != 1 || got.StopSequences[0] != "STOP" {
		t.Errorf("stop sequences not applied: %v", got.StopSequences)
	}
	// The prompt is appended after messages supplied via WithMessages.
	if len(got.Messages) != 2 || got.Messages[0].Role != ai.RoleUser ||
		got.Messages[1].Content[0].(ai.Text).Text != "the prompt" {
		t.Errorf("messages not built correctly: %+v", got.Messages)
	}
}

func TestGenerateEmptyPromptAddsNoMessage(t *testing.T) {
	var got ai.Request
	fake := chatFunc(func(_ context.Context, req ai.Request) (ai.Response, error) {
		got = req
		return ai.Response{}, nil
	})
	_, _ = ai.Generate(context.Background(), fake, "", ai.WithMessages(ai.UserText("only")))
	if len(got.Messages) != 1 {
		t.Errorf("expected 1 message, got %d", len(got.Messages))
	}
}

func TestStreamText(t *testing.T) {
	s := ai.Stream(func(yield func(ai.Chunk, error) bool) {
		yield(ai.Chunk{Type: ai.ChunkText, Text: "foo"}, nil)
		yield(ai.Chunk{Type: ai.ChunkText, Text: "bar"}, nil)
		yield(ai.Chunk{Type: ai.ChunkDone, StopReason: ai.StopEndTurn}, nil)
	})
	got, err := s.Text()
	if err != nil {
		t.Fatal(err)
	}
	if got != "foobar" {
		t.Errorf("Text() = %q, want %q", got, "foobar")
	}
}

func TestStreamTextStopsOnError(t *testing.T) {
	sentinel := errors.New("boom")
	s := ai.Stream(func(yield func(ai.Chunk, error) bool) {
		// A well-behaved producer must stop once yield returns false.
		if !yield(ai.Chunk{Type: ai.ChunkText, Text: "foo"}, nil) {
			return
		}
		if !yield(ai.Chunk{}, sentinel) {
			return
		}
		yield(ai.Chunk{Type: ai.ChunkText, Text: "unreached"}, nil)
	})
	got, err := s.Text()
	if !errors.Is(err, sentinel) {
		t.Fatalf("err = %v, want %v", err, sentinel)
	}
	if got != "foo" {
		t.Errorf("partial text = %q, want %q", got, "foo")
	}
}

func TestStreamCollect(t *testing.T) {
	s := ai.Stream(func(yield func(ai.Chunk, error) bool) {
		yield(ai.Chunk{Type: ai.ChunkText, Text: "hello "}, nil)
		yield(ai.Chunk{Type: ai.ChunkText, Text: "there"}, nil)
		yield(ai.Chunk{Type: ai.ChunkDone, StopReason: ai.StopEndTurn, Usage: &ai.Usage{InputTokens: 3, OutputTokens: 2}}, nil)
	})
	resp, err := s.Collect()
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "hello there" {
		t.Errorf("text = %q", resp.Text())
	}
	if resp.StopReason != ai.StopEndTurn {
		t.Errorf("stop = %q", resp.StopReason)
	}
	if resp.Usage.TotalTokens() != 5 {
		t.Errorf("usage total = %d, want 5", resp.Usage.TotalTokens())
	}
}

// chatFunc adapts a function to ai.ChatModel for tests that only exercise
// Generate.
type chatFunc func(context.Context, ai.Request) (ai.Response, error)

func (f chatFunc) Name() string { return "func" }
func (f chatFunc) Generate(ctx context.Context, req ai.Request) (ai.Response, error) {
	return f(ctx, req)
}
func (f chatFunc) Stream(ctx context.Context, req ai.Request) ai.Stream {
	return func(yield func(ai.Chunk, error) bool) {
		resp, err := f(ctx, req)
		if err != nil {
			yield(ai.Chunk{}, err)
			return
		}
		yield(ai.Chunk{Type: ai.ChunkText, Text: resp.Text()}, nil)
	}
}
