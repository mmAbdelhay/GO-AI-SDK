package memory_test

import (
	"context"
	"strings"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/aitest"
	"github.com/mmabdelhay/go-ai-sdk/memory"
)

func TestInMemoryStore(t *testing.T) {
	ctx := context.Background()
	s := memory.NewInMemory()

	msgs, err := s.Messages(ctx, "none")
	if err != nil || len(msgs) != 0 {
		t.Fatalf("missing conversation should be empty, got %v, %v", msgs, err)
	}

	_ = s.Append(ctx, "c1", ai.UserText("one"))
	_ = s.Append(ctx, "c1", ai.AssistantText("two"))
	msgs, _ = s.Messages(ctx, "c1")
	if len(msgs) != 2 || msgs[1].Role != ai.RoleAssistant {
		t.Errorf("msgs = %+v", msgs)
	}

	// Returned slice is a copy; mutating it must not affect the store.
	msgs[0] = ai.UserText("mutated")
	fresh, _ := s.Messages(ctx, "c1")
	if fresh[0].Content[0].(ai.Text).Text != "one" {
		t.Error("store leaked internal slice")
	}

	_ = s.Clear(ctx, "c1")
	msgs, _ = s.Messages(ctx, "c1")
	if len(msgs) != 0 {
		t.Errorf("clear failed: %v", msgs)
	}
}

func TestTruncateKeepsSystemAndRecent(t *testing.T) {
	long := strings.Repeat("x ", 400) // ~200 tokens per message
	msgs := []ai.Message{
		ai.System("rules"),
		ai.UserText(long),
		ai.AssistantText(long),
		ai.UserText("latest question"),
	}
	dropped := 0
	fitted, err := memory.Truncate{OnDrop: func(n int) { dropped = n }}.
		Fit(context.Background(), msgs, 100)
	if err != nil {
		t.Fatal(err)
	}
	if fitted[0].Role != ai.RoleSystem {
		t.Error("system message was dropped")
	}
	lastText := fitted[len(fitted)-1].Content[0].(ai.Text).Text
	if lastText != "latest question" {
		t.Errorf("latest message lost, last = %q", lastText)
	}
	if dropped == 0 {
		t.Error("drop not reported")
	}
}

func TestTruncateNoopUnderBudget(t *testing.T) {
	msgs := []ai.Message{ai.UserText("short")}
	fitted, _ := memory.Truncate{}.Fit(context.Background(), msgs, 1000)
	if len(fitted) != 1 {
		t.Errorf("under-budget conversation was modified")
	}
}

func TestSummarize(t *testing.T) {
	summarizer := &aitest.Model{Responses: []ai.Response{
		aitest.TextResponse("Earlier, the user introduced themselves as Ada."),
	}}
	long := strings.Repeat("chatter ", 200)
	msgs := []ai.Message{
		ai.UserText("My name is Ada. " + long),
		ai.AssistantText(long),
		ai.UserText("q1"),
		ai.AssistantText("a1"),
		ai.UserText("q2"),
		ai.AssistantText("a2"),
	}
	fitted, err := memory.Summarize{Model: summarizer, Keep: 4}.
		Fit(context.Background(), msgs, 100)
	if err != nil {
		t.Fatal(err)
	}
	// One summary message + the 4 kept.
	if len(fitted) != 5 {
		t.Fatalf("len = %d, want 5", len(fitted))
	}
	if !strings.Contains(fitted[0].Content[0].(ai.Text).Text, "Ada") {
		t.Errorf("summary lost key fact: %+v", fitted[0])
	}
}
