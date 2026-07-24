package anthropic

import (
	"encoding/json"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

func ptr[T any](v T) *T { return &v }

// sampleRequest exercises system extraction, every content part type, defaults,
// and a provider option, so the golden request is broad coverage for the
// translation layer.
func sampleRequest() ai.Request {
	return ai.Request{
		Model:       "claude-sonnet-4-20250514",
		System:      "You are helpful.",
		Temperature: ptr(0.7),
		MaxTokens:   256,
		Messages: []ai.Message{
			ai.System("Also be concise."), // folded into system field
			{Role: ai.RoleUser, Content: []ai.Part{
				ai.Text{Text: "What is in this image?"},
				ai.Image{Data: []byte("PNGDATA"), MediaType: "image/png"},
			}},
			{Role: ai.RoleAssistant, Content: []ai.Part{
				ai.ToolCall{ID: "call_1", Name: "lookup", Input: json.RawMessage(`{"q":"x"}`)},
			}},
			{Role: ai.RoleTool, Content: []ai.Part{
				ai.ToolResult{CallID: "call_1", Content: []ai.Part{ai.Text{Text: "42"}}},
			}},
		},
		ProviderOptions: []ai.ProviderOption{WithTopK(40)},
	}
}

func TestToWireGolden(t *testing.T) {
	c := New("test-key")
	w, err := c.toWire(sampleRequest(), false)
	if err != nil {
		t.Fatal(err)
	}
	got, err := json.MarshalIndent(w, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	goldenBytes(t, "request.golden.json", got)
}

func TestToWireSystemFolding(t *testing.T) {
	c := New("k")
	w, err := c.toWire(sampleRequest(), false)
	if err != nil {
		t.Fatal(err)
	}
	if w.System != "You are helpful.\n\nAlso be concise." {
		t.Errorf("system folding wrong: %q", w.System)
	}
	// The system message must not appear in the wire messages array.
	for _, m := range w.Messages {
		if m.Role == "system" {
			t.Errorf("system role leaked into messages")
		}
	}
	// Tool role maps to a user message on the wire.
	last := w.Messages[len(w.Messages)-1]
	if last.Role != "user" || last.Content[0].Type != "tool_result" {
		t.Errorf("tool result not mapped to user/tool_result: %+v", last)
	}
}

func TestToWireDefaults(t *testing.T) {
	c := New("k", WithModel("default-model"), WithDefaultMaxTokens(99))
	w, err := c.toWire(ai.Request{Messages: []ai.Message{ai.UserText("hi")}}, false)
	if err != nil {
		t.Fatal(err)
	}
	if w.Model != "default-model" {
		t.Errorf("default model = %q", w.Model)
	}
	if w.MaxTokens != 99 {
		t.Errorf("default max tokens = %d", w.MaxTokens)
	}
}

func TestToWireNoModelErrors(t *testing.T) {
	c := New("k")
	if _, err := c.toWire(ai.Request{Messages: []ai.Message{ai.UserText("hi")}}, false); err == nil {
		t.Errorf("expected error when no model configured")
	}
}

func TestFromWireGolden(t *testing.T) {
	raw := readTestdata(t, "response.json")
	var wr wireResponse
	if err := json.Unmarshal(raw, &wr); err != nil {
		t.Fatal(err)
	}
	resp, err := fromWire(wr, raw)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "The image shows a gopher." {
		t.Errorf("text = %q", resp.Text())
	}
	if resp.StopReason != ai.StopEndTurn {
		t.Errorf("stop = %q", resp.StopReason)
	}
	if resp.Usage.InputTokens != 25 || resp.Usage.OutputTokens != 12 {
		t.Errorf("usage = %+v", resp.Usage)
	}
	if resp.Model != "claude-sonnet-4-20250514" {
		t.Errorf("model = %q", resp.Model)
	}
}

func TestFromWireToolUse(t *testing.T) {
	wr := wireResponse{
		Model:      "m",
		StopReason: "tool_use",
		Content: []wireBlock{
			{Type: "text", Text: "Let me check."},
			{Type: "tool_use", ID: "t1", Name: "search", Input: json.RawMessage(`{"q":"go"}`)},
		},
	}
	resp, err := fromWire(wr, nil)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StopReason != ai.StopToolUse {
		t.Errorf("stop = %q", resp.StopReason)
	}
	tc, ok := resp.Message.Content[1].(ai.ToolCall)
	if !ok {
		t.Fatalf("second part is %T, want ai.ToolCall", resp.Message.Content[1])
	}
	if tc.Name != "search" || string(tc.Input) != `{"q":"go"}` {
		t.Errorf("tool call = %+v", tc)
	}
}
