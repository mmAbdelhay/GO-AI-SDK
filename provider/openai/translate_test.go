package openai

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

var update = flag.Bool("update", false, "update golden files")

func goldenBytes(t *testing.T, name string, got []byte) {
	t.Helper()
	path := filepath.Join("testdata", name)
	if *update {
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading golden %s: %v (run with -update to create)", path, err)
	}
	if string(got) != string(want) {
		t.Errorf("golden mismatch for %s\n--- got ---\n%s\n--- want ---\n%s", name, got, want)
	}
}

func ptr[T any](v T) *T { return &v }

func sampleRequest() ai.Request {
	return ai.Request{
		Model:       "gpt-4o",
		System:      "You are helpful.",
		Temperature: ptr(0.7),
		MaxTokens:   256,
		Messages: []ai.Message{
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
		Tools: []ai.ToolDef{{
			Name: "lookup", Description: "Looks things up",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`),
		}},
		ToolChoice:      ai.ToolAuto,
		ProviderOptions: []ai.ProviderOption{WithSeed(42)},
	}
}

func TestToWireGolden(t *testing.T) {
	c := New("k")
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

func TestToWireLegacyMaxTokens(t *testing.T) {
	req := ai.Request{Model: "m", MaxTokens: 10, Messages: []ai.Message{ai.UserText("hi")}}

	modern, err := New("k").toWire(req, false)
	if err != nil {
		t.Fatal(err)
	}
	if modern.MaxCompletionTokens == nil || modern.MaxTokens != nil {
		t.Errorf("modern client should set max_completion_tokens only")
	}

	legacy, err := New("k", WithLegacyMaxTokens()).toWire(req, false)
	if err != nil {
		t.Fatal(err)
	}
	if legacy.MaxTokens == nil || legacy.MaxCompletionTokens != nil {
		t.Errorf("legacy client should set max_tokens only")
	}
}

func TestToWireStreamSetsUsageOption(t *testing.T) {
	w, err := New("k", WithModel("m")).toWire(ai.Request{Messages: []ai.Message{ai.UserText("hi")}}, true)
	if err != nil {
		t.Fatal(err)
	}
	if !w.Stream || w.StreamOptions == nil || !w.StreamOptions.IncludeUsage {
		t.Errorf("stream request must request usage: %+v", w)
	}
}

func TestFromWireToolCalls(t *testing.T) {
	body := `{
		"model":"m",
		"choices":[{"message":{"role":"assistant","content":"","tool_calls":[
			{"id":"c1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"go\"}"}}
		]},"finish_reason":"tool_calls"}],
		"usage":{"prompt_tokens":5,"completion_tokens":2}}`
	var wr oaResponse
	if err := json.Unmarshal([]byte(body), &wr); err != nil {
		t.Fatal(err)
	}
	resp, err := New("k").fromWire(wr, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if resp.StopReason != ai.StopToolUse {
		t.Errorf("stop = %q", resp.StopReason)
	}
	tc := resp.Message.Content[0].(ai.ToolCall)
	if tc.Name != "lookup" || string(tc.Input) != `{"q":"go"}` {
		t.Errorf("tool call = %+v", tc)
	}
}

func TestToolResultBecomesToolMessage(t *testing.T) {
	msgs, err := toWireMessages(ai.Message{Role: ai.RoleTool, Content: []ai.Part{
		ai.ToolResult{CallID: "c1", Content: []ai.Part{ai.Text{Text: "ok"}}},
		ai.ToolResult{CallID: "c2", Content: []ai.Part{ai.Text{Text: "bad"}}, IsError: true},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 2 || msgs[0].Role != "tool" || msgs[0].ToolCallID != "c1" {
		t.Fatalf("msgs = %+v", msgs)
	}
	if !strings.HasPrefix(msgs[1].Content.(string), "ERROR:") {
		t.Errorf("error result not marked: %+v", msgs[1])
	}
}
