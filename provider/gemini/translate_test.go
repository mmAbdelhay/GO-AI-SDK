package gemini

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
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

func TestToWireGolden(t *testing.T) {
	req := ai.Request{
		System:      "You are helpful.",
		Temperature: ptr(0.7),
		MaxTokens:   256,
		Messages: []ai.Message{
			ai.System("Be concise."),
			{Role: ai.RoleUser, Content: []ai.Part{
				ai.Text{Text: "What is in this image?"},
				ai.Image{Data: []byte("PNGDATA"), MediaType: "image/png"},
			}},
			{Role: ai.RoleAssistant, Content: []ai.Part{
				ai.ToolCall{ID: "lookup", Name: "lookup", Input: json.RawMessage(`{"q":"x"}`)},
			}},
			{Role: ai.RoleTool, Content: []ai.Part{
				ai.ToolResult{CallID: "lookup", Content: []ai.Part{ai.Text{Text: "42"}}},
			}},
		},
		Tools: []ai.ToolDef{{
			Name: "lookup", Description: "Looks things up",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"additionalProperties":false}`),
		}},
		ToolChoice: ai.ToolAuto,
	}
	w, err := toWire(req)
	if err != nil {
		t.Fatal(err)
	}
	got, err := json.MarshalIndent(w, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	goldenBytes(t, "request.golden.json", got)
}

func TestSystemFoldedIntoInstruction(t *testing.T) {
	w, err := toWire(ai.Request{
		System:   "First.",
		Messages: []ai.Message{ai.System("Second."), ai.UserText("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if w.SystemInstruction == nil || w.SystemInstruction.Parts[0].Text != "First.\n\nSecond." {
		t.Errorf("system instruction = %+v", w.SystemInstruction)
	}
	if len(w.Contents) != 1 {
		t.Errorf("system message leaked into contents: %+v", w.Contents)
	}
}

func TestAssistantRoleIsModel(t *testing.T) {
	w, err := toWire(ai.Request{Messages: []ai.Message{ai.AssistantText("hello")}})
	if err != nil {
		t.Fatal(err)
	}
	if w.Contents[0].Role != "model" {
		t.Errorf("role = %q, want model", w.Contents[0].Role)
	}
}

func TestFromWireFunctionCall(t *testing.T) {
	body := `{
		"candidates":[{"content":{"role":"model","parts":[
			{"functionCall":{"name":"lookup","args":{"q":"go"}}}
		]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}}`
	var wr gResponse
	if err := json.Unmarshal([]byte(body), &wr); err != nil {
		t.Fatal(err)
	}
	resp, err := fromWire(wr, "m", []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	// Function calls imply tool use even though Gemini reports STOP.
	if resp.StopReason != ai.StopToolUse {
		t.Errorf("stop = %q, want tool_use", resp.StopReason)
	}
	tc := resp.Message.Content[0].(ai.ToolCall)
	if tc.Name != "lookup" || tc.ID != "lookup" {
		t.Errorf("tool call = %+v", tc)
	}
}

func TestSchemaAdditionalPropsStripped(t *testing.T) {
	in := json.RawMessage(`{"type":"object","additionalProperties":false,"properties":{}}`)
	out := stripAdditionalProps(in)
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	if _, present := m["additionalProperties"]; present {
		t.Errorf("additionalProperties not stripped: %s", out)
	}
}
