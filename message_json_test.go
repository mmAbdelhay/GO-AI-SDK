package ai_test

import (
	"encoding/json"
	"reflect"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

func TestMessageJSONRoundTrip(t *testing.T) {
	msgs := []ai.Message{
		ai.System("be terse"),
		{Role: ai.RoleUser, Content: []ai.Part{
			ai.Text{Text: "look at these"},
			ai.Image{Data: []byte("PNG"), MediaType: "image/png"},
			ai.Image{URL: "https://example.com/x.jpg"},
			ai.Audio{Data: []byte("WAV"), MediaType: "audio/wav"},
			ai.Document{Data: []byte("PDF"), MediaType: "application/pdf", Name: "report.pdf"},
			ai.Document{URL: "https://example.com/doc.pdf"},
		}},
		{Role: ai.RoleAssistant, Content: []ai.Part{
			ai.Text{Text: "calling a tool"},
			ai.ToolCall{ID: "c1", Name: "lookup", Input: json.RawMessage(`{"q":"x"}`)},
		}},
		{Role: ai.RoleTool, Content: []ai.Part{
			ai.ToolResult{CallID: "c1", Content: []ai.Part{ai.Text{Text: "42"}}, IsError: false},
			ai.ToolResult{CallID: "c2", Content: []ai.Part{ai.Text{Text: "boom"}}, IsError: true},
		}},
	}

	for _, orig := range msgs {
		data, err := json.Marshal(orig)
		if err != nil {
			t.Fatalf("marshal %v: %v", orig.Role, err)
		}
		var got ai.Message
		if err := json.Unmarshal(data, &got); err != nil {
			t.Fatalf("unmarshal %s: %v", data, err)
		}
		if !reflect.DeepEqual(orig, got) {
			t.Errorf("round trip mismatch\n orig = %#v\n got  = %#v\n json = %s", orig, got, data)
		}
	}
}

func TestMessageSliceJSONRoundTrip(t *testing.T) {
	conv := []ai.Message{ai.UserText("hi"), ai.AssistantText("hello")}
	data, err := json.Marshal(conv)
	if err != nil {
		t.Fatal(err)
	}
	var got []ai.Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(conv, got) {
		t.Errorf("conversation round trip mismatch: %#v vs %#v", conv, got)
	}
}

func TestMessageUnmarshalUnknownPartErrors(t *testing.T) {
	var m ai.Message
	err := json.Unmarshal([]byte(`{"role":"user","content":[{"type":"hologram"}]}`), &m)
	if err == nil {
		t.Fatal("expected error for unknown part type")
	}
}
