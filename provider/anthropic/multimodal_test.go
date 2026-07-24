package anthropic

import (
	"errors"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

func TestDocumentTranslates(t *testing.T) {
	c := New("k", WithModel("m"))
	w, err := c.toWire(ai.Request{Messages: []ai.Message{
		{Role: ai.RoleUser, Content: []ai.Part{
			ai.Document{Data: []byte("PDFDATA"), MediaType: "application/pdf", Name: "report.pdf"},
		}},
	}}, false)
	if err != nil {
		t.Fatal(err)
	}
	block := w.Messages[0].Content[0]
	if block.Type != "document" || block.Source == nil {
		t.Fatalf("block = %+v", block)
	}
	if block.Source.Type != "base64" || block.Source.MediaType != "application/pdf" {
		t.Errorf("source = %+v", block.Source)
	}
	if block.Title != "report.pdf" {
		t.Errorf("title = %q", block.Title)
	}
}

func TestDocumentURLTranslates(t *testing.T) {
	c := New("k", WithModel("m"))
	w, err := c.toWire(ai.Request{Messages: []ai.Message{
		{Role: ai.RoleUser, Content: []ai.Part{ai.Document{URL: "https://example.com/x.pdf"}}},
	}}, false)
	if err != nil {
		t.Fatal(err)
	}
	src := w.Messages[0].Content[0].Source
	if src.Type != "url" || src.URL != "https://example.com/x.pdf" {
		t.Errorf("source = %+v", src)
	}
}

func TestAudioIsExplicitlyUnsupported(t *testing.T) {
	c := New("k", WithModel("m"))
	_, err := c.toWire(ai.Request{Messages: []ai.Message{
		{Role: ai.RoleUser, Content: []ai.Part{ai.Audio{Data: []byte("x"), MediaType: "audio/wav"}}},
	}}, false)
	if err == nil {
		t.Fatal("expected explicit error for unsupported audio, got nil (silent drop is forbidden)")
	}
	if !containsSub(err.Error(), "audio") {
		t.Errorf("error should mention audio: %v", err)
	}
	// Sanity: it is a plain translation error, not a normalized API error.
	var apiErr *ai.APIError
	if errors.As(err, &apiErr) {
		t.Errorf("unexpected APIError: %v", err)
	}
}

func containsSub(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
