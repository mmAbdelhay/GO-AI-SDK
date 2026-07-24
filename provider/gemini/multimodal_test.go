package gemini

import (
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

func TestAudioAndDocumentTranslate(t *testing.T) {
	w, err := toWire(ai.Request{Messages: []ai.Message{
		{Role: ai.RoleUser, Content: []ai.Part{
			ai.Audio{Data: []byte("WAV"), MediaType: "audio/wav"},
			ai.Document{Data: []byte("PDF"), MediaType: "application/pdf"},
			ai.Document{URL: "https://example.com/x.pdf"},
		}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	parts := w.Contents[0].Parts
	if len(parts) != 3 {
		t.Fatalf("parts = %d, want 3", len(parts))
	}
	if parts[0].InlineData == nil || parts[0].InlineData.MimeType != "audio/wav" {
		t.Errorf("audio part = %+v", parts[0])
	}
	if parts[1].InlineData == nil || parts[1].InlineData.MimeType != "application/pdf" {
		t.Errorf("document inline part = %+v", parts[1])
	}
	if parts[2].FileData == nil || parts[2].FileData.FileURI != "https://example.com/x.pdf" {
		t.Errorf("document url part = %+v", parts[2])
	}
}
