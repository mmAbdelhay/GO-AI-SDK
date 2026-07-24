package openai

import (
	"strings"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

func TestAudioTranslates(t *testing.T) {
	msgs, err := toWireMessages(ai.Message{Role: ai.RoleUser, Content: []ai.Part{
		ai.Text{Text: "transcribe this"},
		ai.Audio{Data: []byte("WAVDATA"), MediaType: "audio/wav"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	parts, ok := msgs[0].Content.([]oaContentPart)
	if !ok {
		t.Fatalf("content is %T, want []oaContentPart (array form)", msgs[0].Content)
	}
	var audio *oaContentPart
	for i := range parts {
		if parts[i].Type == "input_audio" {
			audio = &parts[i]
		}
	}
	if audio == nil || audio.InputAudio == nil {
		t.Fatalf("no input_audio part: %+v", parts)
	}
	if audio.InputAudio.Format != "wav" {
		t.Errorf("format = %q, want wav", audio.InputAudio.Format)
	}
}

func TestDocumentTranslatesToFile(t *testing.T) {
	msgs, err := toWireMessages(ai.Message{Role: ai.RoleUser, Content: []ai.Part{
		ai.Document{Data: []byte("PDF"), MediaType: "application/pdf", Name: "r.pdf"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	parts := msgs[0].Content.([]oaContentPart)
	if parts[0].Type != "file" || parts[0].File == nil {
		t.Fatalf("part = %+v", parts[0])
	}
	if parts[0].File.Filename != "r.pdf" || !strings.HasPrefix(parts[0].File.FileData, "data:application/pdf;base64,") {
		t.Errorf("file = %+v", parts[0].File)
	}
}

func TestAudioFormatMapping(t *testing.T) {
	tests := map[string]string{
		"audio/wav": "wav", "audio/mpeg": "mp3", "audio/mp3": "mp3",
		"audio/flac": "flac", "audio/ogg": "ogg", "audio/weird": "weird",
	}
	for in, want := range tests {
		if got := audioFormat(in); got != want {
			t.Errorf("audioFormat(%q) = %q, want %q", in, got, want)
		}
	}
}
