package openai_test

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/aitest"
	"github.com/mmabdelhay/go-ai-sdk/provider/openai"
)

func TestSpeechSynthesizeOffline(t *testing.T) {
	tr := &aitest.Transport{Responses: []*http.Response{
		aitest.NewResponse(200, nil, "BINARY_AUDIO_BYTES"),
	}}
	s := openai.NewSpeech("k", "tts-1", openai.WithHTTPClient(tr.Client()))

	resp, err := s.Synthesize(context.Background(), ai.SpeechRequest{
		Text: "hello world", Voice: "nova", Format: "mp3",
	})
	if err != nil {
		t.Fatal(err)
	}
	if string(resp.Audio) != "BINARY_AUDIO_BYTES" {
		t.Errorf("audio = %q", resp.Audio)
	}
	if resp.MediaType != "audio/mpeg" {
		t.Errorf("media type = %q", resp.MediaType)
	}

	body, _ := tr.LastRequestBody()
	var sent map[string]any
	_ = json.Unmarshal(body, &sent)
	if sent["input"] != "hello world" || sent["voice"] != "nova" || sent["model"] != "tts-1" {
		t.Errorf("request body = %s", body)
	}
	url, _ := tr.LastRequestURL()
	if !strings.HasSuffix(url, "/audio/speech") {
		t.Errorf("url = %q", url)
	}
}

func TestSpeechErrorNormalized(t *testing.T) {
	tr := &aitest.Transport{Responses: []*http.Response{
		aitest.NewResponse(429, nil, `{"error":{"message":"slow","type":"rate_limit_error"}}`),
	}}
	s := openai.NewSpeech("k", "tts-1", openai.WithHTTPClient(tr.Client()))
	if _, err := s.Synthesize(context.Background(), ai.SpeechRequest{Text: "x"}); err == nil {
		t.Fatal("expected error")
	}
}

func TestTranscribeOffline(t *testing.T) {
	tr := &aitest.Transport{Responses: []*http.Response{
		aitest.NewResponse(200, nil, `{"text":"hello world","language":"english"}`),
	}}
	tc := openai.NewTranscriber("k", "whisper-1", openai.WithHTTPClient(tr.Client()))

	res, err := tc.Transcribe(context.Background(), ai.TranscriptionRequest{
		Audio: []byte("RIFF....WAVE"), MediaType: "audio/wav", Language: "en",
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "hello world" || res.Language != "english" {
		t.Errorf("result = %+v", res)
	}

	// The request must be multipart with the file and model fields.
	hdr, _ := tr.LastRequestHeader()
	if ct := hdr.Get("Content-Type"); !strings.HasPrefix(ct, "multipart/form-data") {
		t.Errorf("content type = %q, want multipart", ct)
	}
	body, _ := tr.LastRequestBody()
	s := string(body)
	if !strings.Contains(s, `name="model"`) || !strings.Contains(s, "whisper-1") {
		t.Errorf("multipart body missing model field")
	}
	if !strings.Contains(s, `name="file"`) || !strings.Contains(s, "RIFF") {
		t.Errorf("multipart body missing file")
	}
}

func TestFakeSpeechAndTranscriber(t *testing.T) {
	// The aitest fakes satisfy the interfaces and record calls.
	var sm ai.SpeechModel = &aitest.Speech{}
	resp, err := sm.Synthesize(context.Background(), ai.SpeechRequest{Text: "hi"})
	if err != nil || len(resp.Audio) == 0 {
		t.Fatalf("fake speech: %v, %q", err, resp.Audio)
	}

	fakeT := &aitest.Transcriber{Text: "scripted"}
	var trI ai.Transcriber = fakeT
	res, err := trI.Transcribe(context.Background(), ai.TranscriptionRequest{Audio: []byte("x")})
	if err != nil || res.Text != "scripted" {
		t.Fatalf("fake transcriber: %v, %q", err, res.Text)
	}
	if len(fakeT.Requests()) != 1 {
		t.Errorf("transcriber did not record request")
	}
}
