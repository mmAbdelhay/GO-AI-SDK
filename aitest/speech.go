package aitest

import (
	"context"
	"sync"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// Speech is a fake [ai.SpeechModel]. It returns scripted audio (default some
// fixed bytes) and records every request. Safe for concurrent use.
type Speech struct {
	// ModelName is returned by Name (default "fake-tts").
	ModelName string
	// Audio is the audio returned by every Synthesize call (default []byte("AUDIO")).
	Audio []byte
	// MediaType is reported on the response (default "audio/mpeg").
	MediaType string
	// Err, when non-nil, is returned by every call.
	Err error

	mu       sync.Mutex
	requests []ai.SpeechRequest
}

// Name implements [ai.SpeechModel].
func (s *Speech) Name() string {
	if s.ModelName != "" {
		return s.ModelName
	}
	return "fake-tts"
}

// Synthesize implements [ai.SpeechModel].
func (s *Speech) Synthesize(_ context.Context, req ai.SpeechRequest) (ai.SpeechResponse, error) {
	s.mu.Lock()
	s.requests = append(s.requests, req)
	s.mu.Unlock()
	if s.Err != nil {
		return ai.SpeechResponse{}, s.Err
	}
	audio := s.Audio
	if audio == nil {
		audio = []byte("AUDIO")
	}
	mt := s.MediaType
	if mt == "" {
		mt = "audio/mpeg"
	}
	return ai.SpeechResponse{Audio: audio, MediaType: mt}, nil
}

// Requests returns a copy of the recorded synthesis requests.
func (s *Speech) Requests() []ai.SpeechRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]ai.SpeechRequest, len(s.requests))
	copy(out, s.requests)
	return out
}

var _ ai.SpeechModel = (*Speech)(nil)

// Transcriber is a fake [ai.Transcriber]. It returns scripted text and records
// every request. Safe for concurrent use.
type Transcriber struct {
	// ModelName is returned by Name (default "fake-stt").
	ModelName string
	// Text is returned by every Transcribe call (default "fake transcription").
	Text string
	// Language is reported on the result.
	Language string
	// Err, when non-nil, is returned by every call.
	Err error

	mu       sync.Mutex
	requests []ai.TranscriptionRequest
}

// Name implements [ai.Transcriber].
func (t *Transcriber) Name() string {
	if t.ModelName != "" {
		return t.ModelName
	}
	return "fake-stt"
}

// Transcribe implements [ai.Transcriber].
func (t *Transcriber) Transcribe(_ context.Context, req ai.TranscriptionRequest) (ai.Transcription, error) {
	t.mu.Lock()
	t.requests = append(t.requests, req)
	t.mu.Unlock()
	if t.Err != nil {
		return ai.Transcription{}, t.Err
	}
	text := t.Text
	if text == "" {
		text = "fake transcription"
	}
	return ai.Transcription{Text: text, Language: t.Language}, nil
}

// Requests returns a copy of the recorded transcription requests.
func (t *Transcriber) Requests() []ai.TranscriptionRequest {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make([]ai.TranscriptionRequest, len(t.requests))
	copy(out, t.requests)
	return out
}

var _ ai.Transcriber = (*Transcriber)(nil)
