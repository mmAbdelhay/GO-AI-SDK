package ai

import "context"

// SpeechModel synthesizes speech audio from text (text-to-speech). Provider
// implementations live in provider packages; a fake lives in aitest.
type SpeechModel interface {
	// Name returns the speech model identifier.
	Name() string
	// Synthesize renders req.Text to audio bytes.
	Synthesize(ctx context.Context, req SpeechRequest) (SpeechResponse, error)
}

// SpeechRequest is a text-to-speech request.
type SpeechRequest struct {
	// Text is the content to speak.
	Text string
	// Voice selects the provider voice (for example "alloy"). Empty uses the
	// provider default.
	Voice string
	// Format is the desired audio container/codec (for example "mp3", "wav").
	// Empty uses the provider default.
	Format string
	// Speed, when non-nil, scales the speaking rate.
	Speed *float64
}

// SpeechResponse is synthesized audio.
type SpeechResponse struct {
	// Audio is the raw audio bytes.
	Audio []byte
	// MediaType is the IANA media type of Audio (for example "audio/mpeg").
	MediaType string
}

// Transcriber converts speech audio to text (speech-to-text).
type Transcriber interface {
	// Name returns the transcription model identifier.
	Name() string
	// Transcribe returns the text spoken in req.Audio.
	Transcribe(ctx context.Context, req TranscriptionRequest) (Transcription, error)
}

// TranscriptionRequest is a speech-to-text request.
type TranscriptionRequest struct {
	// Audio is the raw audio bytes.
	Audio []byte
	// MediaType is the IANA media type of Audio; providers use it (and Filename)
	// to infer the format.
	MediaType string
	// Filename is a hint (for example "speech.mp3") for providers that key on
	// the extension. Optional.
	Filename string
	// Language is an optional ISO-639-1 hint (for example "en").
	Language string
	// Prompt is optional text to guide the transcription style/vocabulary.
	Prompt string
}

// Transcription is the result of a [Transcriber.Transcribe] call.
type Transcription struct {
	// Text is the transcribed text.
	Text string
	// Language is the detected (or requested) language, when reported.
	Language string
}
