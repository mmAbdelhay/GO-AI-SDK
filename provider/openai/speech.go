package openai

import (
	"context"
	"fmt"
	"io"
	"net/http"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// Speech implements [ai.SpeechModel] against the OpenAI text-to-speech API.
type Speech struct {
	client *Client
	model  string
}

// NewSpeech constructs a Speech model (for example model "gpt-4o-mini-tts" or
// "tts-1"). Client options (base URL, HTTP client, provider name) apply as for
// [New].
func NewSpeech(apiKey, model string, opts ...Option) *Speech {
	return &Speech{client: New(apiKey, opts...), model: model}
}

// Name implements [ai.SpeechModel].
func (s *Speech) Name() string { return s.model }

type oaSpeechRequest struct {
	Model          string   `json:"model"`
	Input          string   `json:"input"`
	Voice          string   `json:"voice"`
	ResponseFormat string   `json:"response_format,omitempty"`
	Speed          *float64 `json:"speed,omitempty"`
}

// speechMediaTypes maps a response_format token to the media type OpenAI returns.
var speechMediaTypes = map[string]string{
	"mp3": "audio/mpeg", "opus": "audio/opus", "aac": "audio/aac",
	"flac": "audio/flac", "wav": "audio/wav", "pcm": "audio/pcm",
}

// Synthesize implements [ai.SpeechModel]. The response body is the raw audio.
func (s *Speech) Synthesize(ctx context.Context, req ai.SpeechRequest) (ai.SpeechResponse, error) {
	voice := req.Voice
	if voice == "" {
		voice = "alloy"
	}
	format := req.Format
	if format == "" {
		format = "mp3"
	}
	resp, err := s.client.post(ctx, "/audio/speech", oaSpeechRequest{
		Model: s.model, Input: req.Text, Voice: voice, ResponseFormat: format, Speed: req.Speed,
	})
	if err != nil {
		return ai.SpeechResponse{}, fmt.Errorf("%s: request failed: %w", s.client.provider, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return ai.SpeechResponse{}, fmt.Errorf("%s: reading response: %w", s.client.provider, err)
	}
	if resp.StatusCode != http.StatusOK {
		return ai.SpeechResponse{}, s.client.parseError(resp, body)
	}
	mediaType := speechMediaTypes[format]
	if mediaType == "" {
		mediaType = "application/octet-stream"
	}
	return ai.SpeechResponse{Audio: body, MediaType: mediaType}, nil
}

var _ ai.SpeechModel = (*Speech)(nil)
