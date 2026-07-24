package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// Transcriber implements [ai.Transcriber] against the OpenAI speech-to-text API.
type Transcriber struct {
	client *Client
	model  string
}

// NewTranscriber constructs a Transcriber (for example model "whisper-1" or
// "gpt-4o-transcribe"). Client options apply as for [New].
func NewTranscriber(apiKey, model string, opts ...Option) *Transcriber {
	return &Transcriber{client: New(apiKey, opts...), model: model}
}

// Name implements [ai.Transcriber].
func (tr *Transcriber) Name() string { return tr.model }

type oaTranscriptionResponse struct {
	Text     string `json:"text"`
	Language string `json:"language"`
}

// Transcribe implements [ai.Transcriber]. The audio is uploaded as a multipart
// form, built with the standard library's mime/multipart.
func (tr *Transcriber) Transcribe(ctx context.Context, req ai.TranscriptionRequest) (ai.Transcription, error) {
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)

	filename := req.Filename
	if filename == "" {
		filename = "audio" + extForMediaType(req.MediaType)
	}
	fw, err := mw.CreateFormFile("file", filename)
	if err != nil {
		return ai.Transcription{}, err
	}
	if _, err := fw.Write(req.Audio); err != nil {
		return ai.Transcription{}, err
	}
	fields := map[string]string{"model": tr.model}
	if req.Language != "" {
		fields["language"] = req.Language
	}
	if req.Prompt != "" {
		fields["prompt"] = req.Prompt
	}
	for k, v := range fields {
		if err := mw.WriteField(k, v); err != nil {
			return ai.Transcription{}, err
		}
	}
	if err := mw.Close(); err != nil {
		return ai.Transcription{}, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		tr.client.baseURL+"/audio/transcriptions", &buf)
	if err != nil {
		return ai.Transcription{}, err
	}
	httpReq.Header.Set("Content-Type", mw.FormDataContentType())
	httpReq.Header.Set("Authorization", "Bearer "+tr.client.apiKey)

	resp, err := tr.client.httpClient.Do(httpReq)
	if err != nil {
		return ai.Transcription{}, fmt.Errorf("%s: request failed: %w", tr.client.provider, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return ai.Transcription{}, fmt.Errorf("%s: reading response: %w", tr.client.provider, err)
	}
	if resp.StatusCode != http.StatusOK {
		return ai.Transcription{}, tr.client.parseError(resp, body)
	}
	var wr oaTranscriptionResponse
	if err := json.Unmarshal(body, &wr); err != nil {
		return ai.Transcription{}, fmt.Errorf("%s: decoding response: %w", tr.client.provider, err)
	}
	lang := wr.Language
	if lang == "" {
		lang = req.Language
	}
	return ai.Transcription{Text: wr.Text, Language: lang}, nil
}

func extForMediaType(mediaType string) string {
	switch mediaType {
	case "audio/mpeg", "audio/mp3":
		return ".mp3"
	case "audio/wav", "audio/x-wav", "audio/wave":
		return ".wav"
	case "audio/flac":
		return ".flac"
	case "audio/ogg":
		return ".ogg"
	case "audio/webm":
		return ".webm"
	default:
		return ".mp3"
	}
}

var _ ai.Transcriber = (*Transcriber)(nil)
