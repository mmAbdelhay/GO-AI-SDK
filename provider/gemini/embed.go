package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// Embedder implements [ai.Embedder] against Gemini's batchEmbedContents API.
type Embedder struct {
	client *Client
	model  string
}

// NewEmbedder constructs an Embedder for the given embedding model (for example
// "text-embedding-004").
func NewEmbedder(apiKey, model string, opts ...Option) *Embedder {
	return &Embedder{client: New(apiKey, opts...), model: model}
}

// Name implements [ai.Embedder].
func (e *Embedder) Name() string { return e.model }

// Embed implements [ai.Embedder]. Gemini does not report token usage for
// embeddings, so EmbedResult.Usage is zero.
func (e *Embedder) Embed(ctx context.Context, texts []string) (ai.EmbedResult, error) {
	if len(texts) == 0 {
		return ai.EmbedResult{Model: e.model}, nil
	}
	reqs := make([]gEmbedOne, 0, len(texts))
	for _, t := range texts {
		reqs = append(reqs, gEmbedOne{
			Model:   "models/" + e.model,
			Content: gContent{Parts: []gPart{{Text: t}}},
		})
	}
	resp, err := e.client.post(ctx, "/models/"+e.model+":batchEmbedContents", gEmbedRequest{Requests: reqs})
	if err != nil {
		return ai.EmbedResult{}, fmt.Errorf("gemini: request failed: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return ai.EmbedResult{}, fmt.Errorf("gemini: reading response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return ai.EmbedResult{}, parseError(resp, raw)
	}
	var wr gEmbedResponse
	if err := json.Unmarshal(raw, &wr); err != nil {
		return ai.EmbedResult{}, fmt.Errorf("gemini: decoding response: %w", err)
	}
	if len(wr.Embeddings) != len(texts) {
		return ai.EmbedResult{}, fmt.Errorf("gemini: got %d embeddings for %d inputs", len(wr.Embeddings), len(texts))
	}
	res := ai.EmbedResult{Model: e.model}
	for _, d := range wr.Embeddings {
		res.Vectors = append(res.Vectors, d.Values)
	}
	return res, nil
}

var _ ai.Embedder = (*Embedder)(nil)
