package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// Embedder implements [ai.Embedder] against the Embeddings API.
type Embedder struct {
	client *Client
	model  string
}

// NewEmbedder constructs an Embedder. The model (for example
// "text-embedding-3-small") is required, because vector stores key on it.
// Client options (base URL, HTTP client, provider name) apply as for [New].
func NewEmbedder(apiKey, model string, opts ...Option) *Embedder {
	return &Embedder{client: New(apiKey, opts...), model: model}
}

// Name implements [ai.Embedder].
func (e *Embedder) Name() string { return e.model }

// Embed implements [ai.Embedder].
func (e *Embedder) Embed(ctx context.Context, texts []string) (ai.EmbedResult, error) {
	if len(texts) == 0 {
		return ai.EmbedResult{Model: e.model}, nil
	}
	resp, err := e.client.post(ctx, "/embeddings", oaEmbedRequest{
		Model: e.model, Input: texts, EncodingFormat: "float",
	})
	if err != nil {
		return ai.EmbedResult{}, fmt.Errorf("%s: request failed: %w", e.client.provider, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return ai.EmbedResult{}, fmt.Errorf("%s: reading response: %w", e.client.provider, err)
	}
	if resp.StatusCode != http.StatusOK {
		return ai.EmbedResult{}, e.client.parseError(resp, raw)
	}
	var wr oaEmbedResponse
	if err := json.Unmarshal(raw, &wr); err != nil {
		return ai.EmbedResult{}, fmt.Errorf("%s: decoding response: %w", e.client.provider, err)
	}
	if len(wr.Data) != len(texts) {
		return ai.EmbedResult{}, fmt.Errorf("%s: got %d embeddings for %d inputs", e.client.provider, len(wr.Data), len(texts))
	}
	res := ai.EmbedResult{Model: e.model, Usage: ai.Usage{InputTokens: wr.Usage.PromptTokens}}
	for _, d := range wr.Data {
		res.Vectors = append(res.Vectors, d.Embedding)
	}
	return res, nil
}

var _ ai.Embedder = (*Embedder)(nil)
