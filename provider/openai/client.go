// Package openai implements the [ai.ChatModel] and [ai.Embedder] interfaces
// against the OpenAI Chat Completions and Embeddings APIs, using only the
// standard library.
//
// Because several providers expose OpenAI-compatible APIs, this client is
// reusable: the groq and xai packages are thin constructors over it that set the
// base URL and compatibility knobs. Use [WithProviderName] and
// [WithLegacyMaxTokens] when pointing it at another compatible endpoint.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

const defaultBaseURL = "https://api.openai.com/v1"

// Client is an OpenAI-compatible chat client implementing [ai.ChatModel].
type Client struct {
	apiKey          string
	baseURL         string
	model           string
	provider        string
	legacyMaxTokens bool
	httpClient      *http.Client
}

// Option configures a [Client].
type Option func(*Client)

// WithModel sets the default model used when a request does not specify one.
func WithModel(model string) Option { return func(c *Client) { c.model = model } }

// WithBaseURL overrides the API base URL (for compatible providers, proxies, and
// testing). It should include the version prefix, e.g. ".../v1".
func WithBaseURL(url string) Option { return func(c *Client) { c.baseURL = url } }

// WithHTTPClient sets the underlying HTTP client. The library performs no
// retries of its own (design principle P9); configure them here.
func WithHTTPClient(hc *http.Client) Option { return func(c *Client) { c.httpClient = hc } }

// WithProviderName sets the provider label used in normalized errors, for
// OpenAI-compatible endpoints (e.g. "groq").
func WithProviderName(name string) Option { return func(c *Client) { c.provider = name } }

// WithLegacyMaxTokens makes the client send the deprecated "max_tokens" field
// instead of "max_completion_tokens". Some compatible providers only accept the
// legacy field.
func WithLegacyMaxTokens() Option { return func(c *Client) { c.legacyMaxTokens = true } }

// New constructs a Client. The apiKey is sent as a bearer token.
func New(apiKey string, opts ...Option) *Client {
	c := &Client{
		apiKey:     apiKey,
		baseURL:    defaultBaseURL,
		provider:   "openai",
		httpClient: http.DefaultClient,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Name returns the default model identifier, or the provider name if none is
// set.
func (c *Client) Name() string {
	if c.model != "" {
		return c.model
	}
	return c.provider
}

func (c *Client) post(ctx context.Context, path string, body any) (*http.Response, error) {
	b, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("%s: marshaling request: %w", c.provider, err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("content-type", "application/json")
	req.Header.Set("authorization", "Bearer "+c.apiKey)
	return c.httpClient.Do(req)
}

// Generate implements [ai.ChatModel].
func (c *Client) Generate(ctx context.Context, req ai.Request) (ai.Response, error) {
	w, err := c.toWire(req, false)
	if err != nil {
		return ai.Response{}, err
	}
	resp, err := c.post(ctx, "/chat/completions", w)
	if err != nil {
		return ai.Response{}, fmt.Errorf("%s: request failed: %w", c.provider, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return ai.Response{}, fmt.Errorf("%s: reading response: %w", c.provider, err)
	}
	if resp.StatusCode != http.StatusOK {
		return ai.Response{}, c.parseError(resp, raw)
	}
	var wr oaResponse
	if err := json.Unmarshal(raw, &wr); err != nil {
		return ai.Response{}, fmt.Errorf("%s: decoding response: %w", c.provider, err)
	}
	return c.fromWire(wr, raw)
}

// Stream implements [ai.ChatModel]. Setup errors are delivered as the first
// (Chunk, error) pair.
func (c *Client) Stream(ctx context.Context, req ai.Request) ai.Stream {
	return func(yield func(ai.Chunk, error) bool) {
		w, err := c.toWire(req, true)
		if err != nil {
			yield(ai.Chunk{}, err)
			return
		}
		resp, err := c.post(ctx, "/chat/completions", w)
		if err != nil {
			yield(ai.Chunk{}, fmt.Errorf("%s: request failed: %w", c.provider, err))
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			raw, _ := io.ReadAll(resp.Body)
			yield(ai.Chunk{}, c.parseError(resp, raw))
			return
		}
		c.parseSSE(resp.Body, yield)
	}
}

var _ ai.ChatModel = (*Client)(nil)
