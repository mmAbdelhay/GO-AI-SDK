// Package gemini implements the [ai.ChatModel] and [ai.Embedder] interfaces
// against Google's Gemini API (generateContent), using only the standard
// library.
//
// One documented compromise: Gemini's function-calling protocol has no call IDs.
// This package uses the function name as [ai.ToolCall.ID], so correlation
// between calls and results is by name. Multiple simultaneous calls to the same
// function are therefore indistinguishable on the wire — a Gemini API
// limitation, surfaced here rather than hidden.
package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

const defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"

// Client is a Gemini chat client implementing [ai.ChatModel].
type Client struct {
	apiKey     string
	baseURL    string
	model      string
	httpClient *http.Client
}

// Option configures a [Client].
type Option func(*Client)

// WithModel sets the default model (for example "gemini-2.0-flash") used when a
// request does not specify one.
func WithModel(model string) Option { return func(c *Client) { c.model = model } }

// WithBaseURL overrides the API base URL.
func WithBaseURL(url string) Option { return func(c *Client) { c.baseURL = url } }

// WithHTTPClient sets the underlying HTTP client. The library performs no
// retries of its own (design principle P9); configure them here.
func WithHTTPClient(hc *http.Client) Option { return func(c *Client) { c.httpClient = hc } }

// New constructs a Client. The apiKey is sent in the x-goog-api-key header.
func New(apiKey string, opts ...Option) *Client {
	c := &Client{apiKey: apiKey, baseURL: defaultBaseURL, httpClient: http.DefaultClient}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Name returns the default model identifier, or "gemini" if none is set.
func (c *Client) Name() string {
	if c.model != "" {
		return c.model
	}
	return "gemini"
}

func (c *Client) post(ctx context.Context, path string, body any) (*http.Response, error) {
	b, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("gemini: marshaling request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("content-type", "application/json")
	req.Header.Set("x-goog-api-key", c.apiKey)
	return c.httpClient.Do(req)
}

func (c *Client) resolveModel(req ai.Request) (string, error) {
	if req.Model != "" {
		return req.Model, nil
	}
	if c.model != "" {
		return c.model, nil
	}
	return "", fmt.Errorf("gemini: no model set on request or client")
}

// Generate implements [ai.ChatModel].
func (c *Client) Generate(ctx context.Context, req ai.Request) (ai.Response, error) {
	model, err := c.resolveModel(req)
	if err != nil {
		return ai.Response{}, err
	}
	w, err := toWire(req)
	if err != nil {
		return ai.Response{}, err
	}
	resp, err := c.post(ctx, "/models/"+model+":generateContent", w)
	if err != nil {
		return ai.Response{}, fmt.Errorf("gemini: request failed: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return ai.Response{}, fmt.Errorf("gemini: reading response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return ai.Response{}, parseError(resp, raw)
	}
	var wr gResponse
	if err := json.Unmarshal(raw, &wr); err != nil {
		return ai.Response{}, fmt.Errorf("gemini: decoding response: %w", err)
	}
	return fromWire(wr, model, raw)
}

// Stream implements [ai.ChatModel]. Setup errors are delivered as the first
// (Chunk, error) pair.
func (c *Client) Stream(ctx context.Context, req ai.Request) ai.Stream {
	return func(yield func(ai.Chunk, error) bool) {
		model, err := c.resolveModel(req)
		if err != nil {
			yield(ai.Chunk{}, err)
			return
		}
		w, err := toWire(req)
		if err != nil {
			yield(ai.Chunk{}, err)
			return
		}
		resp, err := c.post(ctx, "/models/"+model+":streamGenerateContent?alt=sse", w)
		if err != nil {
			yield(ai.Chunk{}, fmt.Errorf("gemini: request failed: %w", err))
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			raw, _ := io.ReadAll(resp.Body)
			yield(ai.Chunk{}, parseError(resp, raw))
			return
		}
		parseSSE(resp.Body, yield)
	}
}

var _ ai.ChatModel = (*Client)(nil)
