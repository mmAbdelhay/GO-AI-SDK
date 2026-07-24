package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

const (
	defaultBaseURL = "https://api.anthropic.com"
	defaultVersion = "2023-06-01"
	// defaultMaxTokens is applied when neither the request nor the client sets a
	// limit. Anthropic requires max_tokens, so a value is always sent; this
	// default is documented rather than hidden.
	defaultMaxTokens = 1024
)

// Client is an Anthropic Messages API client implementing [ai.ChatModel]. It
// depends only on the standard library.
type Client struct {
	apiKey           string
	baseURL          string
	version          string
	defaultModel     string
	defaultMaxTokens int
	httpClient       *http.Client
}

// Option configures a [Client].
type Option func(*Client)

// WithModel sets the default model used when a request does not specify one.
func WithModel(model string) Option {
	return func(c *Client) { c.defaultModel = model }
}

// WithBaseURL overrides the API base URL (useful for proxies and testing).
func WithBaseURL(url string) Option {
	return func(c *Client) { c.baseURL = url }
}

// WithVersion overrides the anthropic-version header.
func WithVersion(v string) Option {
	return func(c *Client) { c.version = v }
}

// WithHTTPClient sets the underlying HTTP client. Supply a custom client to
// control timeouts, transports, and (in tests) a fake round tripper. The library
// performs no retries of its own (design principle P9); configure them here.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) { c.httpClient = hc }
}

// WithDefaultMaxTokens sets the max_tokens applied when a request leaves it zero.
func WithDefaultMaxTokens(n int) Option {
	return func(c *Client) { c.defaultMaxTokens = n }
}

// New constructs a Client. The apiKey is sent in the x-api-key header.
func New(apiKey string, opts ...Option) *Client {
	c := &Client{
		apiKey:           apiKey,
		baseURL:          defaultBaseURL,
		version:          defaultVersion,
		defaultMaxTokens: defaultMaxTokens,
		httpClient:       http.DefaultClient,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Name returns the default model identifier, or "anthropic" if none is set.
func (c *Client) Name() string {
	if c.defaultModel != "" {
		return c.defaultModel
	}
	return "anthropic"
}

// newHTTPRequest builds an *http.Request for the Messages endpoint with the
// required headers and the marshaled body.
func (c *Client) newHTTPRequest(ctx context.Context, w wireRequest) (*http.Request, error) {
	body, err := json.Marshal(w)
	if err != nil {
		return nil, fmt.Errorf("anthropic: marshaling request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("content-type", "application/json")
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", c.version)
	return req, nil
}

// Generate implements [ai.ChatModel].
func (c *Client) Generate(ctx context.Context, req ai.Request) (ai.Response, error) {
	w, err := c.toWire(req, false)
	if err != nil {
		return ai.Response{}, err
	}
	httpReq, err := c.newHTTPRequest(ctx, w)
	if err != nil {
		return ai.Response{}, err
	}
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return ai.Response{}, fmt.Errorf("anthropic: request failed: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return ai.Response{}, fmt.Errorf("anthropic: reading response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return ai.Response{}, parseError(resp, raw)
	}

	var wr wireResponse
	if err := json.Unmarshal(raw, &wr); err != nil {
		return ai.Response{}, fmt.Errorf("anthropic: decoding response: %w", err)
	}
	return fromWire(wr, raw)
}

// Stream implements [ai.ChatModel]. Setup errors are delivered as the first
// (Chunk, error) pair yielded by the returned [ai.Stream].
func (c *Client) Stream(ctx context.Context, req ai.Request) ai.Stream {
	return func(yield func(ai.Chunk, error) bool) {
		w, err := c.toWire(req, true)
		if err != nil {
			yield(ai.Chunk{}, err)
			return
		}
		httpReq, err := c.newHTTPRequest(ctx, w)
		if err != nil {
			yield(ai.Chunk{}, err)
			return
		}
		resp, err := c.httpClient.Do(httpReq)
		if err != nil {
			yield(ai.Chunk{}, fmt.Errorf("anthropic: request failed: %w", err))
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

// compile-time assertion that Client satisfies the interface.
var _ ai.ChatModel = (*Client)(nil)
