// Package xai provides an [ai.ChatModel] for xAI's OpenAI-compatible API. It is
// a thin constructor over the openai package with xAI's base URL and
// compatibility settings applied; all openai client options are accepted.
package xai

import (
	"github.com/mmabdelhay/go-ai-sdk/provider/openai"
)

// BaseURL is xAI's OpenAI-compatible endpoint.
const BaseURL = "https://api.x.ai/v1"

// New constructs an xAI chat client. Options may override any default,
// including the base URL.
func New(apiKey string, opts ...openai.Option) *openai.Client {
	defaults := []openai.Option{
		openai.WithBaseURL(BaseURL),
		openai.WithProviderName("xai"),
		// xAI documents the legacy max_tokens field.
		openai.WithLegacyMaxTokens(),
	}
	return openai.New(apiKey, append(defaults, opts...)...)
}
