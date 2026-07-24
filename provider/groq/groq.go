// Package groq provides an [ai.ChatModel] for Groq's OpenAI-compatible API. It
// is a thin constructor over the openai package with Groq's base URL and
// compatibility settings applied; all openai client options are accepted.
package groq

import (
	"github.com/mmabdelhay/go-ai-sdk/provider/openai"
)

// BaseURL is Groq's OpenAI-compatible endpoint.
const BaseURL = "https://api.groq.com/openai/v1"

// New constructs a Groq chat client. Options may override any default,
// including the base URL.
func New(apiKey string, opts ...openai.Option) *openai.Client {
	defaults := []openai.Option{
		openai.WithBaseURL(BaseURL),
		openai.WithProviderName("groq"),
	}
	return openai.New(apiKey, append(defaults, opts...)...)
}
