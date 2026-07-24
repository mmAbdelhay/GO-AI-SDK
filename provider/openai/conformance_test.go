package openai_test

import (
	"net/http"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/aitest"
	"github.com/mmabdelhay/go-ai-sdk/provider/openai"
)

func oaGenerateResponse() *http.Response {
	return aitest.NewResponse(200, nil, `{
		"model":"m",
		"choices":[{"message":{"role":"assistant","content":"Hello, world"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":9,"completion_tokens":7}}`)
}

func oaStreamResponse() *http.Response {
	return aitest.NewSSEResponse(`data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"content":", world"},"finish_reason":"stop"}]}

data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":7}}

data: [DONE]

`)
}

func oaRateLimitResponse() *http.Response {
	return aitest.NewResponse(429, nil,
		`{"error":{"message":"Rate limit reached","type":"rate_limit_error","code":"rate_limit_exceeded"}}`)
}

func TestConformance(t *testing.T) {
	aitest.RunConformance(t, aitest.Conformance{
		NewClient: func(hc *http.Client) ai.ChatModel {
			return openai.New("k", openai.WithModel("m"), openai.WithHTTPClient(hc))
		},
		GenerateResponse:  oaGenerateResponse,
		StreamResponse:    oaStreamResponse,
		RateLimitResponse: oaRateLimitResponse,
	})
}
