package gemini_test

import (
	"net/http"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/aitest"
	"github.com/mmabdelhay/go-ai-sdk/provider/gemini"
)

func TestConformance(t *testing.T) {
	aitest.RunConformance(t, aitest.Conformance{
		NewClient: func(hc *http.Client) ai.ChatModel {
			return gemini.New("k", gemini.WithModel("m"), gemini.WithHTTPClient(hc))
		},
		GenerateResponse: func() *http.Response {
			return aitest.NewResponse(200, nil, `{
				"candidates":[{"content":{"role":"model","parts":[{"text":"Hello, world"}]},"finishReason":"STOP"}],
				"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":7}}`)
		},
		StreamResponse: func() *http.Response {
			return aitest.NewSSEResponse(`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}]}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":", world"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":7}}

`)
		},
		RateLimitResponse: func() *http.Response {
			return aitest.NewResponse(429, nil,
				`{"error":{"code":429,"message":"quota exceeded","status":"RESOURCE_EXHAUSTED"}}`)
		},
	})
}
