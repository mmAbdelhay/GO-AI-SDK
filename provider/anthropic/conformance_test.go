package anthropic_test

import (
	"net/http"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/aitest"
	"github.com/mmabdelhay/go-ai-sdk/provider/anthropic"
)

func TestConformance(t *testing.T) {
	aitest.RunConformance(t, aitest.Conformance{
		NewClient: func(hc *http.Client) ai.ChatModel {
			return anthropic.New("k", anthropic.WithModel("m"), anthropic.WithHTTPClient(hc))
		},
		GenerateResponse: func() *http.Response {
			return aitest.NewResponse(200, nil, `{
				"type":"message","role":"assistant","model":"m",
				"content":[{"type":"text","text":"Hello, world"}],
				"stop_reason":"end_turn",
				"usage":{"input_tokens":9,"output_tokens":7}}`)
		},
		StreamResponse: func() *http.Response {
			return aitest.NewSSEResponse(`event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":9,"output_tokens":1}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":", world"}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}

event: message_stop
data: {"type":"message_stop"}

`)
		},
		RateLimitResponse: func() *http.Response {
			return aitest.NewResponse(429, nil,
				`{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`)
		},
	})
}
