package anthropic_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/aitest"
	"github.com/mmabdelhay/go-ai-sdk/provider/anthropic"
)

const sampleResponseBody = `{
  "id": "msg_1", "type": "message", "role": "assistant",
  "model": "claude-sonnet-4-20250514",
  "content": [{"type": "text", "text": "Hi there!"}],
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 5, "output_tokens": 3}
}`

func TestClientGenerateOffline(t *testing.T) {
	tr := &aitest.Transport{Responses: []*http.Response{
		aitest.NewResponse(http.StatusOK, nil, sampleResponseBody),
	}}
	client := anthropic.New("test-key",
		anthropic.WithModel("claude-sonnet-4-20250514"),
		anthropic.WithHTTPClient(tr.Client()),
	)

	resp, err := ai.Generate(context.Background(), client, "Hello",
		ai.WithTemperature(0.3), ai.WithMaxTokens(64))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "Hi there!" {
		t.Errorf("text = %q", resp.Text())
	}
	if resp.Usage.InputTokens != 5 || resp.Usage.OutputTokens != 3 {
		t.Errorf("usage = %+v", resp.Usage)
	}

	// Assert the outgoing request carried the expected headers and body.
	hdr, _ := tr.LastRequestHeader()
	if hdr.Get("x-api-key") != "test-key" {
		t.Errorf("x-api-key = %q", hdr.Get("x-api-key"))
	}
	if hdr.Get("anthropic-version") == "" {
		t.Errorf("missing anthropic-version header")
	}
	body, _ := tr.LastRequestBody()
	var sent map[string]any
	if err := json.Unmarshal(body, &sent); err != nil {
		t.Fatal(err)
	}
	if sent["model"] != "claude-sonnet-4-20250514" {
		t.Errorf("sent model = %v", sent["model"])
	}
	if sent["max_tokens"].(float64) != 64 {
		t.Errorf("sent max_tokens = %v", sent["max_tokens"])
	}
	if sent["stream"] != nil && sent["stream"].(bool) {
		t.Errorf("non-stream request set stream=true")
	}
}

func TestClientGenerateErrorOffline(t *testing.T) {
	tr := &aitest.Transport{Responses: []*http.Response{
		aitest.NewResponse(http.StatusTooManyRequests, nil,
			`{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`),
	}}
	client := anthropic.New("k", anthropic.WithModel("m"), anthropic.WithHTTPClient(tr.Client()))

	_, err := ai.Generate(context.Background(), client, "Hello")
	if !errors.Is(err, ai.ErrRateLimited) {
		t.Errorf("err = %v, want ErrRateLimited", err)
	}
}

func TestClientStreamOffline(t *testing.T) {
	sse := `event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":9,"output_tokens":1}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi "}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"there"}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}

event: message_stop
data: {"type":"message_stop"}

`
	tr := &aitest.Transport{Responses: []*http.Response{aitest.NewSSEResponse(sse)}}
	client := anthropic.New("k", anthropic.WithModel("m"), anthropic.WithHTTPClient(tr.Client()))

	text, err := ai.GenerateStream(context.Background(), client, "hi").Text()
	if err != nil {
		t.Fatal(err)
	}
	if text != "Hi there" {
		t.Errorf("streamed text = %q", text)
	}

	// The streaming request must set stream=true.
	body, _ := tr.LastRequestBody()
	var sent map[string]any
	_ = json.Unmarshal(body, &sent)
	if sent["stream"] != true {
		t.Errorf("stream request did not set stream=true: %v", sent["stream"])
	}
}

func TestClientName(t *testing.T) {
	if got := anthropic.New("k", anthropic.WithModel("m")).Name(); got != "m" {
		t.Errorf("Name() = %q", got)
	}
	if got := anthropic.New("k").Name(); got != "anthropic" {
		t.Errorf("Name() default = %q", got)
	}
}
