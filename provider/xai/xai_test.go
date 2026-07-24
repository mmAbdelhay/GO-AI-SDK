package xai_test

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/aitest"
	"github.com/mmabdelhay/go-ai-sdk/provider/openai"
	"github.com/mmabdelhay/go-ai-sdk/provider/xai"
)

func TestNewTargetsXAIWithLegacyMaxTokens(t *testing.T) {
	tr := &aitest.Transport{Responses: []*http.Response{
		aitest.NewResponse(200, nil, `{
			"model":"grok-3",
			"choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1}}`),
	}}
	c := xai.New("k", openai.WithModel("grok-3"), openai.WithHTTPClient(tr.Client()))

	if _, err := ai.Generate(context.Background(), c, "hi", ai.WithMaxTokens(64)); err != nil {
		t.Fatal(err)
	}
	url, _ := tr.LastRequestURL()
	if !strings.HasPrefix(url, xai.BaseURL) {
		t.Errorf("request went to %q, want prefix %q", url, xai.BaseURL)
	}
	body, _ := tr.LastRequestBody()
	var sent map[string]any
	_ = json.Unmarshal(body, &sent)
	if sent["max_tokens"] == nil || sent["max_completion_tokens"] != nil {
		t.Errorf("xAI must use the legacy max_tokens field, sent: %s", body)
	}
}
