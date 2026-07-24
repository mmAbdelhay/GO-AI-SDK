package groq_test

import (
	"context"
	"net/http"
	"strings"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/aitest"
	"github.com/mmabdelhay/go-ai-sdk/provider/groq"
	"github.com/mmabdelhay/go-ai-sdk/provider/openai"
)

func TestNewTargetsGroqEndpoint(t *testing.T) {
	tr := &aitest.Transport{Responses: []*http.Response{
		aitest.NewResponse(200, nil, `{
			"model":"llama-3.3-70b-versatile",
			"choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1}}`),
	}}
	c := groq.New("k",
		openai.WithModel("llama-3.3-70b-versatile"),
		openai.WithHTTPClient(tr.Client()))

	if _, err := ai.Generate(context.Background(), c, "hi"); err != nil {
		t.Fatal(err)
	}
	url, _ := tr.LastRequestURL()
	if !strings.HasPrefix(url, groq.BaseURL) {
		t.Errorf("request went to %q, want prefix %q", url, groq.BaseURL)
	}
}
