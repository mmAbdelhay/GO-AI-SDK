package openai_test

import (
	"context"
	"errors"
	"net/http"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/aitest"
	"github.com/mmabdelhay/go-ai-sdk/provider/openai"
)

func TestEmbedOffline(t *testing.T) {
	tr := &aitest.Transport{Responses: []*http.Response{
		aitest.NewResponse(200, nil, `{
			"data":[{"embedding":[0.1,0.2]},{"embedding":[0.3,0.4]}],
			"model":"text-embedding-3-small",
			"usage":{"prompt_tokens":6}}`),
	}}
	e := openai.NewEmbedder("k", "text-embedding-3-small", openai.WithHTTPClient(tr.Client()))

	res, err := e.Embed(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Vectors) != 2 || res.Vectors[1][0] != 0.3 {
		t.Errorf("vectors = %+v", res.Vectors)
	}
	if res.Model != "text-embedding-3-small" || res.Usage.InputTokens != 6 {
		t.Errorf("meta = %+v", res)
	}
}

func TestEmbedCountMismatch(t *testing.T) {
	tr := &aitest.Transport{Responses: []*http.Response{
		aitest.NewResponse(200, nil, `{"data":[{"embedding":[0.1]}]}`),
	}}
	e := openai.NewEmbedder("k", "m", openai.WithHTTPClient(tr.Client()))
	if _, err := e.Embed(context.Background(), []string{"a", "b"}); err == nil {
		t.Fatal("expected count-mismatch error")
	}
}

func TestEmbedErrorNormalized(t *testing.T) {
	tr := &aitest.Transport{Responses: []*http.Response{
		aitest.NewResponse(429, nil, `{"error":{"message":"slow","type":"rate_limit_error"}}`),
	}}
	e := openai.NewEmbedder("k", "m", openai.WithHTTPClient(tr.Client()))
	if _, err := e.Embed(context.Background(), []string{"a"}); !errors.Is(err, ai.ErrRateLimited) {
		t.Fatalf("err = %v, want ErrRateLimited", err)
	}
}
