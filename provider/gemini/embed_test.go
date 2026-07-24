package gemini_test

import (
	"context"
	"net/http"
	"testing"

	"github.com/mmabdelhay/go-ai-sdk/aitest"
	"github.com/mmabdelhay/go-ai-sdk/provider/gemini"
)

func TestEmbedOffline(t *testing.T) {
	tr := &aitest.Transport{Responses: []*http.Response{
		aitest.NewResponse(200, nil, `{"embeddings":[{"values":[0.1,0.2]},{"values":[0.3,0.4]}]}`),
	}}
	e := gemini.NewEmbedder("k", "text-embedding-004", gemini.WithHTTPClient(tr.Client()))

	res, err := e.Embed(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Vectors) != 2 || res.Vectors[0][1] != 0.2 {
		t.Errorf("vectors = %+v", res.Vectors)
	}
	if res.Model != "text-embedding-004" {
		t.Errorf("model = %q", res.Model)
	}
}
