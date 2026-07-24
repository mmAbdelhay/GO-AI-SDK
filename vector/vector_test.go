package vector_test

import (
	"context"
	"errors"
	"testing"

	"github.com/mmabdelhay/go-ai-sdk/aitest"
	"github.com/mmabdelhay/go-ai-sdk/vector"
)

func seed(t *testing.T) (*vector.InMemory, *aitest.Embedder) {
	t.Helper()
	ctx := context.Background()
	emb := &aitest.Embedder{}
	store := vector.NewInMemory()

	texts := map[string]struct{ text, topic string }{
		"d1": {"the go programming language has goroutines", "go"},
		"d2": {"goroutines make concurrency in go simple", "go"},
		"d3": {"paris is the capital of france", "travel"},
	}
	for id, d := range texts {
		res, err := emb.Embed(ctx, []string{d.text})
		if err != nil {
			t.Fatal(err)
		}
		err = store.Upsert(ctx, res.Model, vector.Document{
			ID: id, Text: d.text, Embedding: res.Vectors[0],
			Metadata: map[string]string{"topic": d.topic},
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	return store, emb
}

func TestQueryRanksRelevantFirst(t *testing.T) {
	ctx := context.Background()
	store, emb := seed(t)

	res, _ := emb.Embed(ctx, []string{"goroutines and concurrency in go"})
	matches, err := store.Query(ctx, vector.Query{
		Embedding: res.Vectors[0], Model: res.Model, TopK: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 2 {
		t.Fatalf("matches = %d, want 2", len(matches))
	}
	for _, m := range matches {
		if m.Document.Metadata["topic"] != "go" {
			t.Errorf("irrelevant doc ranked in top 2: %+v", m.Document.ID)
		}
	}
	if matches[0].Score < matches[1].Score {
		t.Errorf("matches not sorted by score")
	}
}

func TestQueryMetadataFilter(t *testing.T) {
	ctx := context.Background()
	store, emb := seed(t)

	res, _ := emb.Embed(ctx, []string{"anything"})
	matches, err := store.Query(ctx, vector.Query{
		Embedding: res.Vectors[0], Model: res.Model, TopK: 10,
		Filters: []vector.Filter{vector.Where("topic", "travel")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 1 || matches[0].Document.ID != "d3" {
		t.Errorf("filter failed: %+v", matches)
	}
}

func TestQueryRefusesModelMismatch(t *testing.T) {
	ctx := context.Background()
	store, emb := seed(t)

	res, _ := emb.Embed(ctx, []string{"query"})
	_, err := store.Query(ctx, vector.Query{
		Embedding: res.Vectors[0], Model: "some-other-model",
	})
	if !errors.Is(err, vector.ErrModelMismatch) {
		t.Fatalf("err = %v, want ErrModelMismatch", err)
	}
}

func TestQueryDimensionMismatch(t *testing.T) {
	ctx := context.Background()
	store, _ := seed(t)

	_, err := store.Query(ctx, vector.Query{
		Embedding: []float32{1, 2}, Model: "fake-embedder",
	})
	if !errors.Is(err, vector.ErrDimensionMismatch) {
		t.Fatalf("err = %v, want ErrDimensionMismatch", err)
	}
}

func TestUpsertValidation(t *testing.T) {
	ctx := context.Background()
	s := vector.NewInMemory()
	if err := s.Upsert(ctx, "", vector.Document{ID: "x", Embedding: []float32{1}}); err == nil {
		t.Error("empty model accepted")
	}
	if err := s.Upsert(ctx, "m", vector.Document{Embedding: []float32{1}}); err == nil {
		t.Error("empty ID accepted")
	}
	if err := s.Upsert(ctx, "m", vector.Document{ID: "x"}); err == nil {
		t.Error("empty embedding accepted")
	}
}

func TestDeleteAndReplace(t *testing.T) {
	ctx := context.Background()
	store, emb := seed(t)

	if err := store.Delete(ctx, "d3", "missing-id"); err != nil {
		t.Fatal(err)
	}
	if store.Len() != 2 {
		t.Errorf("len = %d, want 2", store.Len())
	}

	// Upsert with an existing ID replaces.
	res, _ := emb.Embed(ctx, []string{"replacement text"})
	_ = store.Upsert(ctx, res.Model, vector.Document{ID: "d1", Text: "replacement text", Embedding: res.Vectors[0]})
	if store.Len() != 2 {
		t.Errorf("replace changed count: %d", store.Len())
	}
}
