package postgres_test

import (
	"context"
	"errors"
	"os"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/store/postgres"
	"github.com/mmabdelhay/go-ai-sdk/vector"
)

// These tests hit a real PostgreSQL with pgvector. They run only when
// TEST_POSTGRES_DSN is set (no database is available in CI/sandbox by default),
// exactly like the provider integration tests. The DSN's database must have, or
// permit creating, the pgvector extension.
//
//	TEST_POSTGRES_DSN='postgres://user:pass@localhost:5432/aitest' \
//	    go test ./store/postgres
func testStore(t *testing.T) *postgres.Store {
	t.Helper()
	dsn := os.Getenv("TEST_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("TEST_POSTGRES_DSN not set")
	}
	ctx := context.Background()
	s, err := postgres.Open(ctx, dsn, postgres.WithDimension(8))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Migrate(ctx); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	return s
}

func TestMemoryStore(t *testing.T) {
	ctx := context.Background()
	s := testStore(t)
	id := "conv-test"
	_ = s.Clear(ctx, id)

	if err := s.Append(ctx, id, ai.UserText("one"), ai.AssistantText("two")); err != nil {
		t.Fatal(err)
	}
	msgs, err := s.Messages(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 2 || msgs[1].Role != ai.RoleAssistant {
		t.Errorf("messages = %+v", msgs)
	}
	_ = s.Clear(ctx, id)
}

func TestVectorStore(t *testing.T) {
	ctx := context.Background()
	s := testStore(t)
	_ = s.Delete(ctx, "d1", "d2", "d3")

	model := "test-embed-8"
	docs := []vector.Document{
		{ID: "d1", Text: "go concurrency", Metadata: map[string]string{"topic": "go"},
			Embedding: []float32{1, 1, 0, 0, 0, 0, 0, 0}},
		{ID: "d2", Text: "python data", Metadata: map[string]string{"topic": "py"},
			Embedding: []float32{0, 0, 1, 1, 0, 0, 0, 0}},
	}
	if err := s.Upsert(ctx, model, docs...); err != nil {
		t.Fatal(err)
	}

	matches, err := s.Query(ctx, vector.Query{
		Embedding: []float32{1, 1, 0, 0, 0, 0, 0, 0}, Model: model, TopK: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 1 || matches[0].Document.ID != "d1" {
		t.Errorf("nearest = %+v", matches)
	}

	// Model mismatch is refused.
	if _, err := s.Query(ctx, vector.Query{
		Embedding: []float32{1, 1, 0, 0, 0, 0, 0, 0}, Model: "other",
	}); !errors.Is(err, vector.ErrModelMismatch) {
		t.Errorf("err = %v, want ErrModelMismatch", err)
	}
	_ = s.Delete(ctx, "d1", "d2")
}

// TestDimensionMismatchNoDB verifies the dimension guard without a database:
// the check happens before any query is issued.
func TestDimensionMismatchNoDB(t *testing.T) {
	s := postgres.FromPool(nil, postgres.WithDimension(8))
	_, err := s.Query(context.Background(), vector.Query{
		Embedding: []float32{1, 2, 3}, Model: "m",
	})
	if !errors.Is(err, vector.ErrDimensionMismatch) {
		t.Fatalf("err = %v, want ErrDimensionMismatch", err)
	}
}
