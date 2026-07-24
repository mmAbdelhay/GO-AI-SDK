package sqlite_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/agent"
	"github.com/mmabdelhay/go-ai-sdk/aitest"
	"github.com/mmabdelhay/go-ai-sdk/store/sqlite"
	"github.com/mmabdelhay/go-ai-sdk/vector"
)

func openStore(t *testing.T) *sqlite.Store {
	t.Helper()
	s, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func TestMemoryStore(t *testing.T) {
	ctx := context.Background()
	s := openStore(t)

	msgs, err := s.Messages(ctx, "none")
	if err != nil || len(msgs) != 0 {
		t.Fatalf("missing conversation: %v, %v", msgs, err)
	}

	_ = s.Append(ctx, "c1", ai.UserText("one"))
	_ = s.Append(ctx, "c1", ai.AssistantText("two"), ai.UserText("three"))
	msgs, err = s.Messages(ctx, "c1")
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 3 || msgs[1].Role != ai.RoleAssistant || msgs[2].Content[0].(ai.Text).Text != "three" {
		t.Errorf("messages = %+v", msgs)
	}

	_ = s.Clear(ctx, "c1")
	if msgs, _ := s.Messages(ctx, "c1"); len(msgs) != 0 {
		t.Errorf("clear failed: %+v", msgs)
	}
}

func TestMemoryStoreRoundTripsRichContent(t *testing.T) {
	ctx := context.Background()
	s := openStore(t)

	orig := ai.Message{Role: ai.RoleAssistant, Content: []ai.Part{
		ai.Text{Text: "here"},
		ai.ToolCall{ID: "c1", Name: "lookup", Input: json.RawMessage(`{"q":"x"}`)},
	}}
	if err := s.Append(ctx, "c", orig); err != nil {
		t.Fatal(err)
	}
	got, err := s.Messages(ctx, "c")
	if err != nil {
		t.Fatal(err)
	}
	tc, ok := got[0].Content[1].(ai.ToolCall)
	if !ok || tc.Name != "lookup" || string(tc.Input) != `{"q":"x"}` {
		t.Errorf("tool call did not round trip: %+v", got[0].Content)
	}
}

func seed(t *testing.T) (*sqlite.Store, *aitest.Embedder) {
	t.Helper()
	ctx := context.Background()
	s := openStore(t)
	emb := &aitest.Embedder{}
	docs := map[string]struct{ text, topic string }{
		"d1": {"go has goroutines for concurrency", "go"},
		"d2": {"goroutines make concurrency simple", "go"},
		"d3": {"paris is the capital of france", "travel"},
	}
	for id, d := range docs {
		res, _ := emb.Embed(ctx, []string{d.text})
		err := s.Upsert(ctx, res.Model, vector.Document{
			ID: id, Text: d.text, Embedding: res.Vectors[0],
			Metadata: map[string]string{"topic": d.topic},
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	return s, emb
}

func TestVectorQuery(t *testing.T) {
	ctx := context.Background()
	s, emb := seed(t)

	res, _ := emb.Embed(ctx, []string{"concurrency with goroutines in go"})
	matches, err := s.Query(ctx, vector.Query{Embedding: res.Vectors[0], Model: res.Model, TopK: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 2 {
		t.Fatalf("matches = %d, want 2", len(matches))
	}
	for _, m := range matches {
		if m.Document.Metadata["topic"] != "go" {
			t.Errorf("irrelevant doc %q in top 2", m.Document.ID)
		}
	}
	if matches[0].Score < matches[1].Score {
		t.Errorf("not sorted by score")
	}
}

func TestVectorMetadataFilter(t *testing.T) {
	ctx := context.Background()
	s, emb := seed(t)
	res, _ := emb.Embed(ctx, []string{"anything"})
	matches, err := s.Query(ctx, vector.Query{
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

func TestVectorRefusesModelMismatch(t *testing.T) {
	ctx := context.Background()
	s, emb := seed(t)
	res, _ := emb.Embed(ctx, []string{"q"})
	_, err := s.Query(ctx, vector.Query{Embedding: res.Vectors[0], Model: "other-model"})
	if !errors.Is(err, vector.ErrModelMismatch) {
		t.Fatalf("err = %v, want ErrModelMismatch", err)
	}
}

func TestVectorDimensionMismatch(t *testing.T) {
	ctx := context.Background()
	s, _ := seed(t)
	_, err := s.Query(ctx, vector.Query{Embedding: []float32{1, 2}, Model: "fake-embedder"})
	if !errors.Is(err, vector.ErrDimensionMismatch) {
		t.Fatalf("err = %v, want ErrDimensionMismatch", err)
	}
}

func TestVectorUpsertReplaceAndDelete(t *testing.T) {
	ctx := context.Background()
	s, emb := seed(t)

	res, _ := emb.Embed(ctx, []string{"replacement"})
	if err := s.Upsert(ctx, res.Model, vector.Document{ID: "d1", Text: "replacement", Embedding: res.Vectors[0]}); err != nil {
		t.Fatal(err)
	}
	// Replaced text should be retrievable.
	matches, _ := s.Query(ctx, vector.Query{Embedding: res.Vectors[0], Model: res.Model, TopK: 5})
	var found string
	for _, m := range matches {
		if m.Document.ID == "d1" {
			found = m.Document.Text
		}
	}
	if found != "replacement" {
		t.Errorf("upsert did not replace: %q", found)
	}

	if err := s.Delete(ctx, "d3", "missing"); err != nil {
		t.Fatal(err)
	}
	all, _ := s.Query(ctx, vector.Query{Embedding: res.Vectors[0], Model: res.Model, TopK: 10})
	for _, m := range all {
		if m.Document.ID == "d3" {
			t.Errorf("d3 not deleted")
		}
	}
}

// TestAgentWithSQLiteMemory proves the store is usable as agent memory,
// end to end, with no network.
func TestAgentWithSQLiteMemory(t *testing.T) {
	ctx := context.Background()
	store := openStore(t)

	calls := 0
	model := &aitest.Model{GenerateFunc: func(_ context.Context, req ai.Request) (ai.Response, error) {
		calls++
		if calls == 1 {
			return aitest.TextResponse("Nice to meet you, Ada."), nil
		}
		for _, m := range req.Messages {
			for _, p := range m.Content {
				if txt, ok := p.(ai.Text); ok && txt.Text == "My name is Ada." {
					return aitest.TextResponse("Your name is Ada."), nil
				}
			}
		}
		return aitest.TextResponse("I forget."), nil
	}}

	a := agent.New(model, agent.WithMemory(store, "conv-1"))
	if _, err := a.Run(ctx, "My name is Ada."); err != nil {
		t.Fatal(err)
	}
	res, err := a.Run(ctx, "What is my name?")
	if err != nil {
		t.Fatal(err)
	}
	if res.Output != "Your name is Ada." {
		t.Errorf("agent lost memory across runs: %q", res.Output)
	}
}
