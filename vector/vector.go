// Package vector defines the VectorStore interface for similarity search with
// metadata filtering, plus an in-memory implementation. Server-backed stores
// (pgvector, ...) live in separate submodules.
//
// Every upsert records the embedding model that produced the vectors, and every
// query names the model it embedded with; the store refuses to answer a query
// whose model does not match the stored vectors, because cross-model cosine
// similarity is meaningless.
package vector

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"
)

// Errors returned by stores.
var (
	// ErrModelMismatch means the query's embedding model does not match the
	// model the stored vectors were produced with.
	ErrModelMismatch = errors.New("vector: query embedding model does not match stored vectors")
	// ErrDimensionMismatch means a vector's width differs from others in the
	// same store or query.
	ErrDimensionMismatch = errors.New("vector: embedding dimensions do not match")
)

// Document is a stored item: text, metadata, and its embedding.
type Document struct {
	ID       string
	Text     string
	Metadata map[string]string
	// Embedding is the vector for Text, produced by the model named at upsert.
	Embedding []float32
}

// Filter is an equality condition on document metadata.
type Filter struct {
	Key   string
	Value string
}

// Where builds a metadata equality filter.
func Where(key, value string) Filter { return Filter{Key: key, Value: value} }

// Query is a similarity search request.
type Query struct {
	// Embedding is the query vector.
	Embedding []float32
	// Model names the embedding model that produced Embedding. Required: the
	// store rejects queries against vectors from a different model.
	Model string
	// TopK is the maximum number of matches (default 5).
	TopK int
	// Filters restricts candidates to documents matching all conditions.
	Filters []Filter
}

// Match is a query hit.
type Match struct {
	Document Document
	// Score is cosine similarity in [-1, 1], higher is more similar.
	Score float32
}

// Store is the vector store contract.
type Store interface {
	// Upsert inserts or replaces documents whose embeddings were produced by
	// the named model.
	Upsert(ctx context.Context, model string, docs ...Document) error
	// Query returns the most similar documents.
	Query(ctx context.Context, q Query) ([]Match, error)
	// Delete removes documents by ID. Missing IDs are not an error.
	Delete(ctx context.Context, ids ...string) error
}

type entry struct {
	doc   Document
	model string
}

// InMemory is a Store backed by process memory with exact (brute-force) cosine
// search. Safe for concurrent use.
type InMemory struct {
	mu      sync.RWMutex
	entries map[string]entry
}

// NewInMemory returns an empty in-memory store.
func NewInMemory() *InMemory {
	return &InMemory{entries: make(map[string]entry)}
}

// Upsert implements [Store].
func (s *InMemory) Upsert(_ context.Context, model string, docs ...Document) error {
	if model == "" {
		return fmt.Errorf("vector: embedding model is required on upsert")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, d := range docs {
		if d.ID == "" {
			return fmt.Errorf("vector: document ID is required")
		}
		if len(d.Embedding) == 0 {
			return fmt.Errorf("vector: document %q has no embedding", d.ID)
		}
		s.entries[d.ID] = entry{doc: d, model: model}
	}
	return nil
}

// Query implements [Store].
func (s *InMemory) Query(_ context.Context, q Query) ([]Match, error) {
	if q.Model == "" {
		return nil, fmt.Errorf("vector: query model is required")
	}
	if len(q.Embedding) == 0 {
		return nil, fmt.Errorf("vector: query embedding is required")
	}
	topK := q.TopK
	if topK <= 0 {
		topK = 5
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	var matches []Match
	otherModels := map[string]bool{}
	for _, e := range s.entries {
		if e.model != q.Model {
			otherModels[e.model] = true
			continue
		}
		if !matchesFilters(e.doc, q.Filters) {
			continue
		}
		if len(e.doc.Embedding) != len(q.Embedding) {
			return nil, fmt.Errorf("%w: query %d vs stored %d (doc %q)",
				ErrDimensionMismatch, len(q.Embedding), len(e.doc.Embedding), e.doc.ID)
		}
		matches = append(matches, Match{Document: e.doc, Score: cosine(q.Embedding, e.doc.Embedding)})
	}
	// A query against a store that only holds other models' vectors is a
	// mistake we refuse to hide behind an empty result.
	if len(matches) == 0 && len(otherModels) > 0 && !s.hasModelLocked(q.Model) {
		models := make([]string, 0, len(otherModels))
		for m := range otherModels {
			models = append(models, m)
		}
		sort.Strings(models)
		return nil, fmt.Errorf("%w: query model %q, stored %v", ErrModelMismatch, q.Model, models)
	}

	sort.Slice(matches, func(i, j int) bool { return matches[i].Score > matches[j].Score })
	if len(matches) > topK {
		matches = matches[:topK]
	}
	return matches, nil
}

func (s *InMemory) hasModelLocked(model string) bool {
	for _, e := range s.entries {
		if e.model == model {
			return true
		}
	}
	return false
}

// Delete implements [Store].
func (s *InMemory) Delete(_ context.Context, ids ...string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, id := range ids {
		delete(s.entries, id)
	}
	return nil
}

// Len returns the number of stored documents.
func (s *InMemory) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.entries)
}

func matchesFilters(d Document, filters []Filter) bool {
	for _, f := range filters {
		if d.Metadata[f.Key] != f.Value {
			return false
		}
	}
	return true
}

func cosine(a, b []float32) float32 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}

var _ Store = (*InMemory)(nil)
