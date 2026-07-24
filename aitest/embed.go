package aitest

import (
	"context"
	"hash/fnv"
	"math"
	"sync"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// Embedder is a deterministic fake [ai.Embedder]: the same text always produces
// the same unit-length vector, and similar usage requires no network. It records
// every call. Safe for concurrent use.
type Embedder struct {
	// ModelName is returned by Name (default "fake-embedder").
	ModelName string
	// Dimensions is the vector width (default 8).
	Dimensions int
	// Err, when non-nil, is returned by every Embed call.
	Err error

	mu    sync.Mutex
	calls [][]string
}

// Name implements [ai.Embedder].
func (e *Embedder) Name() string {
	if e.ModelName != "" {
		return e.ModelName
	}
	return "fake-embedder"
}

// Embed implements [ai.Embedder] with deterministic hash-derived vectors.
func (e *Embedder) Embed(_ context.Context, texts []string) (ai.EmbedResult, error) {
	e.mu.Lock()
	e.calls = append(e.calls, append([]string(nil), texts...))
	e.mu.Unlock()
	if e.Err != nil {
		return ai.EmbedResult{}, e.Err
	}
	dim := e.Dimensions
	if dim <= 0 {
		dim = 8
	}
	res := ai.EmbedResult{Model: e.Name()}
	for _, t := range texts {
		res.Vectors = append(res.Vectors, hashVector(t, dim))
		res.Usage.InputTokens += len(t) / 4
	}
	return res, nil
}

// Calls returns the recorded inputs of every Embed call.
func (e *Embedder) Calls() [][]string {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([][]string, len(e.calls))
	copy(out, e.calls)
	return out
}

// hashVector derives a stable unit vector from text. Overlapping 3-grams hash
// into buckets, so texts sharing words get higher cosine similarity than
// unrelated ones — enough structure to test retrieval pipelines.
func hashVector(text string, dim int) []float32 {
	v := make([]float64, dim)
	for i := 0; i+3 <= len(text); i++ {
		h := fnv.New32a()
		_, _ = h.Write([]byte(text[i : i+3]))
		v[int(h.Sum32())%dim]++
	}
	var norm float64
	for _, x := range v {
		norm += x * x
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		norm = 1
	}
	out := make([]float32, dim)
	for i, x := range v {
		out[i] = float32(x / norm)
	}
	return out
}

var _ ai.Embedder = (*Embedder)(nil)
