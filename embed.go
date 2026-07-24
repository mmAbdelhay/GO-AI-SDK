package ai

import "context"

// Embedder produces vector embeddings for texts. Implementations live in
// provider packages (for example openai.NewEmbedder); a deterministic fake lives
// in aitest.
type Embedder interface {
	// Name returns the embedding model identifier. Vector stores record it so
	// vectors from different models are never mixed in one query.
	Name() string
	// Embed returns one vector per input text, in order.
	Embed(ctx context.Context, texts []string) (EmbedResult, error)
}

// EmbedResult is the outcome of an embedding call.
type EmbedResult struct {
	// Vectors holds one embedding per input text, in input order.
	Vectors [][]float32
	// Model is the embedding model that produced the vectors.
	Model string
	// Usage reports token consumption when the provider supplies it.
	Usage Usage
}
