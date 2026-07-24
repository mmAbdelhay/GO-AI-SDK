// Package rerank defines the Reranker interface for reordering retrieved
// documents by relevance to a query. Provider-backed implementations plug in
// behind this interface; a scripted fake lives in aitest.
package rerank

import "context"

// Result is one reranked document.
type Result struct {
	// Index is the document's position in the input slice.
	Index int
	// Score is the relevance score; higher is more relevant.
	Score float64
}

// Reranker reorders documents by relevance to a query.
type Reranker interface {
	// Rerank scores documents against query and returns the topN most relevant
	// (all of them when topN <= 0), ordered by descending score.
	Rerank(ctx context.Context, query string, documents []string, topN int) ([]Result, error)
}
