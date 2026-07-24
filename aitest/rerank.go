package aitest

import (
	"context"
	"sort"
	"strings"

	"github.com/mmabdelhay/go-ai-sdk/rerank"
)

// Reranker is a fake [rerank.Reranker]. When Results is set it returns that
// script; otherwise it scores by naive case-insensitive term overlap between the
// query and each document, which is deterministic and needs no network.
type Reranker struct {
	// Results, when non-nil, is returned verbatim (truncated to topN).
	Results []rerank.Result
	// Err, when non-nil, is returned by every call.
	Err error
}

// Rerank implements [rerank.Reranker].
func (r *Reranker) Rerank(_ context.Context, query string, documents []string, topN int) ([]rerank.Result, error) {
	if r.Err != nil {
		return nil, r.Err
	}
	results := r.Results
	if results == nil {
		terms := strings.Fields(strings.ToLower(query))
		for i, d := range documents {
			ld := strings.ToLower(d)
			score := 0.0
			for _, t := range terms {
				if strings.Contains(ld, t) {
					score++
				}
			}
			results = append(results, rerank.Result{Index: i, Score: score})
		}
		sort.SliceStable(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	}
	if topN > 0 && len(results) > topN {
		results = results[:topN]
	}
	return results, nil
}

var _ rerank.Reranker = (*Reranker)(nil)
