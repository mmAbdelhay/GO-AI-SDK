package middleware

import (
	"context"
	"sync"
	"time"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// ModelPrice is the cost of one million tokens for a model.
type ModelPrice struct {
	InputPerMTok  float64
	OutputPerMTok float64
}

// Pricing maps model identifiers to prices. Callers own the table; the library
// ships no price list, because prices change faster than releases.
type Pricing map[string]ModelPrice

// UsageTracker accumulates token usage per model across calls. Attach it with
// [UsageTracker.Middleware]. Safe for concurrent use.
type UsageTracker struct {
	mu       sync.Mutex
	perModel map[string]ai.Usage
	calls    int
}

// NewUsageTracker returns an empty tracker.
func NewUsageTracker() *UsageTracker {
	return &UsageTracker{perModel: make(map[string]ai.Usage)}
}

// Middleware returns the middleware recording into the tracker.
func (t *UsageTracker) Middleware() Middleware {
	return func(next ai.ChatModel) ai.ChatModel {
		name := next.Name()
		return WithHooks(Hooks{
			After: func(_ context.Context, req ai.Request, resp ai.Response, err error, _ time.Duration) {
				if err != nil {
					return
				}
				model := resp.Model
				if model == "" {
					model = req.Model
				}
				if model == "" {
					model = name
				}
				t.mu.Lock()
				defer t.mu.Unlock()
				u := t.perModel[model]
				u.InputTokens += resp.Usage.InputTokens
				u.OutputTokens += resp.Usage.OutputTokens
				u.CacheCreationTokens += resp.Usage.CacheCreationTokens
				u.CacheReadTokens += resp.Usage.CacheReadTokens
				t.perModel[model] = u
				t.calls++
			},
		})(next)
	}
}

// Total returns aggregate usage across all models.
func (t *UsageTracker) Total() ai.Usage {
	t.mu.Lock()
	defer t.mu.Unlock()
	var total ai.Usage
	for _, u := range t.perModel {
		total.InputTokens += u.InputTokens
		total.OutputTokens += u.OutputTokens
		total.CacheCreationTokens += u.CacheCreationTokens
		total.CacheReadTokens += u.CacheReadTokens
	}
	return total
}

// PerModel returns a copy of usage keyed by model.
func (t *UsageTracker) PerModel() map[string]ai.Usage {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make(map[string]ai.Usage, len(t.perModel))
	for k, v := range t.perModel {
		out[k] = v
	}
	return out
}

// Calls returns how many successful calls were recorded.
func (t *UsageTracker) Calls() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.calls
}

// Cost prices the tracked usage with the given table. Models absent from the
// table contribute zero; check coverage with your own table before relying on
// the number.
func (t *UsageTracker) Cost(p Pricing) float64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	var cost float64
	for model, u := range t.perModel {
		price, ok := p[model]
		if !ok {
			continue
		}
		cost += float64(u.InputTokens)/1e6*price.InputPerMTok +
			float64(u.OutputTokens)/1e6*price.OutputPerMTok
	}
	return cost
}
