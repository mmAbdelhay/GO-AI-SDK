// Package middleware provides composable wrappers around [ai.ChatModel]:
// structured logging via log/slog, before/after hooks, and token/cost
// accounting. The OpenTelemetry integration lives in a separate submodule (per
// design principle P2) and builds on the same [Hooks] mechanism.
package middleware

import (
	"context"
	"time"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// Middleware wraps a ChatModel with additional behavior.
type Middleware func(ai.ChatModel) ai.ChatModel

// Chain applies middlewares to model. The first middleware becomes the
// outermost wrapper:
//
//	m := middleware.Chain(client, middleware.Logging(logger), tracker.Middleware())
func Chain(model ai.ChatModel, mws ...Middleware) ai.ChatModel {
	for i := len(mws) - 1; i >= 0; i-- {
		model = mws[i](model)
	}
	return model
}

// Hooks observes every model call. Any field may be nil.
type Hooks struct {
	// Before runs before each Generate or Stream call. It may return a derived
	// context (return ctx unchanged otherwise).
	Before func(ctx context.Context, req ai.Request) context.Context
	// After runs after each Generate call, and after each Stream completes,
	// with the outcome. For streams, resp carries the collected usage and stop
	// reason (not the text). Duration covers the full call.
	After func(ctx context.Context, req ai.Request, resp ai.Response, err error, d time.Duration)
	// OnChunk runs for every streamed chunk.
	OnChunk func(chunk ai.Chunk)
}

type hookedModel struct {
	inner ai.ChatModel
	h     Hooks
}

// WithHooks returns a middleware invoking h around every call.
func WithHooks(h Hooks) Middleware {
	return func(m ai.ChatModel) ai.ChatModel { return &hookedModel{inner: m, h: h} }
}

func (m *hookedModel) Name() string { return m.inner.Name() }

func (m *hookedModel) Generate(ctx context.Context, req ai.Request) (ai.Response, error) {
	if m.h.Before != nil {
		ctx = m.h.Before(ctx, req)
	}
	start := time.Now()
	resp, err := m.inner.Generate(ctx, req)
	if m.h.After != nil {
		m.h.After(ctx, req, resp, err, time.Since(start))
	}
	return resp, err
}

func (m *hookedModel) Stream(ctx context.Context, req ai.Request) ai.Stream {
	return func(yield func(ai.Chunk, error) bool) {
		if m.h.Before != nil {
			ctx = m.h.Before(ctx, req)
		}
		start := time.Now()
		var (
			final   ai.Response
			failure error
		)
		m.inner.Stream(ctx, req)(func(c ai.Chunk, err error) bool {
			if err != nil {
				failure = err
			} else {
				if m.h.OnChunk != nil {
					m.h.OnChunk(c)
				}
				if c.Type == ai.ChunkDone {
					if c.Usage != nil {
						final.Usage = *c.Usage
					}
					final.StopReason = c.StopReason
				}
			}
			return yield(c, err)
		})
		if m.h.After != nil {
			m.h.After(ctx, req, final, failure, time.Since(start))
		}
	}
}
