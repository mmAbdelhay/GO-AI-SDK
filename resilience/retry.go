// Package resilience wraps any [ai.ChatModel] with retry, provider fallback,
// circuit breaking, and client-side rate limiting. Per design principle P9,
// nothing here is implicit: every behavior is constructed explicitly, fully
// configured, and observable through callbacks.
//
// The wrappers compose:
//
//	model := resilience.NewFallback(resilience.FallbackConfig{},
//		resilience.NewRetry(anthropicClient, resilience.RetryConfig{}),
//		openaiClient,
//	)
//
// Streaming semantics: a stream that fails before yielding any content can be
// retried or failed over transparently; once content has been yielded to the
// consumer, the error is delivered as-is, because silently restarting a
// half-consumed stream would duplicate output.
package resilience

import (
	"context"
	"errors"
	"math/rand/v2"
	"time"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// RetryConfig configures [NewRetry]. The zero value gives 3 attempts with
// exponential backoff from 500ms to 8s and full jitter.
type RetryConfig struct {
	// MaxAttempts is the total number of attempts, including the first
	// (default 3).
	MaxAttempts int
	// BaseDelay is the backoff before the first retry (default 500ms). Each
	// subsequent retry doubles it.
	BaseDelay time.Duration
	// MaxDelay caps the backoff (default 8s).
	MaxDelay time.Duration
	// Jitter is the fraction of each delay randomized away, in [0, 1]
	// (default 0.5). 0 disables jitter.
	Jitter float64
	// RetryIf decides whether an error is retryable. Default: rate limits,
	// overload, and provider-unavailable errors.
	RetryIf func(error) bool
	// OnRetry observes every retry decision before the wait.
	OnRetry func(attempt int, err error, delay time.Duration)
}

func (c RetryConfig) withDefaults() RetryConfig {
	if c.MaxAttempts <= 0 {
		c.MaxAttempts = 3
	}
	if c.BaseDelay <= 0 {
		c.BaseDelay = 500 * time.Millisecond
	}
	if c.MaxDelay <= 0 {
		c.MaxDelay = 8 * time.Second
	}
	if c.Jitter == 0 {
		c.Jitter = 0.5
	}
	if c.RetryIf == nil {
		c.RetryIf = Retryable
	}
	return c
}

// Retryable is the default retry predicate: transient provider conditions.
func Retryable(err error) bool {
	return errors.Is(err, ai.ErrRateLimited) ||
		errors.Is(err, ai.ErrOverloaded) ||
		errors.Is(err, ai.ErrProviderUnavailable)
}

// retryModel wraps a ChatModel with retry.
type retryModel struct {
	inner ai.ChatModel
	cfg   RetryConfig
}

// NewRetry wraps model with retry-and-backoff on transient errors. A provider
// supplied Retry-After (via [ai.APIError]) takes precedence over computed
// backoff for that wait.
func NewRetry(model ai.ChatModel, cfg RetryConfig) ai.ChatModel {
	return &retryModel{inner: model, cfg: cfg.withDefaults()}
}

func (r *retryModel) Name() string { return r.inner.Name() }

func (r *retryModel) Generate(ctx context.Context, req ai.Request) (ai.Response, error) {
	var lastErr error
	for attempt := 1; ; attempt++ {
		resp, err := r.inner.Generate(ctx, req)
		if err == nil {
			return resp, nil
		}
		lastErr = err
		if attempt >= r.cfg.MaxAttempts || !r.cfg.RetryIf(err) {
			return ai.Response{}, err
		}
		delay := r.delay(attempt, err)
		if r.cfg.OnRetry != nil {
			r.cfg.OnRetry(attempt, err, delay)
		}
		if err := sleep(ctx, delay); err != nil {
			return ai.Response{}, errors.Join(err, lastErr)
		}
	}
}

func (r *retryModel) Stream(ctx context.Context, req ai.Request) ai.Stream {
	return func(yield func(ai.Chunk, error) bool) {
		for attempt := 1; ; attempt++ {
			yielded := false
			var failure error
			r.inner.Stream(ctx, req)(func(c ai.Chunk, err error) bool {
				if err != nil && !yielded {
					failure = err // eligible for retry: nothing delivered yet
					return false
				}
				if err != nil {
					failure = err
					yielded = true // mark as delivered; not retryable
					return yield(c, err)
				}
				yielded = true
				return yield(c, nil)
			})
			if failure == nil || yielded {
				return // clean end, or error already delivered downstream
			}
			if attempt >= r.cfg.MaxAttempts || !r.cfg.RetryIf(failure) {
				yield(ai.Chunk{}, failure)
				return
			}
			delay := r.delay(attempt, failure)
			if r.cfg.OnRetry != nil {
				r.cfg.OnRetry(attempt, failure, delay)
			}
			if err := sleep(ctx, delay); err != nil {
				yield(ai.Chunk{}, errors.Join(err, failure))
				return
			}
		}
	}
}

func (r *retryModel) delay(attempt int, err error) time.Duration {
	var apiErr *ai.APIError
	if errors.As(err, &apiErr) && apiErr.RetryAfter > 0 {
		return apiErr.RetryAfter
	}
	d := r.cfg.BaseDelay << (attempt - 1)
	if d > r.cfg.MaxDelay {
		d = r.cfg.MaxDelay
	}
	if r.cfg.Jitter > 0 {
		d -= time.Duration(rand.Float64() * r.cfg.Jitter * float64(d))
	}
	return d
}

func sleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}
