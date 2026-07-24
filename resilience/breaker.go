package resilience

import (
	"context"
	"errors"
	"sync"
	"time"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// ErrCircuitOpen is returned (wrapped) when the circuit breaker is open and
// calls are being rejected without reaching the provider.
var ErrCircuitOpen = errors.New("resilience: circuit open")

// BreakerState is the circuit breaker's state.
type BreakerState int

const (
	// BreakerClosed means calls flow normally.
	BreakerClosed BreakerState = iota
	// BreakerOpen means calls are rejected immediately.
	BreakerOpen
	// BreakerHalfOpen means one trial call is allowed through.
	BreakerHalfOpen
)

func (s BreakerState) String() string {
	switch s {
	case BreakerClosed:
		return "closed"
	case BreakerOpen:
		return "open"
	case BreakerHalfOpen:
		return "half-open"
	default:
		return "unknown"
	}
}

// BreakerConfig configures [NewBreaker].
type BreakerConfig struct {
	// FailureThreshold is the number of consecutive eligible failures that
	// opens the circuit (default 5).
	FailureThreshold int
	// Cooldown is how long the circuit stays open before allowing a trial call
	// (default 30s).
	Cooldown time.Duration
	// CountIf decides which errors count as failures (default [Retryable]).
	// Errors that don't count (e.g. invalid requests) pass through without
	// affecting the breaker.
	CountIf func(error) bool
	// OnStateChange observes transitions.
	OnStateChange func(from, to BreakerState)
	// now is injectable for tests.
	now func() time.Time
}

type breakerModel struct {
	inner ai.ChatModel
	cfg   BreakerConfig

	mu       sync.Mutex
	state    BreakerState
	failures int
	openedAt time.Time
}

// NewBreaker wraps model with a circuit breaker: after FailureThreshold
// consecutive failures, calls fail fast with an error wrapping [ErrCircuitOpen]
// (and [ai.ErrProviderUnavailable], so existing taxonomy checks match) until the
// cooldown elapses; then one trial call decides whether to close the circuit.
func NewBreaker(model ai.ChatModel, cfg BreakerConfig) ai.ChatModel {
	if cfg.FailureThreshold <= 0 {
		cfg.FailureThreshold = 5
	}
	if cfg.Cooldown <= 0 {
		cfg.Cooldown = 30 * time.Second
	}
	if cfg.CountIf == nil {
		cfg.CountIf = Retryable
	}
	if cfg.now == nil {
		cfg.now = time.Now
	}
	return &breakerModel{inner: model, cfg: cfg}
}

func (b *breakerModel) Name() string { return b.inner.Name() }

// admit decides whether a call may proceed. It returns an error when the
// circuit is open.
func (b *breakerModel) admit() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	switch b.state {
	case BreakerClosed:
		return nil
	case BreakerOpen:
		if b.cfg.now().Sub(b.openedAt) >= b.cfg.Cooldown {
			b.transition(BreakerHalfOpen)
			return nil // the trial call
		}
		return errors.Join(ErrCircuitOpen, ai.ErrProviderUnavailable)
	case BreakerHalfOpen:
		// A trial is already in flight; reject concurrent calls.
		return errors.Join(ErrCircuitOpen, ai.ErrProviderUnavailable)
	}
	return nil
}

func (b *breakerModel) record(err error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err == nil {
		b.failures = 0
		if b.state != BreakerClosed {
			b.transition(BreakerClosed)
		}
		return
	}
	if !b.cfg.CountIf(err) {
		return
	}
	b.failures++
	if b.state == BreakerHalfOpen || (b.state == BreakerClosed && b.failures >= b.cfg.FailureThreshold) {
		b.openedAt = b.cfg.now()
		b.transition(BreakerOpen)
	}
}

// transition must be called with b.mu held.
func (b *breakerModel) transition(to BreakerState) {
	from := b.state
	b.state = to
	if b.cfg.OnStateChange != nil && from != to {
		b.cfg.OnStateChange(from, to)
	}
}

func (b *breakerModel) Generate(ctx context.Context, req ai.Request) (ai.Response, error) {
	if err := b.admit(); err != nil {
		return ai.Response{}, err
	}
	resp, err := b.inner.Generate(ctx, req)
	b.record(err)
	return resp, err
}

func (b *breakerModel) Stream(ctx context.Context, req ai.Request) ai.Stream {
	return func(yield func(ai.Chunk, error) bool) {
		if err := b.admit(); err != nil {
			yield(ai.Chunk{}, err)
			return
		}
		var failure error
		b.inner.Stream(ctx, req)(func(c ai.Chunk, err error) bool {
			if err != nil {
				failure = err
			}
			return yield(c, err)
		})
		b.record(failure)
	}
}
