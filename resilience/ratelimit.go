package resilience

import (
	"context"
	"sync"
	"time"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// RateLimiterConfig configures [NewRateLimiter].
type RateLimiterConfig struct {
	// RequestsPerSecond is the sustained request rate (required, > 0).
	RequestsPerSecond float64
	// Burst is how many requests may proceed immediately from a full bucket
	// (default 1).
	Burst int
	// OnWait observes each wait imposed by the limiter.
	OnWait func(d time.Duration)
	// now is injectable for tests.
	now func() time.Time
}

// rateLimitedModel applies a token-bucket rate limit in front of a ChatModel.
type rateLimitedModel struct {
	inner ai.ChatModel
	cfg   RateLimiterConfig

	mu     sync.Mutex
	tokens float64
	last   time.Time
}

// NewRateLimiter wraps model with a client-side token-bucket rate limit. Calls
// wait (honoring ctx) until a token is available; waits are observable via
// OnWait.
func NewRateLimiter(model ai.ChatModel, cfg RateLimiterConfig) ai.ChatModel {
	if cfg.RequestsPerSecond <= 0 {
		panic("resilience: RequestsPerSecond must be > 0")
	}
	if cfg.Burst <= 0 {
		cfg.Burst = 1
	}
	if cfg.now == nil {
		cfg.now = time.Now
	}
	return &rateLimitedModel{
		inner:  model,
		cfg:    cfg,
		tokens: float64(cfg.Burst),
		last:   cfg.now(),
	}
}

func (r *rateLimitedModel) Name() string { return r.inner.Name() }

// reserve takes a token, returning how long the caller must wait first.
func (r *rateLimitedModel) reserve() time.Duration {
	r.mu.Lock()
	defer r.mu.Unlock()
	now := r.cfg.now()
	r.tokens += now.Sub(r.last).Seconds() * r.cfg.RequestsPerSecond
	if max := float64(r.cfg.Burst); r.tokens > max {
		r.tokens = max
	}
	r.last = now
	r.tokens--
	if r.tokens >= 0 {
		return 0
	}
	return time.Duration(-r.tokens / r.cfg.RequestsPerSecond * float64(time.Second))
}

func (r *rateLimitedModel) wait(ctx context.Context) error {
	d := r.reserve()
	if d <= 0 {
		return nil
	}
	if r.cfg.OnWait != nil {
		r.cfg.OnWait(d)
	}
	return sleep(ctx, d)
}

func (r *rateLimitedModel) Generate(ctx context.Context, req ai.Request) (ai.Response, error) {
	if err := r.wait(ctx); err != nil {
		return ai.Response{}, err
	}
	return r.inner.Generate(ctx, req)
}

func (r *rateLimitedModel) Stream(ctx context.Context, req ai.Request) ai.Stream {
	return func(yield func(ai.Chunk, error) bool) {
		if err := r.wait(ctx); err != nil {
			yield(ai.Chunk{}, err)
			return
		}
		r.inner.Stream(ctx, req)(yield)
	}
}
