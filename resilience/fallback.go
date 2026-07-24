package resilience

import (
	"context"
	"errors"
	"fmt"
	"strings"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// FallbackConfig configures [NewFallback].
type FallbackConfig struct {
	// FailoverIf decides whether an error should trigger the next model.
	// Default: transient conditions ([Retryable]) plus authentication errors,
	// since a misconfigured key on one provider should not take the whole
	// chain down.
	FailoverIf func(error) bool
	// OnFailover observes each failover: which model failed, which is next, and
	// why. This is how callers surface the event (P9: observable).
	OnFailover func(from, to string, err error)
}

type fallbackModel struct {
	cfg    FallbackConfig
	models []ai.ChatModel
}

// NewFallback returns a ChatModel that tries each model in order, failing over
// on errors the config deems eligible. Errors that are not failover-eligible
// (for example invalid requests, which would fail everywhere) are returned
// immediately. If every model fails, the errors are joined.
func NewFallback(cfg FallbackConfig, models ...ai.ChatModel) ai.ChatModel {
	if len(models) == 0 {
		panic("resilience: NewFallback requires at least one model")
	}
	if cfg.FailoverIf == nil {
		cfg.FailoverIf = func(err error) bool {
			return Retryable(err) || errors.Is(err, ai.ErrAuthentication)
		}
	}
	return &fallbackModel{cfg: cfg, models: models}
}

func (f *fallbackModel) Name() string {
	names := make([]string, len(f.models))
	for i, m := range f.models {
		names[i] = m.Name()
	}
	return "fallback(" + strings.Join(names, ",") + ")"
}

func (f *fallbackModel) Generate(ctx context.Context, req ai.Request) (ai.Response, error) {
	var errs []error
	for i, m := range f.models {
		resp, err := m.Generate(ctx, req)
		if err == nil {
			return resp, nil
		}
		errs = append(errs, err)
		if !f.cfg.FailoverIf(err) || i == len(f.models)-1 {
			break
		}
		if f.cfg.OnFailover != nil {
			f.cfg.OnFailover(m.Name(), f.models[i+1].Name(), err)
		}
	}
	return ai.Response{}, fmt.Errorf("resilience: all fallbacks exhausted: %w", errors.Join(errs...))
}

func (f *fallbackModel) Stream(ctx context.Context, req ai.Request) ai.Stream {
	return func(yield func(ai.Chunk, error) bool) {
		var errs []error
		for i, m := range f.models {
			yielded := false
			var failure error
			m.Stream(ctx, req)(func(c ai.Chunk, err error) bool {
				if err != nil && !yielded {
					failure = err // nothing delivered; next model may serve it
					return false
				}
				yielded = true
				return yield(c, err)
			})
			if failure == nil || yielded {
				return
			}
			errs = append(errs, failure)
			if !f.cfg.FailoverIf(failure) || i == len(f.models)-1 {
				break
			}
			if f.cfg.OnFailover != nil {
				f.cfg.OnFailover(m.Name(), f.models[i+1].Name(), failure)
			}
		}
		yield(ai.Chunk{}, fmt.Errorf("resilience: all fallbacks exhausted: %w", errors.Join(errs...)))
	}
}
