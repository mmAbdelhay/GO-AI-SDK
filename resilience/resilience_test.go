package resilience_test

import (
	"context"
	"errors"
	"testing"
	"time"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/aitest"
	"github.com/mmabdelhay/go-ai-sdk/resilience"
)

func rateLimitedErr() error {
	return ai.NewAPIError("test", 429, ai.ErrRateLimited)
}

func fastRetry(m ai.ChatModel, attempts int, onRetry func(int, error, time.Duration)) ai.ChatModel {
	return resilience.NewRetry(m, resilience.RetryConfig{
		MaxAttempts: attempts,
		BaseDelay:   time.Millisecond,
		MaxDelay:    2 * time.Millisecond,
		OnRetry:     onRetry,
	})
}

func TestRetrySucceedsAfterTransientFailures(t *testing.T) {
	calls := 0
	inner := &aitest.Model{GenerateFunc: func(context.Context, ai.Request) (ai.Response, error) {
		calls++
		if calls < 3 {
			return ai.Response{}, rateLimitedErr()
		}
		return aitest.TextResponse("finally"), nil
	}}
	retries := 0
	m := fastRetry(inner, 3, func(int, error, time.Duration) { retries++ })

	resp, err := ai.Generate(context.Background(), m, "hi")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "finally" || calls != 3 || retries != 2 {
		t.Errorf("text=%q calls=%d retries=%d", resp.Text(), calls, retries)
	}
}

func TestRetryDoesNotRetryNonRetryable(t *testing.T) {
	calls := 0
	inner := &aitest.Model{GenerateFunc: func(context.Context, ai.Request) (ai.Response, error) {
		calls++
		return ai.Response{}, ai.NewAPIError("test", 400, ai.ErrInvalidRequest)
	}}
	m := fastRetry(inner, 5, nil)

	_, err := ai.Generate(context.Background(), m, "hi")
	if !errors.Is(err, ai.ErrInvalidRequest) || calls != 1 {
		t.Errorf("err=%v calls=%d, want invalid-request after 1 call", err, calls)
	}
}

func TestRetryExhausted(t *testing.T) {
	inner := &aitest.Model{GenerateFunc: func(context.Context, ai.Request) (ai.Response, error) {
		return ai.Response{}, rateLimitedErr()
	}}
	m := fastRetry(inner, 2, nil)
	if _, err := ai.Generate(context.Background(), m, "hi"); !errors.Is(err, ai.ErrRateLimited) {
		t.Errorf("err = %v", err)
	}
}

func TestRetryStreamSetupError(t *testing.T) {
	calls := 0
	inner := &aitest.Model{StreamFunc: func(context.Context, ai.Request) ai.Stream {
		calls++
		if calls < 2 {
			return aitest.ErrorStream(rateLimitedErr())
		}
		return aitest.StreamOf("ok")
	}}
	m := fastRetry(inner, 3, nil)

	text, err := m.Stream(context.Background(), ai.Request{}).Text()
	if err != nil || text != "ok" {
		t.Errorf("text=%q err=%v after %d calls", text, err, calls)
	}
}

func TestRetryStreamMidwayErrorNotRetried(t *testing.T) {
	calls := 0
	inner := &aitest.Model{StreamFunc: func(context.Context, ai.Request) ai.Stream {
		calls++
		return aitest.ErrorStream(rateLimitedErr(), "partial ")
	}}
	m := fastRetry(inner, 3, nil)

	text, err := m.Stream(context.Background(), ai.Request{}).Text()
	if !errors.Is(err, ai.ErrRateLimited) {
		t.Fatalf("err = %v", err)
	}
	if calls != 1 {
		t.Errorf("mid-stream failure was retried (%d calls); output would duplicate", calls)
	}
	if text != "partial " {
		t.Errorf("partial text = %q", text)
	}
}

func TestFallbackFailsOver(t *testing.T) {
	primary := &aitest.Model{ModelName: "primary", Err: rateLimitedErr()}
	secondary := &aitest.Model{ModelName: "secondary", Responses: []ai.Response{aitest.TextResponse("rescued")}}

	var from, to string
	m := resilience.NewFallback(resilience.FallbackConfig{
		OnFailover: func(f, t string, _ error) { from, to = f, t },
	}, primary, secondary)

	resp, err := ai.Generate(context.Background(), m, "hi")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "rescued" {
		t.Errorf("text = %q", resp.Text())
	}
	// The failover event must be surfaced to the caller.
	if from != "primary" || to != "secondary" {
		t.Errorf("failover not observed: from=%q to=%q", from, to)
	}
}

func TestFallbackDoesNotFailOverOnInvalidRequest(t *testing.T) {
	primary := &aitest.Model{Err: ai.NewAPIError("p", 400, ai.ErrInvalidRequest)}
	secondary := &aitest.Model{Responses: []ai.Response{aitest.TextResponse("nope")}}
	m := resilience.NewFallback(resilience.FallbackConfig{}, primary, secondary)

	_, err := ai.Generate(context.Background(), m, "hi")
	if !errors.Is(err, ai.ErrInvalidRequest) {
		t.Errorf("err = %v", err)
	}
	if secondary.CallCount() != 0 {
		t.Errorf("invalid request leaked to fallback")
	}
}

func TestFallbackAllExhausted(t *testing.T) {
	a := &aitest.Model{Err: rateLimitedErr()}
	b := &aitest.Model{Err: ai.NewAPIError("b", 529, ai.ErrOverloaded)}
	m := resilience.NewFallback(resilience.FallbackConfig{}, a, b)

	_, err := ai.Generate(context.Background(), m, "hi")
	// Both underlying errors must be discoverable in the joined error.
	if !errors.Is(err, ai.ErrRateLimited) || !errors.Is(err, ai.ErrOverloaded) {
		t.Errorf("joined error missing causes: %v", err)
	}
}

func TestFallbackStream(t *testing.T) {
	primary := &aitest.Model{StreamFunc: func(context.Context, ai.Request) ai.Stream {
		return aitest.ErrorStream(rateLimitedErr())
	}}
	secondary := &aitest.Model{StreamFunc: func(context.Context, ai.Request) ai.Stream {
		return aitest.StreamOf("saved")
	}}
	m := resilience.NewFallback(resilience.FallbackConfig{}, primary, secondary)

	text, err := m.Stream(context.Background(), ai.Request{}).Text()
	if err != nil || text != "saved" {
		t.Errorf("text=%q err=%v", text, err)
	}
}

func TestBreakerOpensAndRecovers(t *testing.T) {
	failing := true
	inner := &aitest.Model{GenerateFunc: func(context.Context, ai.Request) (ai.Response, error) {
		if failing {
			return ai.Response{}, rateLimitedErr()
		}
		return aitest.TextResponse("recovered"), nil
	}}

	var transitions []string
	clock := time.Now()
	m := resilience.NewBreaker(inner, resilience.BreakerConfig{
		FailureThreshold: 2,
		Cooldown:         time.Millisecond,
		OnStateChange: func(from, to resilience.BreakerState) {
			transitions = append(transitions, from.String()+"->"+to.String())
		},
	})
	_ = clock
	ctx := context.Background()

	// Two failures open the circuit.
	_, _ = ai.Generate(ctx, m, "1")
	_, _ = ai.Generate(ctx, m, "2")

	// While open, calls fail fast without reaching the provider.
	before := inner.CallCount()
	_, err := ai.Generate(ctx, m, "3")
	if !errors.Is(err, resilience.ErrCircuitOpen) {
		t.Fatalf("err = %v, want ErrCircuitOpen", err)
	}
	if !errors.Is(err, ai.ErrProviderUnavailable) {
		t.Errorf("open-circuit error should also match ErrProviderUnavailable")
	}
	if inner.CallCount() != before {
		t.Errorf("open circuit still called provider")
	}

	// After cooldown, a trial call closes the circuit again.
	failing = false
	time.Sleep(2 * time.Millisecond)
	resp, err := ai.Generate(ctx, m, "4")
	if err != nil || resp.Text() != "recovered" {
		t.Fatalf("trial call: %q, %v", resp.Text(), err)
	}
	want := []string{"closed->open", "open->half-open", "half-open->closed"}
	if len(transitions) != len(want) {
		t.Fatalf("transitions = %v, want %v", transitions, want)
	}
	for i := range want {
		if transitions[i] != want[i] {
			t.Errorf("transition[%d] = %q, want %q", i, transitions[i], want[i])
		}
	}
}

func TestRateLimiterWaits(t *testing.T) {
	inner := &aitest.Model{GenerateFunc: func(context.Context, ai.Request) (ai.Response, error) {
		return aitest.TextResponse("ok"), nil
	}}
	var waits []time.Duration
	m := resilience.NewRateLimiter(inner, resilience.RateLimiterConfig{
		RequestsPerSecond: 1000, // 1ms per token; fast test
		Burst:             1,
		OnWait:            func(d time.Duration) { waits = append(waits, d) },
	})

	ctx := context.Background()
	start := time.Now()
	for range 3 {
		if _, err := ai.Generate(ctx, m, "hi"); err != nil {
			t.Fatal(err)
		}
	}
	if len(waits) == 0 {
		t.Error("no waits observed for burst-exceeding calls")
	}
	if elapsed := time.Since(start); elapsed < time.Millisecond {
		t.Errorf("3 calls at 1000rps finished in %v; limiter did not pace", elapsed)
	}
}

func TestRateLimiterHonorsContext(t *testing.T) {
	inner := &aitest.Model{Responses: []ai.Response{aitest.TextResponse("x")}}
	m := resilience.NewRateLimiter(inner, resilience.RateLimiterConfig{
		RequestsPerSecond: 0.001, // ~17min per token: the second call must wait
		Burst:             1,
	})
	ctx := context.Background()
	if _, err := ai.Generate(ctx, m, "1"); err != nil {
		t.Fatal(err)
	}
	cancelCtx, cancel := context.WithTimeout(ctx, 5*time.Millisecond)
	defer cancel()
	if _, err := ai.Generate(cancelCtx, m, "2"); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want DeadlineExceeded", err)
	}
}

func TestComposition(t *testing.T) {
	// Retry inside, fallback outside: primary exhausts retries, then fallback
	// rescues — the composed pipeline the package doc advertises.
	primary := &aitest.Model{ModelName: "p", Err: rateLimitedErr()}
	secondary := &aitest.Model{ModelName: "s", Responses: []ai.Response{aitest.TextResponse("done")}}

	m := resilience.NewFallback(resilience.FallbackConfig{},
		fastRetry(primary, 2, nil),
		secondary,
	)
	resp, err := ai.Generate(context.Background(), m, "hi")
	if err != nil || resp.Text() != "done" {
		t.Errorf("text=%q err=%v", resp.Text(), err)
	}
	if primary.CallCount() != 2 {
		t.Errorf("primary calls = %d, want 2 (retries before failover)", primary.CallCount())
	}
}
