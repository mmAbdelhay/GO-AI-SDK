package middleware_test

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"
	"time"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/aitest"
	"github.com/mmabdelhay/go-ai-sdk/middleware"
)

func respWithUsage(text string, in, out int) ai.Response {
	r := aitest.TextResponse(text)
	r.Usage = ai.Usage{InputTokens: in, OutputTokens: out}
	r.Model = "test-model"
	return r
}

func TestHooksAroundGenerate(t *testing.T) {
	inner := &aitest.Model{Responses: []ai.Response{respWithUsage("hi", 5, 3)}}
	var beforeReq, afterResp bool
	m := middleware.Chain(inner, middleware.WithHooks(middleware.Hooks{
		Before: func(ctx context.Context, req ai.Request) context.Context {
			beforeReq = len(req.Messages) == 1
			return ctx
		},
		After: func(_ context.Context, _ ai.Request, resp ai.Response, err error, d time.Duration) {
			afterResp = err == nil && resp.Text() == "hi" && d >= 0
		},
	}))

	if _, err := ai.Generate(context.Background(), m, "x"); err != nil {
		t.Fatal(err)
	}
	if !beforeReq || !afterResp {
		t.Errorf("hooks not invoked correctly: before=%v after=%v", beforeReq, afterResp)
	}
}

func TestHooksAroundStream(t *testing.T) {
	inner := &aitest.Model{StreamFunc: func(context.Context, ai.Request) ai.Stream {
		return func(yield func(ai.Chunk, error) bool) {
			if !yield(ai.Chunk{Type: ai.ChunkText, Text: "a"}, nil) {
				return
			}
			yield(ai.Chunk{Type: ai.ChunkDone, StopReason: ai.StopEndTurn,
				Usage: &ai.Usage{InputTokens: 2, OutputTokens: 1}}, nil)
		}
	}}
	chunks := 0
	var finalUsage ai.Usage
	m := middleware.Chain(inner, middleware.WithHooks(middleware.Hooks{
		OnChunk: func(ai.Chunk) { chunks++ },
		After: func(_ context.Context, _ ai.Request, resp ai.Response, _ error, _ time.Duration) {
			finalUsage = resp.Usage
		},
	}))

	if _, err := m.Stream(context.Background(), ai.Request{}).Text(); err != nil {
		t.Fatal(err)
	}
	if chunks != 2 {
		t.Errorf("chunks = %d, want 2", chunks)
	}
	if finalUsage.TotalTokens() != 3 {
		t.Errorf("stream usage not surfaced to After: %+v", finalUsage)
	}
}

func TestLogging(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, nil))

	inner := &aitest.Model{Responses: []ai.Response{respWithUsage("ok", 7, 2)}}
	m := middleware.Chain(inner, middleware.Logging(logger))
	if _, err := ai.Generate(context.Background(), m, "x", ai.WithModel("test-model")); err != nil {
		t.Fatal(err)
	}
	log := buf.String()
	for _, want := range []string{"ai.call", "model=test-model", "input_tokens=7", "output_tokens=2"} {
		if !strings.Contains(log, want) {
			t.Errorf("log missing %q: %s", want, log)
		}
	}

	// Errors log at error level.
	buf.Reset()
	failing := &aitest.Model{Err: ai.NewAPIError("p", 429, ai.ErrRateLimited)}
	m = middleware.Chain(failing, middleware.Logging(logger))
	_, _ = ai.Generate(context.Background(), m, "x")
	if !strings.Contains(buf.String(), "level=ERROR") {
		t.Errorf("error not logged at error level: %s", buf.String())
	}
}

func TestUsageTrackerAndCost(t *testing.T) {
	tracker := middleware.NewUsageTracker()
	inner := &aitest.Model{GenerateFunc: func(context.Context, ai.Request) (ai.Response, error) {
		return respWithUsage("ok", 100, 50), nil
	}}
	m := middleware.Chain(inner, tracker.Middleware())

	ctx := context.Background()
	for range 3 {
		if _, err := ai.Generate(ctx, m, "x"); err != nil {
			t.Fatal(err)
		}
	}

	if tracker.Calls() != 3 {
		t.Errorf("calls = %d", tracker.Calls())
	}
	total := tracker.Total()
	if total.InputTokens != 300 || total.OutputTokens != 150 {
		t.Errorf("total = %+v", total)
	}
	per := tracker.PerModel()
	if per["test-model"].InputTokens != 300 {
		t.Errorf("per-model = %+v", per)
	}

	cost := tracker.Cost(middleware.Pricing{
		"test-model": {InputPerMTok: 1.0, OutputPerMTok: 2.0},
	})
	want := 300.0/1e6*1.0 + 150.0/1e6*2.0
	if cost != want {
		t.Errorf("cost = %v, want %v", cost, want)
	}
	// Unknown models price at zero, not panic.
	if got := tracker.Cost(middleware.Pricing{}); got != 0 {
		t.Errorf("empty pricing cost = %v", got)
	}
}

func TestChainOrder(t *testing.T) {
	var order []string
	mk := func(name string) middleware.Middleware {
		return middleware.WithHooks(middleware.Hooks{
			Before: func(ctx context.Context, _ ai.Request) context.Context {
				order = append(order, name)
				return ctx
			},
		})
	}
	inner := &aitest.Model{Responses: []ai.Response{aitest.TextResponse("x")}}
	m := middleware.Chain(inner, mk("outer"), mk("inner"))
	_, _ = ai.Generate(context.Background(), m, "x")
	if len(order) != 2 || order[0] != "outer" || order[1] != "inner" {
		t.Errorf("order = %v", order)
	}
}
