package otelai_test

import (
	"context"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/aitest"
	"github.com/mmabdelhay/go-ai-sdk/middleware"
	otelai "github.com/mmabdelhay/go-ai-sdk/otel"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func setup(t *testing.T) (*tracetest.SpanRecorder, middleware.Middleware) {
	t.Helper()
	sr := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(sr))
	t.Cleanup(func() { _ = tp.Shutdown(context.Background()) })
	return sr, otelai.Middleware(otelai.WithTracerProvider(tp))
}

func attrsOf(kvs []attribute.KeyValue) map[string]attribute.Value {
	m := make(map[string]attribute.Value, len(kvs))
	for _, kv := range kvs {
		m[string(kv.Key)] = kv.Value
	}
	return m
}

func TestGenerateProducesSpan(t *testing.T) {
	sr, mw := setup(t)

	resp := aitest.TextResponse("hi")
	resp.Model = "test-model"
	resp.Usage = ai.Usage{InputTokens: 12, OutputTokens: 5}
	model := middleware.Chain(&aitest.Model{Responses: []ai.Response{resp}}, mw)

	if _, err := ai.Generate(context.Background(), model, "hello", ai.WithModel("test-model")); err != nil {
		t.Fatal(err)
	}

	spans := sr.Ended()
	if len(spans) != 1 {
		t.Fatalf("spans = %d, want 1", len(spans))
	}
	span := spans[0]
	if span.Name() != "ai.generate" {
		t.Errorf("span name = %q", span.Name())
	}
	if span.Status().Code != codes.Ok {
		t.Errorf("status = %v, want Ok", span.Status().Code)
	}
	attrs := attrsOf(span.Attributes())
	if got := attrs["gen_ai.request.model"].AsString(); got != "test-model" {
		t.Errorf("request model attr = %q", got)
	}
	if got := attrs["gen_ai.usage.input_tokens"].AsInt64(); got != 12 {
		t.Errorf("input tokens attr = %d", got)
	}
	if got := attrs["gen_ai.usage.output_tokens"].AsInt64(); got != 5 {
		t.Errorf("output tokens attr = %d", got)
	}
	if got := attrs["gen_ai.response.finish_reasons"].AsStringSlice(); len(got) != 1 || got[0] != "end_turn" {
		t.Errorf("finish reasons attr = %v", got)
	}
}

func TestErrorSetsSpanStatus(t *testing.T) {
	sr, mw := setup(t)
	model := middleware.Chain(&aitest.Model{Err: ai.NewAPIError("p", 429, ai.ErrRateLimited)}, mw)

	_, _ = ai.Generate(context.Background(), model, "x")

	spans := sr.Ended()
	if len(spans) != 1 {
		t.Fatalf("spans = %d, want 1", len(spans))
	}
	if spans[0].Status().Code != codes.Error {
		t.Errorf("status = %v, want Error", spans[0].Status().Code)
	}
	if len(spans[0].Events()) == 0 {
		t.Errorf("expected a recorded error event")
	}
}

func TestStreamProducesSpanEndingAfterConsumption(t *testing.T) {
	sr, mw := setup(t)
	model := middleware.Chain(&aitest.Model{StreamFunc: func(context.Context, ai.Request) ai.Stream {
		return aitest.StreamOf("a", "b")
	}}, mw)

	stream := ai.GenerateStream(context.Background(), model, "x")
	// Span must not have ended until the stream is drained.
	if len(sr.Ended()) != 0 {
		t.Errorf("span ended before stream consumed")
	}
	if _, err := stream.Text(); err != nil {
		t.Fatal(err)
	}
	if len(sr.Ended()) != 1 {
		t.Fatalf("spans after drain = %d, want 1", len(sr.Ended()))
	}
}
