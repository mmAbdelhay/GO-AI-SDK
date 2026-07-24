// Package otelai instruments github.com/mmabdelhay/go-ai-sdk model calls with
// OpenTelemetry tracing. It plugs into the SDK's middleware.Hooks extension
// point, so tracing composes with the other middleware (logging, usage) exactly
// like any other layer:
//
//	tp := otel.GetTracerProvider()
//	model = middleware.Chain(client, otelai.Middleware(otelai.WithTracerProvider(tp)))
//
// Span attributes follow the OpenTelemetry GenAI semantic conventions
// (gen_ai.*), so they line up with other GenAI-instrumented systems.
package otelai

import (
	"context"
	"time"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/middleware"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

const (
	scopeName  = "github.com/mmabdelhay/go-ai-sdk/otel"
	systemName = "go-ai-sdk"
)

type config struct {
	tracer trace.Tracer
}

// Option configures the tracing middleware.
type Option func(*config)

// WithTracerProvider sets the OpenTelemetry TracerProvider. Defaults to the
// global provider (otel.GetTracerProvider).
func WithTracerProvider(tp trace.TracerProvider) Option {
	return func(c *config) { c.tracer = tp.Tracer(scopeName) }
}

// WithTracer sets the tracer directly.
func WithTracer(t trace.Tracer) Option {
	return func(c *config) { c.tracer = t }
}

// Middleware returns a middleware.Middleware that wraps every model call
// (Generate and Stream alike) in an OpenTelemetry span named "ai.generate". The
// span carries the request model and, on completion, token usage and the finish
// reason. Errors are recorded and set the span status to Error. For streaming
// calls the span ends when the stream is fully consumed, and its usage reflects
// the stream's terminal chunk.
func Middleware(opts ...Option) middleware.Middleware {
	cfg := &config{tracer: otel.GetTracerProvider().Tracer(scopeName)}
	for _, opt := range opts {
		opt(cfg)
	}

	// spanKey stores the active span between the Before and After hooks. A new
	// context value per call keeps concurrent calls isolated.
	type ctxKey struct{}

	return middleware.WithHooks(middleware.Hooks{
		Before: func(ctx context.Context, req ai.Request) context.Context {
			ctx, span := cfg.tracer.Start(ctx, "ai.generate",
				trace.WithSpanKind(trace.SpanKindClient),
				trace.WithAttributes(
					attribute.String("gen_ai.system", systemName),
					attribute.String("gen_ai.operation.name", "chat"),
					attribute.String("gen_ai.request.model", req.Model),
				),
			)
			if req.Temperature != nil {
				span.SetAttributes(attribute.Float64("gen_ai.request.temperature", *req.Temperature))
			}
			if req.MaxTokens > 0 {
				span.SetAttributes(attribute.Int("gen_ai.request.max_tokens", req.MaxTokens))
			}
			return context.WithValue(ctx, ctxKey{}, span)
		},
		After: func(ctx context.Context, _ ai.Request, resp ai.Response, err error, d time.Duration) {
			span, ok := ctx.Value(ctxKey{}).(trace.Span)
			if !ok {
				return
			}
			defer span.End()
			span.SetAttributes(attribute.Int64("gen_ai.client.operation.duration_ms", d.Milliseconds()))
			if err != nil {
				span.RecordError(err)
				span.SetStatus(codes.Error, err.Error())
				return
			}
			if resp.Model != "" {
				span.SetAttributes(attribute.String("gen_ai.response.model", resp.Model))
			}
			span.SetAttributes(
				attribute.Int("gen_ai.usage.input_tokens", resp.Usage.InputTokens),
				attribute.Int("gen_ai.usage.output_tokens", resp.Usage.OutputTokens),
			)
			if resp.StopReason != "" {
				span.SetAttributes(attribute.StringSlice("gen_ai.response.finish_reasons",
					[]string{string(resp.StopReason)}))
			}
			span.SetStatus(codes.Ok, "")
		},
	})
}
