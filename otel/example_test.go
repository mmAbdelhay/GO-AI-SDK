package otelai_test

import (
	"context"
	"fmt"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/aitest"
	"github.com/mmabdelhay/go-ai-sdk/middleware"
	otelai "github.com/mmabdelhay/go-ai-sdk/otel"

	"go.opentelemetry.io/otel/sdk/trace"
)

// Example wraps a model with OpenTelemetry tracing. In production you would pass
// your application's TracerProvider; here a fake model and a no-op provider keep
// it offline.
func Example() {
	tp := trace.NewTracerProvider()
	defer func() { _ = tp.Shutdown(context.Background()) }()

	// Compose tracing with any other middleware.
	model := middleware.Chain(
		&aitest.Model{Responses: []ai.Response{aitest.TextResponse("hi")}},
		otelai.Middleware(otelai.WithTracerProvider(tp)),
	)

	resp, _ := ai.Generate(context.Background(), model, "hello")
	fmt.Println(resp.Text())
	// Output: hi
}
