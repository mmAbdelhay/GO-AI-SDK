package ai

import "context"

// buildRequest constructs a Request from an optional prompt and functional
// options. A non-empty prompt is appended as a trailing user message, so callers
// can mix a convenience prompt with prior history supplied via WithMessages.
func buildRequest(prompt string, opts []Option) Request {
	var req Request
	for _, opt := range opts {
		opt(&req)
	}
	if prompt != "" {
		req.Messages = append(req.Messages, UserText(prompt))
	}
	return req
}

// Generate is the top-level convenience entrypoint for text generation. It builds
// a [Request] from prompt and opts and calls m.Generate. A non-empty prompt is
// appended as a user message after any messages supplied via [WithMessages].
func Generate(ctx context.Context, m ChatModel, prompt string, opts ...Option) (Response, error) {
	return m.Generate(ctx, buildRequest(prompt, opts))
}

// GenerateStream is the top-level convenience entrypoint for streaming
// generation. It builds a [Request] from prompt and opts and calls m.Stream. It
// is named GenerateStream rather than Stream to avoid colliding with the [Stream]
// type, which it returns.
func GenerateStream(ctx context.Context, m ChatModel, prompt string, opts ...Option) Stream {
	return m.Stream(ctx, buildRequest(prompt, opts))
}
