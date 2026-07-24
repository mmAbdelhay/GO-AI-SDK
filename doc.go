// Package ai is a unified, idiomatic Go interface for working with multiple AI
// providers. It provides provider-neutral types for messages, requests,
// responses, streaming, token usage, and a normalized error taxonomy, plus the
// [ChatModel] interface every provider implements.
//
// # Generating text
//
// Construct a provider client (see the provider subpackages) and call the
// top-level helpers, which accept functional [Option] values:
//
//	model := anthropic.New(apiKey, anthropic.WithModel("claude-sonnet-4-20250514"))
//	resp, err := ai.Generate(ctx, model, "Write a haiku about Go.",
//		ai.WithTemperature(0.7), ai.WithMaxTokens(200))
//	fmt.Println(resp.Text())
//
// # Streaming
//
// [GenerateStream] returns a [Stream] iterator that can be ranged over directly,
// or drained with the [Stream.Text] and [Stream.Collect] helpers:
//
//	text, err := ai.GenerateStream(ctx, model, "Tell me a story.").Text()
//
// # Errors
//
// Provider errors normalize onto the package sentinels ([ErrRateLimited],
// [ErrContextLengthExceeded], and so on) so they can be detected uniformly:
//
//	if errors.Is(err, ai.ErrRateLimited) { ... }
//
// Use errors.As with [APIError] for provider-specific detail.
//
// # Testing
//
// The companion aitest package provides scripted fakes so agents and provider
// integrations can be tested without network access or API keys.
package ai
