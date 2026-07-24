// Package anthropic implements the [ai.ChatModel] interface against Anthropic's
// Messages API. It depends only on the standard library.
//
// Construct a client with [New] and use it directly or through the top-level
// helpers in the ai package:
//
//	model := anthropic.New(os.Getenv("ANTHROPIC_API_KEY"),
//		anthropic.WithModel("claude-sonnet-4-20250514"))
//	resp, err := ai.Generate(ctx, model, "Hello!")
//
// Provider-specific parameters that have no neutral equivalent are exposed as
// typed provider options, for example [WithTopK], and passed through
// ai.WithProviderOptions.
//
// The client performs no retries or backoff of its own (design principle P9);
// configure that behavior on the *http.Client supplied via [WithHTTPClient].
package anthropic
