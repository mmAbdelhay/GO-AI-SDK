package anthropic

import ai "github.com/mmabdelhay/go-ai-sdk"

// topKOption carries the Anthropic-specific top_k sampling parameter. It embeds
// [ai.ProviderOptionMarker] to satisfy [ai.ProviderOption] so it can travel
// through the neutral ai.Request.ProviderOptions slice while remaining fully
// typed.
type topKOption struct {
	ai.ProviderOptionMarker
	K int
}

// WithTopK returns a provider option that sets Anthropic's top_k sampling
// parameter. Pass it through [ai.WithProviderOptions]:
//
//	ai.Generate(ctx, model, prompt, ai.WithProviderOptions(anthropic.WithTopK(40)))
func WithTopK(k int) ai.ProviderOption {
	return topKOption{K: k}
}
