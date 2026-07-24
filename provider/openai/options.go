package openai

import ai "github.com/mmabdelhay/go-ai-sdk"

// seedOption carries the OpenAI-specific seed parameter for reproducible
// sampling.
type seedOption struct {
	ai.ProviderOptionMarker
	Seed int
}

// WithSeed returns a provider option setting OpenAI's seed parameter, which
// requests best-effort deterministic sampling. Pass it through
// [ai.WithProviderOptions].
func WithSeed(seed int) ai.ProviderOption {
	return seedOption{Seed: seed}
}
