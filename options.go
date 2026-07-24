package ai

// Option configures a [Request]. Options are applied in order by the top-level
// [Generate] and [Stream] helpers, so later options override earlier ones.
type Option func(*Request)

// WithModel sets the model identifier.
func WithModel(model string) Option {
	return func(r *Request) { r.Model = model }
}

// WithTemperature sets the sampling temperature.
func WithTemperature(t float64) Option {
	return func(r *Request) { r.Temperature = &t }
}

// WithTopP sets nucleus sampling.
func WithTopP(p float64) Option {
	return func(r *Request) { r.TopP = &p }
}

// WithMaxTokens caps the number of output tokens.
func WithMaxTokens(n int) Option {
	return func(r *Request) { r.MaxTokens = n }
}

// WithSystem sets the system prompt.
func WithSystem(s string) Option {
	return func(r *Request) { r.System = s }
}

// WithStopSequences sets sequences that halt generation.
func WithStopSequences(seqs ...string) Option {
	return func(r *Request) { r.StopSequences = seqs }
}

// WithMessages appends messages to the request. Use this to supply conversation
// history alongside (or instead of) the convenience prompt argument.
func WithMessages(msgs ...Message) Option {
	return func(r *Request) { r.Messages = append(r.Messages, msgs...) }
}

// WithProviderOptions attaches provider-specific typed options. Each provider
// inspects only the options it recognizes.
func WithProviderOptions(opts ...ProviderOption) Option {
	return func(r *Request) { r.ProviderOptions = append(r.ProviderOptions, opts...) }
}

// ProviderOption is a marker interface for provider-specific request tuning.
// Concrete options live in provider packages (for example anthropic.WithTopK),
// keeping the core [Request] free of any map[string]any escape hatch while still
// allowing full, type-safe access to provider features.
//
// Because the marker method is unexported, provider packages implement
// ProviderOption by embedding [ProviderOptionMarker] rather than declaring the
// method themselves.
type ProviderOption interface {
	providerOption()
}

// ProviderOptionMarker is embedded in provider-specific option types to satisfy
// [ProviderOption] without exposing the marker method on the public API:
//
//	type topKOption struct {
//		ai.ProviderOptionMarker
//		K int
//	}
type ProviderOptionMarker struct{}

func (ProviderOptionMarker) providerOption() {}
