# Go AI SDK

A unified, idiomatic Go interface for working with multiple AI providers —
Anthropic, OpenAI, Gemini, Groq, and xAI behind one type-safe API. Text
generation, streaming, structured output with generics, agents with tools,
conversation memory, embeddings and vector search, resilience, and
observability. Designed the Go way, not ported from another ecosystem.

## Install

```sh
go get github.com/mmabdelhay/go-ai-sdk
```

Requires Go 1.23+ (range-over-func iterators). The core module has **zero
non-standard-library dependencies**.

## Quickstart

```go
model := anthropic.New(os.Getenv("ANTHROPIC_API_KEY"),
	anthropic.WithModel("claude-sonnet-4-20250514"))

resp, err := ai.Generate(ctx, model, "Write a haiku about Go.",
	ai.WithMaxTokens(200))
fmt.Println(resp.Text())
```

Swap `anthropic.New` for `openai.New`, `gemini.New`, `groq.New`, or `xai.New` —
everything else is identical. The same agent code runs unchanged against all
providers, verified by a shared conformance test suite.

### Streaming

```go
for chunk, err := range ai.GenerateStream(ctx, model, "Tell me a story.") {
	if err != nil { /* handle */ }
	fmt.Print(chunk.Text)
}
// or: text, err := ai.GenerateStream(ctx, model, "...").Text()
```

### Structured output (generics)

```go
type Invoice struct {
	Number string  `json:"number" desc:"The invoice number"`
	Total  float64 `json:"total"`
	Status string  `json:"status" enum:"draft,sent,paid"`
}

inv, err := ai.GenerateObject[Invoice](ctx, model, "Extract: "+email)
```

A JSON Schema is derived from the struct via reflection and tags, sent to the
model, and the output is validated and unmarshaled. Validation failures are fed
back to the model in a configurable repair loop (`ai.WithObjectRepairs`).

### Agents with tools

```go
weather := tool.MustNew("get_weather", "Weather for a city",
	func(ctx context.Context, in struct {
		City string `json:"city"`
	}) (string, error) { return lookup(in.City) })

a := agent.New(model,
	agent.WithTools(weather),
	agent.WithMaxIterations(6),               // hard cap: typed error, not a spin
	agent.WithTokenBudget(20_000),            // per-run budget
	agent.WithApproval(reviewToolCall),       // human gate for side effects
	agent.WithMemory(memory.NewInMemory(), "conv-1"), // remembers across runs
)
res, err := a.Run(ctx, "What's the weather in Cairo in Fahrenheit?")
```

Tool input schemas are derived from Go function signatures. Context
cancellation is honored mid-loop; long conversations degrade via pluggable
strategies (`memory.Truncate`, `memory.Summarize`) — never silently.

### RAG

```go
emb, _ := embedder.Embed(ctx, texts)                 // openai or gemini embedder
store.Upsert(ctx, emb.Model, docs...)                // model recorded with vectors
matches, _ := store.Query(ctx, vector.Query{         // refuses cross-model queries
	Embedding: qv, Model: emb.Model, TopK: 3,
	Filters: []vector.Filter{vector.Where("topic", "go")},
})
```

### Resilience

```go
model := resilience.NewFallback(resilience.FallbackConfig{
	OnFailover: func(from, to string, err error) { log.Printf("%s -> %s: %v", from, to, err) },
},
	resilience.NewRetry(anthropicClient, resilience.RetryConfig{}), // backoff + jitter
	openaiClient,
)
```

Retry, fallback, circuit breaker (`resilience.NewBreaker`), and client-side
rate limiting (`resilience.NewRateLimiter`) — all explicit, configurable, and
observable via callbacks. No hidden retries anywhere in the library.

### Observability

```go
tracker := middleware.NewUsageTracker()
model = middleware.Chain(model,
	middleware.Logging(slog.Default()),
	tracker.Middleware(),
)
// later: tracker.Total(), tracker.PerModel(), tracker.Cost(pricing)
```

### Normalized errors

```go
if errors.Is(err, ai.ErrRateLimited) { ... }  // same check for every provider
var apiErr *ai.APIError
if errors.As(err, &apiErr) { fmt.Println(apiErr.StatusCode, apiErr.RetryAfter) }
```

### Testing without a network

Every interface ships a first-class fake in `aitest`: a scripted `ChatModel`, a
deterministic `Embedder`, a `Reranker`, and a fake HTTP transport for testing
real provider clients offline. `go test ./...` for this entire repository needs
no API key and no network. A shared conformance suite
(`aitest.RunConformance`) keeps every provider behaviorally identical.

## Feature status

| Feature | Status |
| --- | --- |
| Canonical message model (text, image, tool-call, tool-result) | ✅ |
| Text generation + streaming iterators | ✅ |
| Structured output (`GenerateObject[T]`, schema derivation, repair loop) | ✅ |
| Tools + agent loop (caps, budgets, approval hook, cancellation) | ✅ |
| Conversation memory + context strategies (truncate, summarize) | ✅ |
| Providers: Anthropic, OpenAI, Gemini, Groq, xAI | ✅ |
| Embeddings (OpenAI, Gemini) + in-memory vector store + reranker interface | ✅ |
| Resilience: retry, fallback, circuit breaker, rate limiting | ✅ |
| Observability: slog, hooks, usage/cost tracking | ✅ |
| Provider conformance test suite + fakes (`aitest`) | ✅ |
| SQL-backed stores (pgvector, sqlite) as submodules | 🔜 |
| OpenTelemetry tracing submodule | 🔜 |
| Multimodal beyond image input (audio, files) | 🔜 |
| MCP client support | 🔜 |

The deferred items are separate submodules or lower-priority phases by design:
the core stays dependency-free, and `middleware.Hooks` is the extension point
the OTel submodule will build on.

## Package layout

```
.                       core: types, options, errors, Generate/GenerateStream/GenerateObject
schema                  Go struct -> JSON Schema (reflection + tags) + validation
tool                    Tool interface, typed constructor, registry
agent                   the agent loop + safeguards
memory                  ConversationStore + in-memory impl + context strategies
vector                  VectorStore + in-memory impl (model-checked queries)
rerank                  Reranker interface
resilience              retry, fallback, circuit breaker, rate limiter
middleware              slog logging, hooks, usage/cost tracking
provider/anthropic      Anthropic Messages API
provider/openai         OpenAI Chat Completions + Embeddings
provider/gemini         Google Gemini generateContent + embeddings
provider/groq           Groq (OpenAI-compatible, thin wrapper)
provider/xai            xAI (OpenAI-compatible, thin wrapper)
aitest                  fakes, fake transport, conformance suite
examples/               runnable examples
```

## Design principles

- **Idiomatic Go** — constructor injection, small consumer-side interfaces, no
  global state, no `Init()`.
- **Minimal dependencies** — the core is stdlib-only; integrations that need
  dependencies go in separate submodules.
- **`context.Context` first** on every I/O call; cancellation honored mid-agent-loop.
- **Errors are typed values** — one taxonomy across all providers.
- **Streaming via iterators**, not callbacks.
- **Nothing silent** — retries, failovers, truncation, and message drops are all
  observable through explicit configuration.
- **No `map[string]any` APIs** — provider-specific knobs are typed options.

## Testing

```sh
go test ./...                 # everything, offline, no keys
go test -race ./...
go test -run x -fuzz FuzzParseSSE ./provider/anthropic
go test -run x -fuzz FuzzValidate ./schema
ANTHROPIC_API_KEY=... go test -tags integration ./provider/anthropic
```

## License

TBD.
