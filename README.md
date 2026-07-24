# Go AI SDK

A unified, idiomatic Go interface for working with multiple AI providers. One
type-safe API for text generation, streaming, structured output, agents, tools,
embeddings, and more — designed the Go way, not ported from another ecosystem.

> **Status: Phase 1 (Core + Anthropic).** This is the foundation layer. Later
> phases add structured output, tools/agents, memory, more providers, RAG,
> resilience, and observability — see the [roadmap](#roadmap).

## Install

```sh
go get github.com/mmabdelhay/go-ai-sdk
```

Requires Go 1.23+ (range-over-func iterators). The core module has **zero
non-standard-library dependencies**.

## Quickstart

```go
package main

import (
	"context"
	"fmt"
	"os"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/provider/anthropic"
)

func main() {
	model := anthropic.New(os.Getenv("ANTHROPIC_API_KEY"),
		anthropic.WithModel("claude-sonnet-4-20250514"))

	resp, err := ai.Generate(context.Background(), model,
		"Write a haiku about Go.", ai.WithMaxTokens(200))
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Text())
}
```

### Streaming

`GenerateStream` returns an iterator you can `range` over directly, or drain with
`.Text()` / `.Collect()`:

```go
for chunk, err := range ai.GenerateStream(ctx, model, "Tell me a story.") {
	if err != nil { /* handle */ }
	fmt.Print(chunk.Text)
}

// or, if you don't care about streaming:
text, err := ai.GenerateStream(ctx, model, "Tell me a story.").Text()
```

### Normalized errors

Provider errors map onto a shared taxonomy, so detection is identical across
providers:

```go
if errors.Is(err, ai.ErrRateLimited) {
	// back off and retry
}
var apiErr *ai.APIError
if errors.As(err, &apiErr) {
	fmt.Println(apiErr.StatusCode, apiErr.RetryAfter)
}
```

### Testing without a network

Every interface ships with a first-class fake in the `aitest` package, so you can
test agents and app code with no API key and no network:

```go
model := &aitest.Model{Responses: []ai.Response{
	aitest.TextResponse("scripted reply"),
}}
resp, _ := ai.Generate(ctx, model, "hi")
// resp.Text() == "scripted reply"
// model.LastRequest(), model.CallCount() for assertions
```

## Design principles

- **Idiomatic Go, not a transliteration** — explicit constructor injection, small
  interfaces, no global state, no service container.
- **Minimal dependencies** — core is stdlib-only; optional integrations live in
  separate submodules so you pull only what you use.
- **`context.Context` first** on every I/O call.
- **Errors are typed values** — a normalized taxonomy, wrapped with `%w`.
- **Streaming via iterators** (`iter.Seq2`-shaped), not callbacks.
- **No hidden network calls or retries** — retry/timeout behavior is configured on
  the `*http.Client` you supply.
- **Provider-specific options stay type-safe** — no `map[string]any` config blob.

## Feature status

| Feature | Status |
| --- | --- |
| Canonical message model (text, image, tool-call, tool-result) | ✅ Phase 1 |
| Text generation (`Generate`) | ✅ Phase 1 |
| Streaming (`GenerateStream`, `Stream.Text`/`Collect`) | ✅ Phase 1 |
| Functional + typed provider options | ✅ Phase 1 |
| Normalized error taxonomy | ✅ Phase 1 |
| Token usage reporting | ✅ Phase 1 |
| Anthropic provider | ✅ Phase 1 |
| Test fakes (`aitest`) | ✅ Phase 1 |
| Structured output (`Generate[T]`, JSON Schema) | 🔜 Phase 2 |
| Tools + agent loop | 🔜 Phase 3 |
| Conversation memory | 🔜 Phase 4 |
| OpenAI / Gemini / Groq / xAI providers | 🔜 Phase 5 |
| Embeddings, vector store, reranking | 🔜 Phase 6 |
| Fallback, retry, rate limiting | 🔜 Phase 7 |
| Observability (`slog`, OpenTelemetry) | 🔜 Phase 8 |
| Multimodal (audio, files) | 🔜 Phase 9 |
| MCP client support | 🔜 Phase 10 |

## Roadmap

The library is built in reviewable phases. Phase 1 establishes the core
abstractions — the canonical message model, the `ChatModel` interface, streaming,
options, errors, and one complete provider — that every later phase builds on.

## Package layout

```
.                       core types, options, errors, entrypoints (package ai)
provider/anthropic      Anthropic Messages API implementation of ai.ChatModel
aitest                  scripted fakes + fake HTTP transport for offline tests
examples/               runnable examples
```

## Testing

```sh
go test ./...                                   # unit tests, no network or key
go test -race ./...                             # with the race detector
go test -run x -fuzz FuzzParseSSE ./provider/anthropic   # fuzz the SSE parser
ANTHROPIC_API_KEY=... go test -tags integration ./provider/anthropic  # live API
```

## License

TBD.
