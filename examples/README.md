# Examples

Runnable examples for the Go AI SDK.

```sh
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...      # only for rag and resilience
```

| Example | What it shows | Run |
| --- | --- | --- |
| [`generate`](./generate) | One-shot text generation with options, usage, and typed error handling | `go run ./examples/generate` |
| [`stream`](./stream) | Streaming with the `range`-over-func iterator | `go run ./examples/stream` |
| [`object`](./object) | Structured output: `GenerateObject[Invoice]` with schema-tagged structs | `go run ./examples/object` |
| [`agent`](./agent) | Agent loop with tools, approval hook, iteration cap, and token budget | `go run ./examples/agent` |
| [`rag`](./rag) | End-to-end RAG: embeddings + vector store + generation | `go run ./examples/rag` |
| [`resilience`](./resilience) | Retry with backoff + provider failover, with events surfaced | `go run ./examples/resilience` |

For **no-network, no-key** examples (using the `aitest` fakes), see the testable
examples in the root and agent packages:

```sh
go test -run Example -v ./...
```
