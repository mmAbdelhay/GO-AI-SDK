# Examples

Runnable examples for the Go AI SDK. All require an Anthropic API key:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
```

| Example | What it shows | Run |
| --- | --- | --- |
| [`generate`](./generate) | One-shot text generation with options, usage, and typed error handling | `go run ./examples/generate` |
| [`stream`](./stream) | Streaming with the `range`-over-func iterator | `go run ./examples/stream` |

For a **no-network, no-key** example (using the `aitest` fakes), see the testable
examples in the root package:

```sh
go test -run Example -v
```
