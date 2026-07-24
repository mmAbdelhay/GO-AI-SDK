// Module otel provides OpenTelemetry tracing for github.com/mmabdelhay/go-ai-sdk.
// It is a separate submodule so the core stays dependency-free (design principle
// P2): only users who want tracing pull the OpenTelemetry dependency tree.
module github.com/mmabdelhay/go-ai-sdk/otel

go 1.23

require (
	github.com/mmabdelhay/go-ai-sdk v0.0.0
	go.opentelemetry.io/otel v1.34.0
	go.opentelemetry.io/otel/sdk v1.34.0
	go.opentelemetry.io/otel/trace v1.34.0
)

require (
	github.com/go-logr/logr v1.4.2 // indirect
	github.com/go-logr/stdr v1.2.2 // indirect
	github.com/google/uuid v1.6.0 // indirect
	go.opentelemetry.io/auto/sdk v1.1.0 // indirect
	go.opentelemetry.io/otel/metric v1.34.0 // indirect
	golang.org/x/sys v0.29.0 // indirect
)

// Until go-ai-sdk is tagged and published, resolve it from the repo root.
replace github.com/mmabdelhay/go-ai-sdk => ../
