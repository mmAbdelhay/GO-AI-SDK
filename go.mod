module github.com/mmabdelhay/go-ai-sdk

// Phase 1 targets Go 1.23 for range-over-func iterators (iter.Seq2).
// The core module intentionally has zero non-stdlib dependencies (design
// principle P2): optional integrations live in separate submodules.
go 1.23
