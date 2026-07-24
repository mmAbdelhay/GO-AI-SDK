package anthropic

import (
	"strings"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// FuzzParseSSE ensures the streaming parser never panics on arbitrary input,
// which arrives over the network and must be treated as untrusted. Run with:
//
//	go test ./provider/anthropic -run x -fuzz FuzzParseSSE
func FuzzParseSSE(f *testing.F) {
	f.Add("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	f.Add("data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"x\"}}\n\n")
	f.Add("data: not json\n\n")
	f.Add("")
	f.Add("data:{\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"boom\"}}\n\n")

	f.Fuzz(func(t *testing.T, input string) {
		// Must not panic regardless of input; we ignore the yielded values.
		parseSSE(strings.NewReader(input), func(_ ai.Chunk, _ error) bool { return true })
	})
}
