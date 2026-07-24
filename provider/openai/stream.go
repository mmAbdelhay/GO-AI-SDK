package openai

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// parseSSE reads a Chat Completions SSE stream and drives yield. Text deltas
// arrive in choices[0].delta.content; the finish reason arrives on the final
// content chunk; usage arrives in a trailing chunk (stream_options.include_usage
// is always set by this client) whose choices array is empty. The stream ends
// with a literal "data: [DONE]" line.
func (c *Client) parseSSE(r io.Reader, yield func(ai.Chunk, error) bool) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var (
		usage      ai.Usage
		stopReason ai.StopReason
	)
	emitDone := func() {
		u := usage
		yield(ai.Chunk{Type: ai.ChunkDone, Usage: &u, StopReason: stopReason}, nil)
	}

	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(line[len("data:"):])
		if payload == "[DONE]" {
			emitDone()
			return
		}
		var chunk oaStreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			yield(ai.Chunk{}, fmt.Errorf("%s: decoding stream chunk: %w", c.provider, err))
			return
		}
		if chunk.Usage != nil {
			usage = mapUsage(*chunk.Usage)
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		choice := chunk.Choices[0]
		if choice.FinishReason != "" {
			stopReason = mapFinishReason(choice.FinishReason)
		}
		if choice.Delta.Content != "" {
			if !yield(ai.Chunk{Type: ai.ChunkText, Text: choice.Delta.Content}, nil) {
				return
			}
		}
	}
	if err := sc.Err(); err != nil {
		yield(ai.Chunk{}, fmt.Errorf("%s: reading stream: %w", c.provider, err))
		return
	}
	// EOF without [DONE]; still deliver the terminal chunk.
	emitDone()
}
