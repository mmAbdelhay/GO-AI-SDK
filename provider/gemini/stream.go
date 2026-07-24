package gemini

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// parseSSE reads a streamGenerateContent?alt=sse stream: each data line is a
// gResponse fragment whose candidate parts carry text deltas; usage metadata
// accumulates across fragments and the finish reason arrives on the last one.
func parseSSE(r io.Reader, yield func(ai.Chunk, error) bool) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var (
		usage      ai.Usage
		stopReason ai.StopReason = ai.StopEndTurn
	)

	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(line[len("data:"):])
		var frag gResponse
		if err := json.Unmarshal([]byte(payload), &frag); err != nil {
			yield(ai.Chunk{}, fmt.Errorf("gemini: decoding stream chunk: %w", err))
			return
		}
		if frag.UsageMetadata != nil {
			usage = ai.Usage{
				InputTokens:     frag.UsageMetadata.PromptTokenCount,
				OutputTokens:    frag.UsageMetadata.CandidatesTokenCount,
				CacheReadTokens: frag.UsageMetadata.CachedContentTokenCount,
			}
		}
		for _, cand := range frag.Candidates {
			if cand.FinishReason != "" {
				stopReason = mapFinishReason(cand.FinishReason, ai.Message{})
			}
			for _, p := range cand.Content.Parts {
				if p.Text != "" {
					if !yield(ai.Chunk{Type: ai.ChunkText, Text: p.Text}, nil) {
						return
					}
				}
			}
		}
	}
	if err := sc.Err(); err != nil {
		yield(ai.Chunk{}, fmt.Errorf("gemini: reading stream: %w", err))
		return
	}
	u := usage
	yield(ai.Chunk{Type: ai.ChunkDone, Usage: &u, StopReason: stopReason}, nil)
}
