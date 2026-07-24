package anthropic

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// sseEvent is the decoded JSON payload of an Anthropic streaming event. Only the
// fields this package consumes are declared; every event type carries a "type"
// discriminator.
type sseEvent struct {
	Type    string `json:"type"`
	Message *struct {
		Usage wireUsage `json:"usage"`
	} `json:"message"`
	Delta *struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		PartialJSON string `json:"partial_json"`
		StopReason  string `json:"stop_reason"`
	} `json:"delta"`
	Usage *wireUsage `json:"usage"`
	Error *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// parseSSE reads the Anthropic server-sent-events stream from r and drives yield.
// It accumulates input tokens from message_start and output tokens plus the stop
// reason from message_delta, then emits a terminal ai.ChunkDone. If yield returns
// false the parser stops early.
func parseSSE(r io.Reader, yield func(ai.Chunk, error) bool) {
	br := bufio.NewReader(r)

	var (
		data       strings.Builder
		usage      ai.Usage
		stopReason ai.StopReason
	)

	// emitDone reports the terminal chunk. Returns false if iteration should end.
	emitDone := func() {
		u := usage
		yield(ai.Chunk{Type: ai.ChunkDone, Usage: &u, StopReason: stopReason}, nil)
	}

	// dispatch processes one complete event's accumulated data payload. It
	// returns (stop, done): stop when the consumer cancelled, done when the
	// stream is logically finished.
	dispatch := func(payload string) (stop, done bool) {
		if payload == "" {
			return false, false
		}
		var ev sseEvent
		if err := json.Unmarshal([]byte(payload), &ev); err != nil {
			return !yield(ai.Chunk{}, fmt.Errorf("anthropic: decoding stream event: %w", err)), true
		}
		switch ev.Type {
		case "message_start":
			if ev.Message != nil {
				usage.InputTokens = ev.Message.Usage.InputTokens
				usage.CacheCreationTokens = ev.Message.Usage.CacheCreationInputTokens
				usage.CacheReadTokens = ev.Message.Usage.CacheReadInputTokens
			}
		case "content_block_delta":
			if ev.Delta != nil && ev.Delta.Type == "text_delta" && ev.Delta.Text != "" {
				if !yield(ai.Chunk{Type: ai.ChunkText, Text: ev.Delta.Text}, nil) {
					return true, false
				}
			}
		case "message_delta":
			if ev.Delta != nil && ev.Delta.StopReason != "" {
				stopReason = mapStopReason(ev.Delta.StopReason)
			}
			if ev.Usage != nil {
				usage.OutputTokens = ev.Usage.OutputTokens
			}
		case "error":
			var etype, msg string
			if ev.Error != nil {
				etype, msg = ev.Error.Type, ev.Error.Message
			}
			apiErr := ai.NewAPIError(providerName, 0, sentinelForType(etype, msg))
			apiErr.Type = etype
			apiErr.Message = msg
			yield(ai.Chunk{}, apiErr)
			return false, true
		case "message_stop":
			emitDone()
			return false, true
		}
		return false, false
	}

	for {
		line, err := br.ReadString('\n')
		trimmed := strings.TrimRight(line, "\r\n")

		switch {
		case strings.HasPrefix(trimmed, "data:"):
			// Accumulate the payload; SSE allows multiple data lines per event.
			data.WriteString(strings.TrimSpace(trimmed[len("data:"):]))
		case trimmed == "":
			// Blank line terminates an event.
			if data.Len() > 0 {
				stop, done := dispatch(data.String())
				data.Reset()
				if stop || done {
					return
				}
			}
		default:
			// event:, id:, retry:, and comment (":") lines are ignored; the JSON
			// payload carries its own type discriminator.
		}

		if err != nil {
			// Flush any trailing buffered event before finishing.
			if data.Len() > 0 {
				if stop, done := dispatch(data.String()); stop || done {
					return
				}
			}
			if err != io.EOF {
				yield(ai.Chunk{}, fmt.Errorf("anthropic: reading stream: %w", err))
				return
			}
			// Stream ended without an explicit message_stop; emit the terminal
			// chunk so consumers always observe usage and stop reason.
			emitDone()
			return
		}
	}
}
