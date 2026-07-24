package anthropic

import (
	"bytes"
	"errors"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

func collectStream(t *testing.T, sse []byte) (chunks []ai.Chunk, err error) {
	t.Helper()
	parseSSE(bytes.NewReader(sse), func(c ai.Chunk, e error) bool {
		if e != nil {
			err = e
			return false
		}
		chunks = append(chunks, c)
		return true
	})
	return chunks, err
}

func TestParseSSE(t *testing.T) {
	chunks, err := collectStream(t, readTestdata(t, "stream.sse"))
	if err != nil {
		t.Fatal(err)
	}

	var text string
	var done *ai.Chunk
	for i := range chunks {
		switch chunks[i].Type {
		case ai.ChunkText:
			text += chunks[i].Text
		case ai.ChunkDone:
			done = &chunks[i]
		}
	}
	if text != "Hello, world" {
		t.Errorf("streamed text = %q, want %q", text, "Hello, world")
	}
	if done == nil {
		t.Fatal("no terminal ChunkDone emitted")
	}
	if done.StopReason != ai.StopEndTurn {
		t.Errorf("stop reason = %q", done.StopReason)
	}
	// input from message_start, output from message_delta.
	if done.Usage == nil || done.Usage.InputTokens != 9 || done.Usage.OutputTokens != 7 {
		t.Errorf("usage = %+v", done.Usage)
	}
}

func TestParseSSEError(t *testing.T) {
	chunks, err := collectStream(t, readTestdata(t, "stream_error.sse"))
	if err == nil {
		t.Fatal("expected error from stream")
	}
	if !errors.Is(err, ai.ErrOverloaded) {
		t.Errorf("err = %v, want ErrOverloaded", err)
	}
	// The partial text before the error should still have been delivered.
	if len(chunks) == 0 || chunks[0].Text != "partial" {
		t.Errorf("expected partial text chunk before error, got %+v", chunks)
	}
}

func TestParseSSEEarlyStop(t *testing.T) {
	// A consumer that stops after the first chunk must not panic or hang.
	var count int
	parseSSE(bytes.NewReader(readTestdata(t, "stream.sse")), func(_ ai.Chunk, _ error) bool {
		count++
		return false // stop immediately
	})
	if count != 1 {
		t.Errorf("expected exactly 1 yield before stop, got %d", count)
	}
}
