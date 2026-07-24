package aitest

import ai "github.com/mmabdelhay/go-ai-sdk"

// StreamOf builds an [ai.Stream] that yields each of texts as a text chunk,
// followed by a terminal done chunk with StopReason end_turn. It is a convenience
// for scripting streaming responses in tests.
func StreamOf(texts ...string) ai.Stream {
	return func(yield func(ai.Chunk, error) bool) {
		for _, t := range texts {
			if !yield(ai.Chunk{Type: ai.ChunkText, Text: t}, nil) {
				return
			}
		}
		yield(ai.Chunk{Type: ai.ChunkDone, StopReason: ai.StopEndTurn, Usage: &ai.Usage{}}, nil)
	}
}

// ErrorStream builds an [ai.Stream] that yields any leading texts and then fails
// with err, modeling a stream that breaks partway through.
func ErrorStream(err error, texts ...string) ai.Stream {
	return func(yield func(ai.Chunk, error) bool) {
		for _, t := range texts {
			if !yield(ai.Chunk{Type: ai.ChunkText, Text: t}, nil) {
				return
			}
		}
		yield(ai.Chunk{}, err)
	}
}
