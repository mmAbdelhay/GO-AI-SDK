// Package aitest provides first-class fakes for testing code that uses the ai
// package without any network access or API keys.
//
// [Model] is a scripted [ai.ChatModel]: it returns queued responses (or invokes
// a function) and records every request for assertions. [Transport] is a fake
// http.RoundTripper for exercising real provider clients against canned HTTP
// responses.
package aitest

import (
	"context"
	"fmt"
	"sync"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// Model is a scripted fake implementing [ai.ChatModel]. It is safe for
// concurrent use. Configure it in one of two ways:
//
//   - Set Responses to a queue consumed in order by successive Generate calls.
//   - Set GenerateFunc for full control over each response.
//
// Every call is recorded in Requests, and streaming is derived automatically
// from the response text unless StreamFunc is set.
type Model struct {
	// ModelName is returned by Name. Defaults to "fake" when empty.
	ModelName string

	// Responses is a queue of responses returned by successive Generate calls.
	// When exhausted, Generate returns ErrNoResponse unless GenerateFunc is set.
	Responses []ai.Response

	// Err, when non-nil, is returned by every Generate call (and as the stream's
	// first error), taking precedence over Responses and GenerateFunc.
	Err error

	// GenerateFunc, when set, handles every Generate call instead of the queue.
	GenerateFunc func(ctx context.Context, req ai.Request) (ai.Response, error)

	// StreamFunc, when set, handles every Stream call. Otherwise Stream replays
	// the corresponding Generate response as text chunks followed by a done
	// chunk.
	StreamFunc func(ctx context.Context, req ai.Request) ai.Stream

	mu       sync.Mutex
	requests []ai.Request
	next     int
}

// ErrNoResponse is returned by [Model.Generate] when the scripted queue is
// exhausted and no GenerateFunc is configured.
var ErrNoResponse = fmt.Errorf("aitest: no scripted response available")

// Name implements [ai.ChatModel].
func (m *Model) Name() string {
	if m.ModelName != "" {
		return m.ModelName
	}
	return "fake"
}

// Generate implements [ai.ChatModel]. It records req and returns the next
// scripted response.
func (m *Model) Generate(ctx context.Context, req ai.Request) (ai.Response, error) {
	m.mu.Lock()
	m.requests = append(m.requests, req)
	m.mu.Unlock()

	if m.Err != nil {
		return ai.Response{}, m.Err
	}
	if m.GenerateFunc != nil {
		return m.GenerateFunc(ctx, req)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.next >= len(m.Responses) {
		return ai.Response{}, ErrNoResponse
	}
	resp := m.Responses[m.next]
	m.next++
	return resp, nil
}

// Stream implements [ai.ChatModel]. Unless StreamFunc is set, it records req and
// replays the next scripted response as text chunks.
func (m *Model) Stream(ctx context.Context, req ai.Request) ai.Stream {
	if m.StreamFunc != nil {
		return m.StreamFunc(ctx, req)
	}
	return func(yield func(ai.Chunk, error) bool) {
		resp, err := m.Generate(ctx, req)
		if err != nil {
			yield(ai.Chunk{}, err)
			return
		}
		if !yield(ai.Chunk{Type: ai.ChunkText, Text: resp.Text()}, nil) {
			return
		}
		u := resp.Usage
		yield(ai.Chunk{Type: ai.ChunkDone, Usage: &u, StopReason: resp.StopReason}, nil)
	}
}

// Requests returns a copy of the recorded requests, in call order.
func (m *Model) Requests() []ai.Request {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]ai.Request, len(m.requests))
	copy(out, m.requests)
	return out
}

// LastRequest returns the most recently recorded request and whether one exists.
func (m *Model) LastRequest() (ai.Request, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.requests) == 0 {
		return ai.Request{}, false
	}
	return m.requests[len(m.requests)-1], true
}

// CallCount returns how many times Generate has recorded a request.
func (m *Model) CallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.requests)
}

// TextResponse builds a simple assistant text [ai.Response], for scripting.
func TextResponse(text string) ai.Response {
	return ai.Response{
		Message:    ai.AssistantText(text),
		StopReason: ai.StopEndTurn,
	}
}

var _ ai.ChatModel = (*Model)(nil)
