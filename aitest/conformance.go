package aitest

import (
	"context"
	"errors"
	"net/http"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// Conformance is the shared behavioral test suite every provider implementation
// must pass. A provider supplies canned wire-format responses that encode a
// fixed scenario; the suite asserts the provider translates them into identical
// canonical behavior. Fixture contract:
//
//   - GenerateResponse: a 200 response whose body decodes to the assistant text
//     "Hello, world" with usage input=9/output=7 and a natural stop.
//   - StreamResponse: a 200 SSE response streaming "Hello" then ", world" and
//     terminating with usage input=9/output=7.
//   - RateLimitResponse: a 429 response in the provider's error format.
//
// Responses are constructed fresh per test via functions because http.Response
// bodies are single-use.
type Conformance struct {
	// NewClient builds the provider client under test on top of hc.
	NewClient func(hc *http.Client) ai.ChatModel

	GenerateResponse  func() *http.Response
	StreamResponse    func() *http.Response
	RateLimitResponse func() *http.Response
}

// RunConformance runs the shared provider test suite.
func RunConformance(t *testing.T, c Conformance) {
	t.Helper()
	ctx := context.Background()

	t.Run("Generate", func(t *testing.T) {
		tr := &Transport{Responses: []*http.Response{c.GenerateResponse()}}
		model := c.NewClient(tr.Client())

		resp, err := ai.Generate(ctx, model, "hi")
		if err != nil {
			t.Fatal(err)
		}
		if resp.Text() != "Hello, world" {
			t.Errorf("text = %q, want %q", resp.Text(), "Hello, world")
		}
		if resp.StopReason != ai.StopEndTurn {
			t.Errorf("stop = %q, want end_turn", resp.StopReason)
		}
		if resp.Usage.InputTokens != 9 || resp.Usage.OutputTokens != 7 {
			t.Errorf("usage = %+v, want in=9 out=7", resp.Usage)
		}
		if resp.Message.Role != ai.RoleAssistant {
			t.Errorf("role = %q", resp.Message.Role)
		}
	})

	t.Run("Stream", func(t *testing.T) {
		tr := &Transport{Responses: []*http.Response{c.StreamResponse()}}
		model := c.NewClient(tr.Client())

		var text string
		var done *ai.Chunk
		for chunk, err := range ai.GenerateStream(ctx, model, "hi") {
			if err != nil {
				t.Fatal(err)
			}
			switch chunk.Type {
			case ai.ChunkText:
				text += chunk.Text
			case ai.ChunkDone:
				d := chunk
				done = &d
			}
		}
		if text != "Hello, world" {
			t.Errorf("streamed text = %q, want %q", text, "Hello, world")
		}
		if done == nil {
			t.Fatal("no terminal ChunkDone")
		}
		if done.Usage == nil || done.Usage.InputTokens != 9 || done.Usage.OutputTokens != 7 {
			t.Errorf("terminal usage = %+v, want in=9 out=7", done.Usage)
		}
	})

	t.Run("StreamEarlyStop", func(t *testing.T) {
		tr := &Transport{Responses: []*http.Response{c.StreamResponse()}}
		model := c.NewClient(tr.Client())

		count := 0
		for range ai.GenerateStream(ctx, model, "hi") {
			count++
			break // consumer stops early; producer must not panic
		}
		if count != 1 {
			t.Errorf("chunks before break = %d, want 1", count)
		}
	})

	t.Run("RateLimited", func(t *testing.T) {
		tr := &Transport{Responses: []*http.Response{c.RateLimitResponse()}}
		model := c.NewClient(tr.Client())

		_, err := ai.Generate(ctx, model, "hi")
		if !errors.Is(err, ai.ErrRateLimited) {
			t.Errorf("err = %v, want ErrRateLimited", err)
		}
		var apiErr *ai.APIError
		if !errors.As(err, &apiErr) {
			t.Errorf("error is not *ai.APIError: %v", err)
		}
	})

	t.Run("ContextCancelled", func(t *testing.T) {
		tr := &Transport{Responses: []*http.Response{c.GenerateResponse()}}
		model := c.NewClient(tr.Client())

		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		if _, err := ai.Generate(cancelled, model, "hi"); !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v, want context.Canceled", err)
		}
	})
}
