package memory

import (
	"context"
	"fmt"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// Strategy shrinks a conversation to fit a token budget before it is sent to a
// model. Implementations must not mutate the input slice.
type Strategy interface {
	// Fit returns a message list estimated to fit within budget tokens.
	Fit(ctx context.Context, msgs []ai.Message, budget int) ([]ai.Message, error)
}

// EstimateTokens crudely estimates the token count of messages (about four
// characters per token, plus per-message overhead). It exists so strategies can
// budget without a provider tokenizer; providers report exact usage after the
// fact.
func EstimateTokens(msgs []ai.Message) int {
	total := 0
	for _, m := range msgs {
		total += 4 // per-message framing overhead
		for _, p := range m.Content {
			switch v := p.(type) {
			case ai.Text:
				total += len(v.Text) / 4
			case ai.ToolCall:
				total += (len(v.Name) + len(v.Input)) / 4
			case ai.ToolResult:
				total += EstimateTokens([]ai.Message{{Content: v.Content}})
			case ai.Image:
				total += 1000 // coarse image placeholder
			}
		}
	}
	return total
}

// Truncate is a [Strategy] that drops the oldest non-system messages until the
// conversation fits the budget. System messages are always kept. Dropping is
// observable: the count of dropped messages is surfaced via OnDrop when set.
type Truncate struct {
	// OnDrop, when set, is invoked with the number of messages dropped.
	OnDrop func(dropped int)
}

// Fit implements [Strategy].
func (t Truncate) Fit(_ context.Context, msgs []ai.Message, budget int) ([]ai.Message, error) {
	if EstimateTokens(msgs) <= budget {
		return msgs, nil
	}
	var system, rest []ai.Message
	for _, m := range msgs {
		if m.Role == ai.RoleSystem {
			system = append(system, m)
		} else {
			rest = append(rest, m)
		}
	}
	dropped := 0
	for len(rest) > 1 && EstimateTokens(append(system, rest...)) > budget {
		rest = rest[1:]
		dropped++
	}
	// Never lead with an orphaned tool result whose call was dropped.
	for len(rest) > 1 && leadsWithToolResult(rest[0]) {
		rest = rest[1:]
		dropped++
	}
	if t.OnDrop != nil && dropped > 0 {
		t.OnDrop(dropped)
	}
	return append(system, rest...), nil
}

func leadsWithToolResult(m ai.Message) bool {
	for _, p := range m.Content {
		if _, ok := p.(ai.ToolResult); ok {
			return true
		}
	}
	return false
}

// Summarize is a [Strategy] that condenses older messages into a single summary
// message using a model, keeping the most recent Keep messages verbatim.
type Summarize struct {
	// Model produces the summaries.
	Model ai.ChatModel
	// Keep is how many trailing messages survive verbatim (default 4).
	Keep int
	// MaxSummaryTokens caps the summary length (default 500).
	MaxSummaryTokens int
}

// Fit implements [Strategy].
func (s Summarize) Fit(ctx context.Context, msgs []ai.Message, budget int) ([]ai.Message, error) {
	if EstimateTokens(msgs) <= budget {
		return msgs, nil
	}
	keep := s.Keep
	if keep <= 0 {
		keep = 4
	}
	if len(msgs) <= keep {
		return msgs, nil
	}
	maxTok := s.MaxSummaryTokens
	if maxTok <= 0 {
		maxTok = 500
	}

	older, recent := msgs[:len(msgs)-keep], msgs[len(msgs)-keep:]
	var transcript string
	for _, m := range older {
		for _, p := range m.Content {
			if t, ok := p.(ai.Text); ok {
				transcript += string(m.Role) + ": " + t.Text + "\n"
			}
		}
	}

	resp, err := ai.Generate(ctx, s.Model,
		"Summarize this conversation so it can serve as context for continuing it. "+
			"Preserve names, decisions, and open questions.\n\n"+transcript,
		ai.WithMaxTokens(maxTok))
	if err != nil {
		return nil, fmt.Errorf("memory: summarizing: %w", err)
	}

	out := []ai.Message{ai.UserText("Summary of the conversation so far: " + resp.Text())}
	return append(out, recent...), nil
}
