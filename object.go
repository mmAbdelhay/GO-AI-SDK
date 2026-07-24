package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/mmabdelhay/go-ai-sdk/schema"
)

// defaultObjectRepairs is how many repair rounds GenerateObject attempts when
// the model's output fails validation, unless overridden by WithObjectRepairs.
const defaultObjectRepairs = 2

// GenerateObject generates a value of type T. It derives a JSON Schema from T
// (see the schema package for the recognized struct tags), instructs the model
// to produce conforming JSON, validates the output, and unmarshals it into T.
//
// When the output fails validation, the validation error is fed back to the
// model and the generation retried, up to the repair budget configured with
// [WithObjectRepairs] (default 2). If every attempt fails, the returned error
// wraps [ErrInvalidSchema].
//
// It is named GenerateObject rather than Generate because Go does not allow a
// generic function to share the name of the existing [Generate].
func GenerateObject[T any](ctx context.Context, m ChatModel, prompt string, opts ...Option) (T, error) {
	var zero T

	s, err := schema.For[T]()
	if err != nil {
		return zero, fmt.Errorf("ai: deriving schema: %w", err)
	}

	req := buildRequest(prompt, opts)
	repairs := req.objectRepairs
	switch {
	case repairs == 0:
		repairs = defaultObjectRepairs
	case repairs < 0:
		repairs = 0
	}

	instruction := "Respond with a single JSON value that conforms to this JSON Schema. " +
		"Output only the JSON value itself: no prose, no markdown fences, no explanations.\n\nJSON Schema:\n" +
		string(s.JSON())
	if req.System == "" {
		req.System = instruction
	} else {
		req.System += "\n\n" + instruction
	}

	var lastErr error
	for attempt := 0; attempt <= repairs; attempt++ {
		resp, err := m.Generate(ctx, req)
		if err != nil {
			return zero, err
		}

		raw := ExtractJSON(resp.Text())
		if verr := schema.Validate(s, raw); verr != nil {
			lastErr = verr
		} else if uerr := json.Unmarshal(raw, &zero); uerr != nil {
			lastErr = fmt.Errorf("unmarshaling into %T: %w", zero, uerr)
		} else {
			return zero, nil
		}

		// Feed the failure back and try again.
		var fresh T
		zero = fresh
		req.Messages = append(req.Messages,
			resp.Message,
			UserText(fmt.Sprintf(
				"Your response was invalid: %v. Respond again with only a JSON value that conforms to the schema.",
				lastErr)),
		)
	}

	return zero, fmt.Errorf("%w after %d attempt(s): %v", ErrInvalidSchema, repairs+1, lastErr)
}

// ExtractJSON extracts the JSON payload from model output that may be wrapped in
// markdown fences or surrounded by prose. It returns the input unchanged when no
// clear JSON value is found, leaving the error to schema validation.
func ExtractJSON(text string) []byte {
	t := strings.TrimSpace(text)

	// Prefer a fenced block when present.
	if i := strings.Index(t, "```"); i >= 0 {
		rest := t[i+3:]
		if nl := strings.IndexByte(rest, '\n'); nl >= 0 {
			rest = rest[nl+1:] // drop the language line ("json")
		}
		if end := strings.Index(rest, "```"); end >= 0 {
			t = strings.TrimSpace(rest[:end])
		}
	}

	if strings.HasPrefix(t, "{") || strings.HasPrefix(t, "[") {
		return []byte(t)
	}

	// Fall back to the outermost braces/brackets in surrounding prose.
	for _, pair := range [][2]byte{{'{', '}'}, {'[', ']'}} {
		start := strings.IndexByte(t, pair[0])
		end := strings.LastIndexByte(t, pair[1])
		if start >= 0 && end > start {
			return []byte(t[start : end+1])
		}
	}
	return []byte(t)
}
