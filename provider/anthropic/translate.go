package anthropic

import (
	"encoding/base64"
	"fmt"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// toWire translates a provider-neutral ai.Request into the Anthropic wire
// request. It extracts system content to the top-level system field, maps
// content parts to Anthropic content blocks, applies the client defaults for
// model and max_tokens, and folds in any recognized provider options.
//
// stream selects whether the request sets "stream": true.
func (c *Client) toWire(req ai.Request, stream bool) (wireRequest, error) {
	model := req.Model
	if model == "" {
		model = c.defaultModel
	}
	if model == "" {
		return wireRequest{}, fmt.Errorf("anthropic: no model set on request or client")
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = c.defaultMaxTokens
	}

	w := wireRequest{
		Model:         model,
		MaxTokens:     maxTokens,
		System:        req.System,
		Temperature:   req.Temperature,
		TopP:          req.TopP,
		StopSequences: req.StopSequences,
		Stream:        stream,
	}

	for _, t := range req.Tools {
		w.Tools = append(w.Tools, wireTool{Name: t.Name, Description: t.Description, InputSchema: t.InputSchema})
	}
	if tc, err := toWireToolChoice(req.ToolChoice); err != nil {
		return wireRequest{}, err
	} else if tc != nil {
		w.ToolChoice = tc
	}

	// Apply recognized provider options.
	for _, opt := range req.ProviderOptions {
		if o, ok := opt.(topKOption); ok {
			k := o.K
			w.TopK = &k
		}
	}

	for _, m := range req.Messages {
		if m.Role == ai.RoleSystem {
			// Anthropic has no system role; fold system messages into the
			// top-level system field.
			w.System = appendSystem(w.System, m)
			continue
		}
		wm, err := toWireMessage(m)
		if err != nil {
			return wireRequest{}, err
		}
		w.Messages = append(w.Messages, wm)
	}

	return w, nil
}

// toWireToolChoice maps the neutral tool choice onto Anthropic's tool_choice
// values ("required" is Anthropic's "any").
func toWireToolChoice(tc ai.ToolChoice) (*wireToolChoice, error) {
	switch tc.Mode {
	case "":
		return nil, nil
	case "auto":
		return &wireToolChoice{Type: "auto"}, nil
	case "required":
		return &wireToolChoice{Type: "any"}, nil
	case "none":
		return &wireToolChoice{Type: "none"}, nil
	case "tool":
		return &wireToolChoice{Type: "tool", Name: tc.Name}, nil
	default:
		return nil, fmt.Errorf("anthropic: unsupported tool choice mode %q", tc.Mode)
	}
}

func appendSystem(existing string, m ai.Message) string {
	var text string
	for _, p := range m.Content {
		if t, ok := p.(ai.Text); ok {
			text += t.Text
		}
	}
	if existing == "" {
		return text
	}
	if text == "" {
		return existing
	}
	return existing + "\n\n" + text
}

func toWireMessage(m ai.Message) (wireMessage, error) {
	role := string(m.Role)
	// The tool role is represented on the wire as a user message carrying
	// tool_result blocks, per Anthropic's convention.
	if m.Role == ai.RoleTool {
		role = "user"
	}
	wm := wireMessage{Role: role}
	for _, p := range m.Content {
		block, err := toWireBlock(p)
		if err != nil {
			return wireMessage{}, err
		}
		wm.Content = append(wm.Content, block)
	}
	return wm, nil
}

func toWireBlock(p ai.Part) (wireBlock, error) {
	switch v := p.(type) {
	case ai.Text:
		return wireBlock{Type: "text", Text: v.Text}, nil
	case ai.Image:
		if v.URL != "" {
			return wireBlock{Type: "image", Source: &wireImageSource{Type: "url", URL: v.URL}}, nil
		}
		return wireBlock{Type: "image", Source: &wireImageSource{
			Type:      "base64",
			MediaType: v.MediaType,
			Data:      base64.StdEncoding.EncodeToString(v.Data),
		}}, nil
	case ai.ToolCall:
		return wireBlock{Type: "tool_use", ID: v.ID, Name: v.Name, Input: v.Input}, nil
	case ai.ToolResult:
		content := make([]wireBlock, 0, len(v.Content))
		for _, cp := range v.Content {
			cb, err := toWireBlock(cp)
			if err != nil {
				return wireBlock{}, err
			}
			content = append(content, cb)
		}
		return wireBlock{Type: "tool_result", ToolUseID: v.CallID, Content: content, IsError: v.IsError}, nil
	default:
		return wireBlock{}, fmt.Errorf("anthropic: unsupported content part %T", p)
	}
}

// fromWire translates an Anthropic wire response into a provider-neutral
// ai.Response. raw is the untouched response body, preserved on the result.
func fromWire(w wireResponse, raw []byte) (ai.Response, error) {
	msg := ai.Message{Role: ai.RoleAssistant}
	for _, b := range w.Content {
		part, err := fromWireBlock(b)
		if err != nil {
			return ai.Response{}, err
		}
		msg.Content = append(msg.Content, part)
	}
	return ai.Response{
		Model:      w.Model,
		Message:    msg,
		StopReason: mapStopReason(w.StopReason),
		Usage:      mapUsage(w.Usage),
		Raw:        raw,
	}, nil
}

func fromWireBlock(b wireBlock) (ai.Part, error) {
	switch b.Type {
	case "text":
		return ai.Text{Text: b.Text}, nil
	case "tool_use":
		return ai.ToolCall{ID: b.ID, Name: b.Name, Input: b.Input}, nil
	case "image":
		if b.Source == nil {
			return ai.Image{}, nil
		}
		if b.Source.Type == "url" {
			return ai.Image{URL: b.Source.URL}, nil
		}
		data, err := base64.StdEncoding.DecodeString(b.Source.Data)
		if err != nil {
			return nil, fmt.Errorf("anthropic: decoding image data: %w", err)
		}
		return ai.Image{Data: data, MediaType: b.Source.MediaType}, nil
	default:
		return nil, fmt.Errorf("anthropic: unsupported response block type %q", b.Type)
	}
}

func mapStopReason(s string) ai.StopReason {
	switch s {
	case "end_turn":
		return ai.StopEndTurn
	case "max_tokens":
		return ai.StopMaxTokens
	case "stop_sequence":
		return ai.StopStopSequence
	case "tool_use":
		return ai.StopToolUse
	default:
		return ai.StopReason(s)
	}
}

func mapUsage(u wireUsage) ai.Usage {
	return ai.Usage{
		InputTokens:         u.InputTokens,
		OutputTokens:        u.OutputTokens,
		CacheCreationTokens: u.CacheCreationInputTokens,
		CacheReadTokens:     u.CacheReadInputTokens,
	}
}
