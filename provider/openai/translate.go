package openai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

func (c *Client) toWire(req ai.Request, stream bool) (oaRequest, error) {
	model := req.Model
	if model == "" {
		model = c.model
	}
	if model == "" {
		return oaRequest{}, fmt.Errorf("%s: no model set on request or client", c.provider)
	}

	w := oaRequest{
		Model:       model,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stop:        req.StopSequences,
		Stream:      stream,
	}
	if stream {
		w.StreamOptions = &oaStreamOpts{IncludeUsage: true}
	}
	if req.MaxTokens > 0 {
		if c.legacyMaxTokens {
			w.MaxTokens = &req.MaxTokens
		} else {
			w.MaxCompletionTokens = &req.MaxTokens
		}
	}

	if req.System != "" {
		w.Messages = append(w.Messages, oaMessage{Role: "system", Content: req.System})
	}
	for _, m := range req.Messages {
		wm, err := toWireMessages(m)
		if err != nil {
			return oaRequest{}, fmt.Errorf("%s: %w", c.provider, err)
		}
		w.Messages = append(w.Messages, wm...)
	}

	for _, t := range req.Tools {
		w.Tools = append(w.Tools, oaTool{Type: "function", Function: oaFunction{
			Name: t.Name, Description: t.Description, Parameters: t.InputSchema,
		}})
	}
	tc, err := toWireToolChoice(req.ToolChoice)
	if err != nil {
		return oaRequest{}, fmt.Errorf("%s: %w", c.provider, err)
	}
	w.ToolChoice = tc

	for _, opt := range req.ProviderOptions {
		if o, ok := opt.(seedOption); ok {
			s := o.Seed
			w.Seed = &s
		}
	}
	return w, nil
}

func toWireToolChoice(tc ai.ToolChoice) (json.RawMessage, error) {
	switch tc.Mode {
	case "":
		return nil, nil
	case "auto", "required", "none":
		return json.Marshal(tc.Mode)
	case "tool":
		return json.Marshal(map[string]any{
			"type":     "function",
			"function": map[string]string{"name": tc.Name},
		})
	default:
		return nil, fmt.Errorf("unsupported tool choice mode %q", tc.Mode)
	}
}

// toWireMessages translates one canonical message into one or more wire
// messages: tool results become individual role:"tool" messages, per the
// Chat Completions convention.
func toWireMessages(m ai.Message) ([]oaMessage, error) {
	var (
		textParts []oaContentPart
		toolCalls []oaToolCall
		toolMsgs  []oaMessage
		hasImage  bool
	)
	for _, p := range m.Content {
		switch v := p.(type) {
		case ai.Text:
			textParts = append(textParts, oaContentPart{Type: "text", Text: v.Text})
		case ai.Image:
			hasImage = true
			url := v.URL
			if url == "" {
				url = "data:" + v.MediaType + ";base64," + base64.StdEncoding.EncodeToString(v.Data)
			}
			textParts = append(textParts, oaContentPart{Type: "image_url", ImageURL: &oaImageURL{URL: url}})
		case ai.ToolCall:
			var tc oaToolCall
			tc.ID = v.ID
			tc.Type = "function"
			tc.Function.Name = v.Name
			tc.Function.Arguments = string(v.Input)
			toolCalls = append(toolCalls, tc)
		case ai.ToolResult:
			content := ""
			for _, cp := range v.Content {
				if t, ok := cp.(ai.Text); ok {
					content += t.Text
				}
			}
			if v.IsError {
				content = "ERROR: " + content
			}
			toolMsgs = append(toolMsgs, oaMessage{Role: "tool", ToolCallID: v.CallID, Content: content})
		default:
			return nil, fmt.Errorf("unsupported content part %T", p)
		}
	}

	var out []oaMessage
	if len(textParts) > 0 || len(toolCalls) > 0 {
		role := string(m.Role)
		if m.Role == ai.RoleTool {
			role = "user" // free-standing text in a tool-role message
		}
		wm := oaMessage{Role: role, ToolCalls: toolCalls}
		switch {
		case hasImage:
			wm.Content = textParts
		case len(textParts) > 0:
			text := ""
			for _, p := range textParts {
				text += p.Text
			}
			wm.Content = text
		}
		out = append(out, wm)
	}
	return append(out, toolMsgs...), nil
}

func (c *Client) fromWire(w oaResponse, raw []byte) (ai.Response, error) {
	if len(w.Choices) == 0 {
		return ai.Response{}, fmt.Errorf("%s: response has no choices", c.provider)
	}
	choice := w.Choices[0]
	msg := ai.Message{Role: ai.RoleAssistant}
	if choice.Message.Content != "" {
		msg.Content = append(msg.Content, ai.Text{Text: choice.Message.Content})
	}
	for _, tc := range choice.Message.ToolCalls {
		msg.Content = append(msg.Content, ai.ToolCall{
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: json.RawMessage(tc.Function.Arguments),
		})
	}
	return ai.Response{
		Model:      w.Model,
		Message:    msg,
		StopReason: mapFinishReason(choice.FinishReason),
		Usage:      mapUsage(w.Usage),
		Raw:        raw,
	}, nil
}

func mapFinishReason(s string) ai.StopReason {
	switch s {
	case "stop":
		return ai.StopEndTurn
	case "length":
		return ai.StopMaxTokens
	case "tool_calls", "function_call":
		return ai.StopToolUse
	default:
		return ai.StopReason(s)
	}
}

func mapUsage(u oaUsage) ai.Usage {
	return ai.Usage{
		InputTokens:     u.PromptTokens,
		OutputTokens:    u.CompletionTokens,
		CacheReadTokens: u.PromptTokensDetails.CachedTokens,
	}
}
