package gemini

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

func toWire(req ai.Request) (gRequest, error) {
	w := gRequest{}

	system := req.System
	for _, m := range req.Messages {
		if m.Role != ai.RoleSystem {
			continue
		}
		for _, p := range m.Content {
			if t, ok := p.(ai.Text); ok {
				if system != "" {
					system += "\n\n"
				}
				system += t.Text
			}
		}
	}
	if system != "" {
		w.SystemInstruction = &gContent{Parts: []gPart{{Text: system}}}
	}

	for _, m := range req.Messages {
		if m.Role == ai.RoleSystem {
			continue
		}
		content, err := toWireContent(m)
		if err != nil {
			return gRequest{}, err
		}
		w.Contents = append(w.Contents, content)
	}

	if req.Temperature != nil || req.TopP != nil || req.MaxTokens > 0 || len(req.StopSequences) > 0 {
		w.GenerationConfig = &gGenConfig{
			Temperature:     req.Temperature,
			TopP:            req.TopP,
			MaxOutputTokens: req.MaxTokens,
			StopSequences:   req.StopSequences,
		}
	}

	if len(req.Tools) > 0 {
		decls := make([]gFunctionDecl, 0, len(req.Tools))
		for _, t := range req.Tools {
			decls = append(decls, gFunctionDecl{
				Name: t.Name, Description: t.Description, Parameters: stripAdditionalProps(t.InputSchema),
			})
		}
		w.Tools = []gTools{{FunctionDeclarations: decls}}
	}
	tc, err := toWireToolConfig(req.ToolChoice)
	if err != nil {
		return gRequest{}, err
	}
	w.ToolConfig = tc
	return w, nil
}

// stripAdditionalProps removes "additionalProperties" keys, which Gemini's
// function-declaration schema dialect rejects. Only the top level is touched;
// nested occurrences are tolerated by the API.
func stripAdditionalProps(s json.RawMessage) json.RawMessage {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(s, &m); err != nil {
		return s
	}
	delete(m, "additionalProperties")
	out, err := json.Marshal(m)
	if err != nil {
		return s
	}
	return out
}

func toWireToolConfig(tc ai.ToolChoice) (*gToolConfig, error) {
	switch tc.Mode {
	case "":
		return nil, nil
	case "auto":
		return &gToolConfig{gFunctionCallingConfig{Mode: "AUTO"}}, nil
	case "required":
		return &gToolConfig{gFunctionCallingConfig{Mode: "ANY"}}, nil
	case "none":
		return &gToolConfig{gFunctionCallingConfig{Mode: "NONE"}}, nil
	case "tool":
		return &gToolConfig{gFunctionCallingConfig{Mode: "ANY", AllowedFunctionNames: []string{tc.Name}}}, nil
	default:
		return nil, fmt.Errorf("gemini: unsupported tool choice mode %q", tc.Mode)
	}
}

func toWireContent(m ai.Message) (gContent, error) {
	role := "user"
	if m.Role == ai.RoleAssistant {
		role = "model"
	}
	content := gContent{Role: role}
	for _, p := range m.Content {
		switch v := p.(type) {
		case ai.Text:
			content.Parts = append(content.Parts, gPart{Text: v.Text})
		case ai.Image:
			if v.URL != "" {
				content.Parts = append(content.Parts, gPart{FileData: &gFileData{FileURI: v.URL}})
			} else {
				content.Parts = append(content.Parts, gPart{InlineData: &gInlineData{
					MimeType: v.MediaType,
					Data:     base64.StdEncoding.EncodeToString(v.Data),
				}})
			}
		case ai.ToolCall:
			content.Parts = append(content.Parts, gPart{FunctionCall: &gFunctionCall{
				Name: v.Name, Args: v.Input,
			}})
		case ai.ToolResult:
			text := ""
			for _, cp := range v.Content {
				if t, ok := cp.(ai.Text); ok {
					text += t.Text
				}
			}
			payload := map[string]any{"output": text}
			if v.IsError {
				payload = map[string]any{"error": text}
			}
			raw, err := json.Marshal(payload)
			if err != nil {
				return gContent{}, fmt.Errorf("gemini: marshaling tool result: %w", err)
			}
			// Gemini correlates results by function name; ToolCall.ID carries
			// the name (see the package doc for this compromise).
			content.Parts = append(content.Parts, gPart{FunctionResponse: &gFunctionResult{
				Name: v.CallID, Response: raw,
			}})
		default:
			return gContent{}, fmt.Errorf("gemini: unsupported content part %T", p)
		}
	}
	return content, nil
}

func fromWire(w gResponse, model string, raw []byte) (ai.Response, error) {
	if len(w.Candidates) == 0 {
		return ai.Response{}, fmt.Errorf("gemini: response has no candidates")
	}
	cand := w.Candidates[0]
	msg := ai.Message{Role: ai.RoleAssistant}
	for _, p := range cand.Content.Parts {
		switch {
		case p.FunctionCall != nil:
			msg.Content = append(msg.Content, ai.ToolCall{
				ID:    p.FunctionCall.Name, // Gemini has no call IDs; see package doc
				Name:  p.FunctionCall.Name,
				Input: p.FunctionCall.Args,
			})
		case p.Text != "":
			msg.Content = append(msg.Content, ai.Text{Text: p.Text})
		}
	}
	resp := ai.Response{
		Model:      model,
		Message:    msg,
		StopReason: mapFinishReason(cand.FinishReason, msg),
		Raw:        raw,
	}
	if w.UsageMetadata != nil {
		resp.Usage = ai.Usage{
			InputTokens:     w.UsageMetadata.PromptTokenCount,
			OutputTokens:    w.UsageMetadata.CandidatesTokenCount,
			CacheReadTokens: w.UsageMetadata.CachedContentTokenCount,
		}
	}
	return resp, nil
}

func mapFinishReason(s string, msg ai.Message) ai.StopReason {
	// Gemini reports STOP even for function calls; detect tool use from content.
	for _, p := range msg.Content {
		if _, ok := p.(ai.ToolCall); ok {
			return ai.StopToolUse
		}
	}
	switch s {
	case "STOP":
		return ai.StopEndTurn
	case "MAX_TOKENS":
		return ai.StopMaxTokens
	default:
		return ai.StopReason(s)
	}
}
