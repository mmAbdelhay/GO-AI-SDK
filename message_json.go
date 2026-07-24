package ai

import (
	"encoding/json"
	"fmt"
)

// Message JSON encoding. Because Content holds the closed [Part] interface,
// Message needs explicit (un)marshaling so it round-trips losslessly through
// JSON — used by conversation stores (see the store/ submodules) and anywhere a
// canonical message is persisted or transported. Each part is tagged with a
// "type" discriminator.

// partEnvelope is the wire form of a single content part. Only the fields
// relevant to Type are populated.
type partEnvelope struct {
	Type string `json:"type"`

	// text
	Text string `json:"text,omitempty"`

	// image / audio / document
	Data      []byte `json:"data,omitempty"` // base64 via encoding/json
	MediaType string `json:"media_type,omitempty"`
	URL       string `json:"url,omitempty"`
	Name      string `json:"name,omitempty"`

	// tool_call
	ID    string          `json:"id,omitempty"`
	Name_ string          `json:"tool_name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// tool_result
	CallID  string         `json:"call_id,omitempty"`
	Content []partEnvelope `json:"content,omitempty"`
	IsError bool           `json:"is_error,omitempty"`
}

type messageJSON struct {
	Role    Role           `json:"role"`
	Content []partEnvelope `json:"content"`
}

// MarshalJSON implements json.Marshaler.
func (m Message) MarshalJSON() ([]byte, error) {
	envs, err := partsToEnvelopes(m.Content)
	if err != nil {
		return nil, err
	}
	return json.Marshal(messageJSON{Role: m.Role, Content: envs})
}

// UnmarshalJSON implements json.Unmarshaler.
func (m *Message) UnmarshalJSON(data []byte) error {
	var mj messageJSON
	if err := json.Unmarshal(data, &mj); err != nil {
		return err
	}
	parts, err := envelopesToParts(mj.Content)
	if err != nil {
		return err
	}
	m.Role = mj.Role
	m.Content = parts
	return nil
}

func partsToEnvelopes(parts []Part) ([]partEnvelope, error) {
	if parts == nil {
		return nil, nil
	}
	out := make([]partEnvelope, 0, len(parts))
	for _, p := range parts {
		env, err := partToEnvelope(p)
		if err != nil {
			return nil, err
		}
		out = append(out, env)
	}
	return out, nil
}

func partToEnvelope(p Part) (partEnvelope, error) {
	switch v := p.(type) {
	case Text:
		return partEnvelope{Type: "text", Text: v.Text}, nil
	case Image:
		return partEnvelope{Type: "image", Data: v.Data, MediaType: v.MediaType, URL: v.URL}, nil
	case Audio:
		return partEnvelope{Type: "audio", Data: v.Data, MediaType: v.MediaType}, nil
	case Document:
		return partEnvelope{Type: "document", Data: v.Data, MediaType: v.MediaType, URL: v.URL, Name: v.Name}, nil
	case ToolCall:
		return partEnvelope{Type: "tool_call", ID: v.ID, Name_: v.Name, Input: v.Input}, nil
	case ToolResult:
		content, err := partsToEnvelopes(v.Content)
		if err != nil {
			return partEnvelope{}, err
		}
		return partEnvelope{Type: "tool_result", CallID: v.CallID, Content: content, IsError: v.IsError}, nil
	default:
		return partEnvelope{}, fmt.Errorf("ai: cannot marshal unknown content part %T", p)
	}
}

func envelopesToParts(envs []partEnvelope) ([]Part, error) {
	if envs == nil {
		return nil, nil
	}
	out := make([]Part, 0, len(envs))
	for _, e := range envs {
		p, err := envelopeToPart(e)
		if err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, nil
}

func envelopeToPart(e partEnvelope) (Part, error) {
	switch e.Type {
	case "text":
		return Text{Text: e.Text}, nil
	case "image":
		return Image{Data: e.Data, MediaType: e.MediaType, URL: e.URL}, nil
	case "audio":
		return Audio{Data: e.Data, MediaType: e.MediaType}, nil
	case "document":
		return Document{Data: e.Data, MediaType: e.MediaType, URL: e.URL, Name: e.Name}, nil
	case "tool_call":
		return ToolCall{ID: e.ID, Name: e.Name_, Input: e.Input}, nil
	case "tool_result":
		content, err := envelopesToParts(e.Content)
		if err != nil {
			return nil, err
		}
		return ToolResult{CallID: e.CallID, Content: content, IsError: e.IsError}, nil
	default:
		return nil, fmt.Errorf("ai: cannot unmarshal unknown content part type %q", e.Type)
	}
}
