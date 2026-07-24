package anthropic

import "encoding/json"

// This file defines the Anthropic Messages API wire types. They are unexported
// because callers work exclusively with the provider-neutral types in the ai
// package; these exist only at the translation boundary.

type wireRequest struct {
	Model         string          `json:"model"`
	MaxTokens     int             `json:"max_tokens"`
	Messages      []wireMessage   `json:"messages"`
	System        string          `json:"system,omitempty"`
	Temperature   *float64        `json:"temperature,omitempty"`
	TopP          *float64        `json:"top_p,omitempty"`
	TopK          *int            `json:"top_k,omitempty"`
	StopSequences []string        `json:"stop_sequences,omitempty"`
	Stream        bool            `json:"stream,omitempty"`
	Tools         []wireTool      `json:"tools,omitempty"`
	ToolChoice    *wireToolChoice `json:"tool_choice,omitempty"`
}

type wireTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type wireToolChoice struct {
	Type string `json:"type"` // "auto", "any", "tool", or "none"
	Name string `json:"name,omitempty"`
}

type wireMessage struct {
	Role    string      `json:"role"`
	Content []wireBlock `json:"content"`
}

// wireBlock is the union of Anthropic content block types. Only the fields
// relevant to Type are populated. It marshals and unmarshals every block kind
// this package produces or consumes: text, image, tool_use, tool_result.
type wireBlock struct {
	Type string `json:"type"`

	// text
	Text string `json:"text,omitempty"`

	// image
	Source *wireImageSource `json:"source,omitempty"`

	// tool_use
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// document
	Title string `json:"title,omitempty"`

	// tool_result
	ToolUseID string      `json:"tool_use_id,omitempty"`
	Content   []wireBlock `json:"content,omitempty"`
	IsError   bool        `json:"is_error,omitempty"`
}

type wireImageSource struct {
	Type      string `json:"type"` // "base64" or "url"
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type wireResponse struct {
	ID         string      `json:"id"`
	Type       string      `json:"type"`
	Role       string      `json:"role"`
	Model      string      `json:"model"`
	Content    []wireBlock `json:"content"`
	StopReason string      `json:"stop_reason"`
	Usage      wireUsage   `json:"usage"`
}

type wireUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

// wireErrorResponse is the shape of an Anthropic error body.
type wireErrorResponse struct {
	Type  string `json:"type"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}
