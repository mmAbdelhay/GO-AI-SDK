package openai

import "encoding/json"

// Wire types for the OpenAI Chat Completions and Embeddings APIs. Unexported:
// they exist only at the translation boundary.

type oaRequest struct {
	Model               string          `json:"model"`
	Messages            []oaMessage     `json:"messages"`
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	MaxTokens           *int            `json:"max_tokens,omitempty"` // legacy field, see WithLegacyMaxTokens
	Temperature         *float64        `json:"temperature,omitempty"`
	TopP                *float64        `json:"top_p,omitempty"`
	Stop                []string        `json:"stop,omitempty"`
	Stream              bool            `json:"stream,omitempty"`
	StreamOptions       *oaStreamOpts   `json:"stream_options,omitempty"`
	Tools               []oaTool        `json:"tools,omitempty"`
	ToolChoice          json.RawMessage `json:"tool_choice,omitempty"`
	Seed                *int            `json:"seed,omitempty"`
}

type oaStreamOpts struct {
	IncludeUsage bool `json:"include_usage"`
}

type oaTool struct {
	Type     string     `json:"type"` // always "function"
	Function oaFunction `json:"function"`
}

type oaFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

// oaMessage is a chat message. Content is either a plain string or an array of
// content parts, so it is typed as any and built by the translator.
type oaMessage struct {
	Role       string       `json:"role"`
	Content    any          `json:"content,omitempty"`
	ToolCalls  []oaToolCall `json:"tool_calls,omitempty"`
	ToolCallID string       `json:"tool_call_id,omitempty"`
}

type oaContentPart struct {
	Type       string        `json:"type"` // "text", "image_url", "input_audio", "file"
	Text       string        `json:"text,omitempty"`
	ImageURL   *oaImageURL   `json:"image_url,omitempty"`
	InputAudio *oaInputAudio `json:"input_audio,omitempty"`
	File       *oaFile       `json:"file,omitempty"`
}

type oaImageURL struct {
	URL string `json:"url"`
}

type oaInputAudio struct {
	Data   string `json:"data"`   // base64
	Format string `json:"format"` // "wav", "mp3", ...
}

type oaFile struct {
	Filename string `json:"filename,omitempty"`
	FileData string `json:"file_data,omitempty"` // data: URI
	FileID   string `json:"file_id,omitempty"`
}

type oaToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type oaResponse struct {
	Model   string `json:"model"`
	Choices []struct {
		Message struct {
			Role      string       `json:"role"`
			Content   string       `json:"content"`
			ToolCalls []oaToolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage oaUsage `json:"usage"`
}

type oaUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

// oaStreamChunk is one SSE data payload during streaming.
type oaStreamChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *oaUsage `json:"usage"`
}

type oaErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    any    `json:"code"` // string or number depending on provider
	} `json:"error"`
}

type oaEmbedRequest struct {
	Model          string   `json:"model"`
	Input          []string `json:"input"`
	EncodingFormat string   `json:"encoding_format,omitempty"`
}

type oaEmbedResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
	} `json:"usage"`
}
