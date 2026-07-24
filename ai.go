package ai

import (
	"context"
	"encoding/json"
	"iter"
)

// Role identifies who authored a Message.
type Role string

const (
	// RoleSystem is a system instruction that steers the model's behavior.
	RoleSystem Role = "system"
	// RoleUser is input from the end user.
	RoleUser Role = "user"
	// RoleAssistant is output produced by the model.
	RoleAssistant Role = "assistant"
	// RoleTool carries the result of a tool invocation back to the model.
	RoleTool Role = "tool"
)

// Part is a single piece of message content. Its concrete implementations are
// [Text], [Image], [ToolCall] and [ToolResult]. Part is a closed interface: the
// unexported isPart method prevents third-party implementations so that every
// provider translation layer can exhaustively switch over the known part types.
type Part interface {
	isPart()
}

// Text is a plain-text content part.
type Text struct {
	Text string
}

// Image is an image content part. Exactly one of Data or URL should be set. When
// Data is used, MediaType (for example "image/png") must describe the bytes.
type Image struct {
	// Data holds the raw (already decoded) image bytes. Providers that require
	// base64 will encode it at the wire boundary.
	Data []byte
	// MediaType is the IANA media type of Data, e.g. "image/jpeg". Ignored when
	// URL is set.
	MediaType string
	// URL references a remotely hosted image. Mutually exclusive with Data.
	URL string
}

// ToolCall is a request from the model to invoke a tool. It appears in assistant
// messages. Phase 1 models and translates tool calls so the canonical form is
// round-trip complete; the tool registry and agent loop arrive in a later phase.
type ToolCall struct {
	// ID uniquely identifies this call so its result can be correlated.
	ID string
	// Name is the tool the model wants to invoke.
	Name string
	// Input is the raw JSON arguments produced by the model.
	Input json.RawMessage
}

// ToolResult carries the outcome of a [ToolCall] back to the model. It appears in
// user (or tool) messages.
type ToolResult struct {
	// CallID matches the [ToolCall.ID] this result answers.
	CallID string
	// Content is the result payload, usually a single [Text] part.
	Content []Part
	// IsError reports whether the tool failed, so the model can react.
	IsError bool
}

func (Text) isPart()       {}
func (Image) isPart()      {}
func (ToolCall) isPart()   {}
func (ToolResult) isPart() {}

// Message is a single turn in a conversation: a role plus ordered content parts.
// The canonical Message form is provider-neutral; each provider package
// translates it to and from that provider's wire format without loss.
type Message struct {
	Role    Role
	Content []Part
}

// System returns a system Message containing s.
func System(s string) Message {
	return Message{Role: RoleSystem, Content: []Part{Text{Text: s}}}
}

// UserText returns a user Message containing s.
func UserText(s string) Message {
	return Message{Role: RoleUser, Content: []Part{Text{Text: s}}}
}

// AssistantText returns an assistant Message containing s.
func AssistantText(s string) Message {
	return Message{Role: RoleAssistant, Content: []Part{Text{Text: s}}}
}

// text concatenates all Text parts in the message, ignoring other part types.
func (m Message) text() string {
	var b []byte
	for _, p := range m.Content {
		if t, ok := p.(Text); ok {
			b = append(b, t.Text...)
		}
	}
	return string(b)
}

// Usage reports token counts for a request/response pair. Cache fields are zero
// for providers that do not report prompt caching.
type Usage struct {
	InputTokens         int
	OutputTokens        int
	CacheCreationTokens int
	CacheReadTokens     int
}

// TotalTokens returns the sum of input and output tokens.
func (u Usage) TotalTokens() int { return u.InputTokens + u.OutputTokens }

// StopReason describes why the model stopped generating. Values are normalized
// across providers.
type StopReason string

const (
	// StopEndTurn indicates the model finished its turn naturally.
	StopEndTurn StopReason = "end_turn"
	// StopMaxTokens indicates generation hit the token limit.
	StopMaxTokens StopReason = "max_tokens"
	// StopStopSequence indicates a configured stop sequence was produced.
	StopStopSequence StopReason = "stop_sequence"
	// StopToolUse indicates the model wants to call one or more tools.
	StopToolUse StopReason = "tool_use"
)

// Request is a provider-neutral generation request. The [ChatModel] interface
// methods operate on a fully-built Request; the top-level [Generate] and [Stream]
// helpers construct one from functional [Option]s.
type Request struct {
	// Model is the provider model identifier. When empty, providers fall back to
	// the default configured on the client.
	Model string
	// System is a convenience system prompt. It is combined with any
	// RoleSystem messages in Messages.
	System string
	// Messages is the ordered conversation history.
	Messages []Message
	// Temperature, when non-nil, overrides the provider default sampling
	// temperature.
	Temperature *float64
	// TopP, when non-nil, overrides nucleus sampling.
	TopP *float64
	// MaxTokens caps output tokens. Zero means "use the client default".
	MaxTokens int
	// StopSequences halt generation when produced.
	StopSequences []string
	// ProviderOptions carries provider-specific, typed knobs. Each provider
	// inspects only the options it recognizes and ignores the rest.
	ProviderOptions []ProviderOption
}

// Response is a provider-neutral generation result.
type Response struct {
	// Model is the model that produced the response.
	Model string
	// Message is the assistant message, with all content parts.
	Message Message
	// StopReason is the normalized reason generation stopped.
	StopReason StopReason
	// Usage reports token consumption.
	Usage Usage
	// Raw is the provider's raw response body, for debugging and escape hatches.
	Raw json.RawMessage
}

// Text returns the concatenated text parts of the response message.
func (r Response) Text() string { return r.Message.text() }

// ChunkType discriminates the kind of streaming [Chunk].
type ChunkType int

const (
	// ChunkText carries an incremental text delta in Chunk.Text.
	ChunkText ChunkType = iota
	// ChunkToolCallDelta carries an incremental tool-call update. Reserved for
	// the tools phase; text streaming does not emit it.
	ChunkToolCallDelta
	// ChunkDone is the terminal chunk; Chunk.Usage and Chunk.StopReason are set.
	ChunkDone
)

// Chunk is a single event in a [Stream].
type Chunk struct {
	Type ChunkType
	// Text is the incremental text delta for ChunkText chunks.
	Text string
	// Usage is set on the terminal ChunkDone chunk.
	Usage *Usage
	// StopReason is set on the terminal ChunkDone chunk.
	StopReason StopReason
}

// Stream is a pull-free stream of generation events. Its underlying type is
// identical to iter.Seq2[Chunk, error], so it can be ranged over directly:
//
//	for chunk, err := range model.Stream(ctx, req) {
//		if err != nil { return err }
//		fmt.Print(chunk.Text)
//	}
//
// Setup failures (request validation, connection errors before the first event)
// are delivered as the first (Chunk, error) pair rather than a separate return
// value, so a single error path covers the whole stream.
type Stream iter.Seq2[Chunk, error]

// Text drains the stream and returns the concatenation of all text deltas. It
// stops and returns at the first error encountered.
func (s Stream) Text() (string, error) {
	var b []byte
	for chunk, err := range s {
		if err != nil {
			return string(b), err
		}
		b = append(b, chunk.Text...)
	}
	return string(b), nil
}

// Collect drains the stream into a single [Response], accumulating text and
// capturing the terminal usage and stop reason. It returns at the first error.
func (s Stream) Collect() (Response, error) {
	var (
		text []byte
		resp Response
	)
	for chunk, err := range s {
		if err != nil {
			resp.Message = Message{Role: RoleAssistant, Content: []Part{Text{Text: string(text)}}}
			return resp, err
		}
		switch chunk.Type {
		case ChunkText:
			text = append(text, chunk.Text...)
		case ChunkDone:
			if chunk.Usage != nil {
				resp.Usage = *chunk.Usage
			}
			resp.StopReason = chunk.StopReason
		}
	}
	resp.Message = Message{Role: RoleAssistant, Content: []Part{Text{Text: string(text)}}}
	return resp, nil
}

// ChatModel is the central contract implemented by every provider. It is
// deliberately small so it can be implemented by hand in a test (see the aitest
// package).
type ChatModel interface {
	// Name returns a stable identifier for the underlying model.
	Name() string
	// Generate produces a complete response for req.
	Generate(ctx context.Context, req Request) (Response, error)
	// Stream produces an incremental response for req. Cancel via ctx.
	Stream(ctx context.Context, req Request) Stream
}
