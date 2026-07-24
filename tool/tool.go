// Package tool defines the Tool interface used by agents, a type-safe
// constructor that derives a tool's input schema from a Go function signature,
// and a Registry for lookup and execution.
package tool

import (
	"context"
	"encoding/json"
	"fmt"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/schema"
)

// Tool is a callable capability exposed to a model. Implementations must be safe
// for concurrent use.
type Tool interface {
	// Name is the identifier the model uses to call the tool.
	Name() string
	// Description tells the model what the tool does and when to use it.
	Description() string
	// InputSchema describes the tool's JSON input.
	InputSchema() *schema.Schema
	// Call executes the tool with the model-provided JSON input.
	Call(ctx context.Context, input json.RawMessage) (string, error)
}

// funcTool adapts a typed Go function to Tool, deriving the input schema from
// the function's argument type.
type funcTool[I any] struct {
	name        string
	description string
	schema      *schema.Schema
	fn          func(context.Context, I) (string, error)
}

// New builds a Tool from a typed handler function. The input schema is derived
// from I via the schema package, and the model's JSON input is validated against
// it before being unmarshaled and passed to fn:
//
//	type WeatherInput struct {
//		City string `json:"city" desc:"City name"`
//	}
//	weather, err := tool.New("get_weather", "Current weather for a city",
//		func(ctx context.Context, in WeatherInput) (string, error) { ... })
func New[I any](name, description string, fn func(context.Context, I) (string, error)) (Tool, error) {
	if name == "" {
		return nil, fmt.Errorf("tool: name is required")
	}
	s, err := schema.For[I]()
	if err != nil {
		return nil, fmt.Errorf("tool %s: %w", name, err)
	}
	return &funcTool[I]{name: name, description: description, schema: s, fn: fn}, nil
}

// MustNew is like [New] but panics on error. Intended for package-level tool
// variables whose input types are known valid.
func MustNew[I any](name, description string, fn func(context.Context, I) (string, error)) Tool {
	t, err := New(name, description, fn)
	if err != nil {
		panic(err)
	}
	return t
}

func (t *funcTool[I]) Name() string                { return t.name }
func (t *funcTool[I]) Description() string         { return t.description }
func (t *funcTool[I]) InputSchema() *schema.Schema { return t.schema }

func (t *funcTool[I]) Call(ctx context.Context, input json.RawMessage) (string, error) {
	if len(input) == 0 {
		input = json.RawMessage(`{}`)
	}
	if err := schema.Validate(t.schema, input); err != nil {
		return "", fmt.Errorf("invalid input: %w", err)
	}
	var in I
	if err := json.Unmarshal(input, &in); err != nil {
		return "", fmt.Errorf("invalid input: %w", err)
	}
	return t.fn(ctx, in)
}

// RawSchemaTool is an optional interface a [Tool] may implement to supply its
// input schema as raw JSON rather than have it re-serialized from a
// [schema.Schema]. [Def] prefers this when present, preserving full schema
// fidelity for tools whose schema originates outside this library (for example
// tools imported from an MCP server).
type RawSchemaTool interface {
	Tool
	RawInputSchema() json.RawMessage
}

// Def converts a Tool to the provider-neutral [ai.ToolDef].
func Def(t Tool) ai.ToolDef {
	schemaJSON := t.InputSchema().JSON()
	if rst, ok := t.(RawSchemaTool); ok {
		if raw := rst.RawInputSchema(); len(raw) > 0 {
			schemaJSON = raw
		}
	}
	return ai.ToolDef{
		Name:        t.Name(),
		Description: t.Description(),
		InputSchema: schemaJSON,
	}
}

// Registry is an immutable, name-indexed collection of tools.
type Registry struct {
	tools map[string]Tool
	order []string
}

// NewRegistry builds a Registry, rejecting duplicate tool names.
func NewRegistry(tools ...Tool) (*Registry, error) {
	r := &Registry{tools: make(map[string]Tool, len(tools))}
	for _, t := range tools {
		if _, dup := r.tools[t.Name()]; dup {
			return nil, fmt.Errorf("tool: duplicate name %q", t.Name())
		}
		r.tools[t.Name()] = t
		r.order = append(r.order, t.Name())
	}
	return r, nil
}

// Get returns the named tool.
func (r *Registry) Get(name string) (Tool, bool) {
	t, ok := r.tools[name]
	return t, ok
}

// Defs returns provider-neutral declarations for every tool, in registration
// order, ready to place on an ai.Request.
func (r *Registry) Defs() []ai.ToolDef {
	defs := make([]ai.ToolDef, 0, len(r.order))
	for _, name := range r.order {
		defs = append(defs, Def(r.tools[name]))
	}
	return defs
}

// Execute runs the tool named by call and packages the outcome as an
// [ai.ToolResult] suitable for sending back to the model. Tool failures and
// unknown tools become IsError results (visible to the model) rather than Go
// errors; the error return is reserved for context cancellation.
func (r *Registry) Execute(ctx context.Context, call ai.ToolCall) (ai.ToolResult, error) {
	if err := ctx.Err(); err != nil {
		return ai.ToolResult{}, err
	}
	t, ok := r.tools[call.Name]
	if !ok {
		return errResult(call, fmt.Sprintf("unknown tool %q", call.Name)), nil
	}
	out, err := t.Call(ctx, call.Input)
	if err != nil {
		if ctx.Err() != nil {
			return ai.ToolResult{}, ctx.Err()
		}
		return errResult(call, err.Error()), nil
	}
	return ai.ToolResult{CallID: call.ID, Content: []ai.Part{ai.Text{Text: out}}}, nil
}

func errResult(call ai.ToolCall, msg string) ai.ToolResult {
	return ai.ToolResult{
		CallID:  call.ID,
		Content: []ai.Part{ai.Text{Text: msg}},
		IsError: true,
	}
}
