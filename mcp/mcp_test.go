package mcpai_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/agent"
	"github.com/mmabdelhay/go-ai-sdk/aitest"
	mcpai "github.com/mmabdelhay/go-ai-sdk/mcp"
	"github.com/mmabdelhay/go-ai-sdk/tool"

	mcp "github.com/modelcontextprotocol/go-sdk/mcp"
)

// addInput is the typed input for the test MCP server's "add" tool.
type addInput struct {
	A int `json:"a" jsonschema:"first addend"`
	B int `json:"b" jsonschema:"second addend"`
}

// startServer spins up an in-process MCP server exposing "add" and "boom"
// tools, connected to the client over an in-memory transport pair — no
// subprocess, no network. It returns a connected mcpai.Client.
func startServer(t *testing.T) *mcpai.Client {
	t.Helper()
	ctx := context.Background()

	server := mcp.NewServer(&mcp.Implementation{Name: "test-server", Version: "0.0.1"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "add", Description: "Adds two integers"},
		func(_ context.Context, _ *mcp.CallToolRequest, in addInput) (*mcp.CallToolResult, any, error) {
			return &mcp.CallToolResult{
				Content: []mcp.Content{&mcp.TextContent{Text: itoa(in.A + in.B)}},
			}, nil, nil
		})
	mcp.AddTool(server, &mcp.Tool{Name: "boom", Description: "Always fails"},
		func(_ context.Context, _ *mcp.CallToolRequest, _ struct{}) (*mcp.CallToolResult, any, error) {
			return &mcp.CallToolResult{
				IsError: true,
				Content: []mcp.Content{&mcp.TextContent{Text: "kaboom"}},
			}, nil, nil
		})

	clientT, serverT := mcp.NewInMemoryTransports()
	if _, err := server.Connect(ctx, serverT, nil); err != nil {
		t.Fatal(err)
	}
	client, err := mcpai.Connect(ctx, clientT)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return client
}

func itoa(n int) string {
	return strings.TrimSpace(string(mustJSON(n)))
}
func mustJSON(v any) []byte { b, _ := json.Marshal(v); return b }

func TestListAndCallTools(t *testing.T) {
	ctx := context.Background()
	client := startServer(t)

	tools, err := client.Tools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	byName := map[string]tool.Tool{}
	for _, tl := range tools {
		byName[tl.Name()] = tl
	}
	if _, ok := byName["add"]; !ok {
		t.Fatalf("add tool not listed; got %v", byName)
	}

	// The server's JSON schema is forwarded verbatim via RawSchemaTool.
	def := tool.Def(byName["add"])
	if !strings.Contains(string(def.InputSchema), `"a"`) || !strings.Contains(string(def.InputSchema), `"b"`) {
		t.Errorf("input schema not forwarded: %s", def.InputSchema)
	}

	// Calling the tool executes it on the server.
	out, err := byName["add"].Call(ctx, json.RawMessage(`{"a":2,"b":3}`))
	if err != nil {
		t.Fatal(err)
	}
	if out != "5" {
		t.Errorf("add result = %q, want 5", out)
	}
}

func TestToolErrorSurfaces(t *testing.T) {
	ctx := context.Background()
	client := startServer(t)
	tools, _ := client.Tools(ctx)
	var boom tool.Tool
	for _, tl := range tools {
		if tl.Name() == "boom" {
			boom = tl
		}
	}
	if boom == nil {
		t.Fatal("boom tool missing")
	}
	_, err := boom.Call(ctx, json.RawMessage(`{}`))
	if err == nil || !strings.Contains(err.Error(), "kaboom") {
		t.Errorf("expected surfaced tool error, got %v", err)
	}
}

// TestAgentUsesMCPTool proves the end-to-end path: an agent (driven by a
// scripted fake model) calls a tool that actually lives on an MCP server.
func TestAgentUsesMCPTool(t *testing.T) {
	ctx := context.Background()
	client := startServer(t)
	tools, err := client.Tools(ctx)
	if err != nil {
		t.Fatal(err)
	}

	model := &aitest.Model{GenerateFunc: func(_ context.Context, req ai.Request) (ai.Response, error) {
		// After the tool result comes back, answer with it.
		for _, m := range req.Messages {
			for _, p := range m.Content {
				if tr, ok := p.(ai.ToolResult); ok {
					return aitest.TextResponse("The sum is " + tr.Content[0].(ai.Text).Text), nil
				}
			}
		}
		// Otherwise, ask to call the MCP "add" tool.
		return ai.Response{
			Message: ai.Message{Role: ai.RoleAssistant, Content: []ai.Part{
				ai.ToolCall{ID: "c1", Name: "add", Input: json.RawMessage(`{"a":20,"b":22}`)},
			}},
			StopReason: ai.StopToolUse,
		}, nil
	}}

	a := agent.New(model, agent.WithTools(tools...), agent.WithMaxIterations(4))
	res, err := a.Run(ctx, "What is 20 + 22?")
	if err != nil {
		t.Fatal(err)
	}
	if res.Output != "The sum is 42" {
		t.Errorf("agent output = %q, want 'The sum is 42'", res.Output)
	}
	if res.ToolCalls != 1 {
		t.Errorf("tool calls = %d, want 1", res.ToolCalls)
	}
}
