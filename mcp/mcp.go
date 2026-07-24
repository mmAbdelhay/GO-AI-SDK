package mcpai

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"

	"github.com/mmabdelhay/go-ai-sdk/schema"
	"github.com/mmabdelhay/go-ai-sdk/tool"

	mcp "github.com/modelcontextprotocol/go-sdk/mcp"
)

// clientName/clientVersion identify this SDK to MCP servers during handshake.
const (
	clientName    = "go-ai-sdk"
	clientVersion = "0.1.0"
)

// Client is a connected MCP session whose tools can be imported into an agent.
// Construct it with Connect, ConnectCommand, or ConnectHTTP; close it with
// Close.
type Client struct {
	session *mcp.ClientSession
}

// Connect establishes an MCP session over the given transport. Most callers use
// the ConnectCommand or ConnectHTTP helpers instead.
func Connect(ctx context.Context, transport mcp.Transport) (*Client, error) {
	c := mcp.NewClient(&mcp.Implementation{Name: clientName, Version: clientVersion}, nil)
	session, err := c.Connect(ctx, transport, nil)
	if err != nil {
		return nil, fmt.Errorf("mcp: connect: %w", err)
	}
	return &Client{session: session}, nil
}

// ConnectCommand starts an MCP server subprocess and connects to it over its
// stdio, the most common MCP transport. The command is managed by the returned
// client and terminated on Close.
func ConnectCommand(ctx context.Context, cmd *exec.Cmd) (*Client, error) {
	return Connect(ctx, &mcp.CommandTransport{Command: cmd})
}

// ConnectHTTP connects to a streamable-HTTP MCP server at url.
func ConnectHTTP(ctx context.Context, url string) (*Client, error) {
	return Connect(ctx, &mcp.StreamableClientTransport{Endpoint: url})
}

// Close ends the MCP session (and terminates a subprocess transport).
func (c *Client) Close() error { return c.session.Close() }

// Tools lists the server's tools and adapts each into a tool.Tool, ready to pass
// to agent.WithTools. The returned tools call back into this client's session,
// so keep the client open for as long as the agent may invoke them.
func (c *Client) Tools(ctx context.Context) ([]tool.Tool, error) {
	var tools []tool.Tool
	for mcpTool, err := range c.session.Tools(ctx, nil) {
		if err != nil {
			return nil, fmt.Errorf("mcp: listing tools: %w", err)
		}
		tools = append(tools, newRemoteTool(c.session, mcpTool))
	}
	return tools, nil
}

// remoteTool adapts a single MCP server tool to the tool.Tool interface. It
// implements tool.RawSchemaTool so the server's original JSON Schema is passed
// to the model verbatim, without lossy re-derivation.
type remoteTool struct {
	session     *mcp.ClientSession
	name        string
	description string
	rawSchema   json.RawMessage
	schema      *schema.Schema
}

func newRemoteTool(session *mcp.ClientSession, t *mcp.Tool) *remoteTool {
	rt := &remoteTool{
		session:     session,
		name:        t.Name,
		description: t.Description,
		schema:      &schema.Schema{Type: "object"},
	}
	// t.InputSchema is a map[string]any on the client side; marshal it back to
	// JSON for verbatim forwarding, and best-effort decode into our schema type
	// for InputSchema().
	if raw, err := json.Marshal(t.InputSchema); err == nil && len(raw) > 0 && string(raw) != "null" {
		rt.rawSchema = raw
		var s schema.Schema
		if json.Unmarshal(raw, &s) == nil {
			rt.schema = &s
		}
	}
	return rt
}

func (t *remoteTool) Name() string                    { return t.name }
func (t *remoteTool) Description() string             { return t.description }
func (t *remoteTool) InputSchema() *schema.Schema     { return t.schema }
func (t *remoteTool) RawInputSchema() json.RawMessage { return t.rawSchema }

// Call invokes the tool on the MCP server. Input validation is delegated to the
// server (which validates against its own schema); the text content of the
// result is concatenated and returned. A tool-level error (IsError) becomes a Go
// error, which the agent's registry converts into an error tool-result the model
// can see and react to.
func (t *remoteTool) Call(ctx context.Context, input json.RawMessage) (string, error) {
	var args any
	if len(input) > 0 {
		args = input // json.RawMessage marshals as its raw bytes
	}
	res, err := t.session.CallTool(ctx, &mcp.CallToolParams{Name: t.name, Arguments: args})
	if err != nil {
		return "", fmt.Errorf("mcp: calling tool %q: %w", t.name, err)
	}
	text := textOf(res)
	if res.IsError {
		return "", fmt.Errorf("mcp tool %q reported an error: %s", t.name, text)
	}
	return text, nil
}

func textOf(res *mcp.CallToolResult) string {
	var b strings.Builder
	for _, content := range res.Content {
		if tc, ok := content.(*mcp.TextContent); ok {
			b.WriteString(tc.Text)
		}
	}
	return b.String()
}

var (
	_ tool.Tool          = (*remoteTool)(nil)
	_ tool.RawSchemaTool = (*remoteTool)(nil)
)
