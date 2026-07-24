// Package mcpai lets go-ai-sdk agents consume tools exposed by Model Context
// Protocol (MCP) servers. It wraps the official MCP Go SDK
// (github.com/modelcontextprotocol/go-sdk) and adapts each remote MCP tool into
// a tool.Tool, so agents use MCP tools exactly like native ones:
//
//	client, err := mcpai.ConnectCommand(ctx, exec.Command("my-mcp-server"))
//	tools, err := client.Tools(ctx)
//	a := agent.New(model, agent.WithTools(tools...))
package mcpai
