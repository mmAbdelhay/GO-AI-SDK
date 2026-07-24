// Module mcp lets go-ai-sdk agents consume tools from Model Context Protocol
// servers. Separate submodule (design principle P2): only users who need MCP
// pull the protocol SDK. Built on the official MCP Go SDK.
module github.com/mmabdelhay/go-ai-sdk/mcp

go 1.23.0

toolchain go1.24.7

// Pinned to v1.2.0: it is the newest MCP SDK release still targeting Go 1.23
// (v1.4.1+ require Go 1.25). This keeps the whole workspace on one Go version.
require (
	github.com/mmabdelhay/go-ai-sdk v0.0.0
	github.com/modelcontextprotocol/go-sdk v1.2.0
)

require (
	github.com/google/jsonschema-go v0.3.0 // indirect
	github.com/yosida95/uritemplate/v3 v3.0.2 // indirect
	golang.org/x/oauth2 v0.30.0 // indirect
)

replace github.com/mmabdelhay/go-ai-sdk => ../
