package mcp

import (
	"reflect"
	"testing"
)

// Keep concrete transports and the logging decorator aligned with the public,
// composed client contract at compile time.
var (
	_ MCPClient = (*StdioClient)(nil)
	_ MCPClient = (*HTTPClient)(nil)
	_ MCPClient = (*LoggingClientWrapper)(nil)
)

func TestMCPClientPreservesPublicMethodSet(t *testing.T) {
	want := []string{
		"CallTool",
		"Close",
		"Connect",
		"GetPrompt",
		"GetPrompts",
		"GetResources",
		"GetTools",
		"IsConnected",
		"Ping",
		"ReadResource",
		"RefreshCapabilities",
		"SetLogHandler",
	}
	clientType := reflect.TypeOf((*MCPClient)(nil)).Elem()

	if clientType.NumMethod() != len(want) {
		t.Fatalf("MCPClient method count = %d, want %d", clientType.NumMethod(), len(want))
	}
	for index, name := range want {
		if got := clientType.Method(index).Name; got != name {
			t.Fatalf("MCPClient method %d = %q, want %q", index, got, name)
		}
	}
}
