package mcp

import (
	"strings"
	"testing"
)

func TestMCPClientEventLogMessageOmitsUntrustedContent(t *testing.T) {
	const secretCanary = "mcp-router-log-secret-canary" //nolint:gosec // Canary verifies router logs omit remote content.
	got := mcpClientEventLogMessage(LoggingLevelError, "remote echoed "+secretCanary)

	if strings.Contains(got, secretCanary) {
		t.Fatalf("MCP event log leaked untrusted content: %s", got)
	}
	if !strings.Contains(got, string(LoggingLevelError)) {
		t.Fatalf("MCP event log omitted safe severity: %s", got)
	}
}
