package llmprotocol

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func toolTextRequest(tools ...Tool) Request {
	request := validSemanticRequest()
	request.Tools = append([]Tool{{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)}}, tools...)
	return request
}

func functionTool(name, description string) Tool {
	return Tool{Name: name, Description: description, InputSchema: json.RawMessage(`{"type":"object"}`)}
}

func requireToolTextLimit(t *testing.T, err error, message, secret string) {
	t.Helper()
	requireLLMProtocolErrorCode(t, err, "tool_text_limit")
	var protocolError *ProtocolError
	errors.As(err, &protocolError)
	if protocolError.Message != message {
		t.Fatalf("message = %q, want %q", protocolError.Message, message)
	}
	if secret != "" && strings.Contains(err.Error(), secret) {
		t.Fatal("tool text limit error echoed the tool text")
	}
}

func TestToolTextLimitsDefaultBoundaries(t *testing.T) {
	limits := DefaultPolicy().Limits
	atLimit := strings.Repeat("d", limits.ToolDescriptionBytes)
	if err := ValidateRequest(toolTextRequest(functionTool("artifact", atLimit)), limits); err != nil {
		t.Fatalf("description at the default limit was rejected: %v", err)
	}
	over := "secret-" + strings.Repeat("d", limits.ToolDescriptionBytes-6)
	requireToolTextLimit(t, ValidateRequest(toolTextRequest(functionTool("artifact", over)), limits),
		"tools[1].description is 16385 bytes and exceeds the configured limit of 16384 bytes", "secret-")

	if err := ValidateRequest(toolTextRequest(functionTool(strings.Repeat("n", 256), "")), limits); err != nil {
		t.Fatalf("name at the default limit was rejected: %v", err)
	}
	requireToolTextLimit(t, ValidateRequest(toolTextRequest(functionTool(strings.Repeat("n", 257), "")), limits),
		"tools[1].name is 257 bytes and exceeds the configured limit of 256 bytes", "")
}

func TestToolTextLimitsCountUTF8Bytes(t *testing.T) {
	limits := DefaultPolicy().Limits
	// Limits count UTF-8 bytes, not runes.
	fits := strings.Repeat("é", limits.ToolDescriptionBytes/2)
	if err := ValidateRequest(toolTextRequest(functionTool("artifact", fits)), limits); err != nil {
		t.Fatalf("multibyte description at the byte limit was rejected: %v", err)
	}
	requireToolTextLimit(t, ValidateRequest(toolTextRequest(functionTool("artifact", fits+"é")), limits),
		"tools[1].description is 16386 bytes and exceeds the configured limit of 16384 bytes", "")

	// A four-byte rune that straddles the limit counts whole.
	straddle := strings.Repeat("d", limits.ToolDescriptionBytes-1) + "🧰"
	requireToolTextLimit(t, ValidateRequest(toolTextRequest(functionTool("artifact", straddle)), limits),
		"tools[1].description is 16387 bytes and exceeds the configured limit of 16384 bytes", "")

	name := strings.Repeat("工", 86) // 258 bytes, 86 runes
	requireToolTextLimit(t, ValidateRequest(toolTextRequest(functionTool(name, "")), limits),
		"tools[1].name is 258 bytes and exceeds the configured limit of 256 bytes", "")
}

func TestToolTextLimitsFollowConfiguredPolicy(t *testing.T) {
	limits := DefaultPolicy().Limits
	limits.ToolNameBytes = 512
	limits.ToolDescriptionBytes = 64 << 10

	// Claude Code's Artifact tool description is about 20 KiB.
	artifact := functionTool(strings.Repeat("n", 300), strings.Repeat("d", 20_225))
	if err := ValidateRequest(toolTextRequest(artifact), limits); err != nil {
		t.Fatalf("raised limits rejected a tool within them: %v", err)
	}
	requireToolTextLimit(t, ValidateRequest(toolTextRequest(functionTool("artifact", strings.Repeat("d", 64<<10+1))), limits),
		"tools[1].description is 65537 bytes and exceeds the configured limit of 65536 bytes", "")
	requireToolTextLimit(t, ValidateRequest(toolTextRequest(functionTool(strings.Repeat("n", 513), "")), limits),
		"tools[1].name is 513 bytes and exceeds the configured limit of 512 bytes", "")

	custom := Tool{Kind: ToolKindCustom, Name: "apply_patch", Description: strings.Repeat("d", 64<<10+1)}
	requireToolTextLimit(t, ValidateRequest(toolTextRequest(custom), limits),
		"tools[1].description is 65537 bytes and exceeds the configured limit of 65536 bytes", "")
}
