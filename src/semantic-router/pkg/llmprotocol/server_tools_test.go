package llmprotocol

import (
	"encoding/json"
	"errors"
	"testing"
)

// Raw state stays where it belongs: a server tool block holds only its
// provider block and sits in an assistant turn, only text carries citations,
// and a server tool definition carries nothing beside its name.
func TestServerToolStateIsClosedByKind(t *testing.T) {
	block := Content{Kind: ContentServerTool, Raw: json.RawMessage(`{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}`)}
	request := func(role Role, content Content, tools ...Tool) Request {
		return Request{Generation: 1, Model: "m", Tools: tools, Messages: []Message{
			{Role: RoleUser, Content: []Content{{Kind: ContentText, Text: "search"}}},
			{Role: role, Content: []Content{content}},
		}}
	}
	server := Tool{Kind: ToolKindServer, Name: "web_search", Raw: json.RawMessage(`{"type":"web_search_20250305","name":"web_search"}`)}
	if err := ValidateRequest(request(RoleAssistant, block, server), DefaultPolicy().Limits); err != nil {
		t.Fatalf("a server tool and its block were refused: %v", err)
	}
	withSchema := server
	withSchema.InputSchema = json.RawMessage(`{"type":"object"}`)
	for name, invalid := range map[string]Request{
		"block in a user turn":      request(RoleUser, block),
		"block with text":           request(RoleAssistant, Content{Kind: ContentServerTool, Raw: block.Raw, Text: "x"}),
		"block without its block":   request(RoleAssistant, Content{Kind: ContentServerTool}),
		"citations on an image":     request(RoleUser, Content{Kind: ContentImage, URL: "https://example.com/a.png", Raw: json.RawMessage(`[]`)}),
		"citations not an array":    request(RoleAssistant, Content{Kind: ContentText, Text: "x", Raw: json.RawMessage(`{}`)}),
		"server tool with a schema": request(RoleAssistant, Content{Kind: ContentText, Text: "x"}, withSchema),
		"raw on a function tool":    request(RoleAssistant, Content{Kind: ContentText, Text: "x"}, Tool{Name: "f", InputSchema: json.RawMessage(`{"type":"object"}`), Raw: server.Raw}),
	} {
		var protocolError *ProtocolError
		if err := ValidateRequest(invalid, DefaultPolicy().Limits); !errors.As(err, &protocolError) || protocolError.Category != ErrorInvalidRequest {
			t.Errorf("%s: validation returned %v", name, err)
		}
	}
}
