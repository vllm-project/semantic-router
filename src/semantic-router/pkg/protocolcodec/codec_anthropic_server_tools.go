package protocolcodec

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Anthropic server tools, such as web_search_20250305, run on Anthropic's side
// and have no neutral form. Their definitions, the server_tool_use and
// *_tool_result blocks they leave in a reply, and the citations they attach
// to text are carried as written, so only Anthropic Messages encodes them.

// isAnthropicServerToolBlock names the blocks a server tool leaves in an
// assistant turn: its call and its result, whatever the tool.
func isAnthropicServerToolBlock(typeName string) bool {
	return typeName == "server_tool_use" ||
		typeName != "tool_result" && strings.HasSuffix(typeName, "_tool_result")
}

func decodeAnthropicServerTool(body json.RawMessage, typeName string) (llmprotocol.Tool, error) {
	var definition struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(body, &definition); err != nil || definition.Name == "" {
		return llmprotocol.Tool{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool",
			"Anthropic tool "+typeName+" requires a name", err)
	}
	return llmprotocol.Tool{Kind: llmprotocol.ToolKindServer, Name: definition.Name, Raw: append(json.RawMessage(nil), body...)}, nil
}

// decodeAnthropicServerToolBlock reads a server tool block, reporting false
// for any other block.
func decodeAnthropicServerToolBlock(body json.RawMessage) (llmprotocol.Content, bool) {
	typeName, err := anthropicContentDiscriminator(body)
	if err != nil || !isAnthropicServerToolBlock(typeName) {
		return llmprotocol.Content{}, false
	}
	return llmprotocol.Content{Kind: llmprotocol.ContentServerTool, Raw: append(json.RawMessage(nil), body...)}, true
}

// anthropicTextCitations keeps the citations of a text block, which only a
// server tool produces here, and rejects citations of any other block.
func anthropicTextCitations(typeName string, raw json.RawMessage) (json.RawMessage, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) || bytes.Equal(trimmed, []byte("[]")) {
		return nil, nil
	}
	if typeName != "text" || trimmed[0] != '[' {
		return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_citations",
			"Anthropic citations are carried only on text, as an array", nil)
	}
	return append(json.RawMessage(nil), trimmed...), nil
}
