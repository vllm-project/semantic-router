package protocolcodec

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Claude Code sends tool_reference blocks inside tool results once tool search
// runs. The refusal has to name the block, or the log cannot say which of the
// many block types a client sent.
func TestAnthropicUnsupportedContentNamesTheType(t *testing.T) {
	body := []byte(`{"model":"m","max_tokens":64,"messages":[
		{"role":"user","content":"find a tool"},
		{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"ToolSearch","input":{"query":"notebook"}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"tool_reference","tool_name":"NotebookEdit"}]}]}]}`)
	_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, body)
	require.ErrorContains(t, err, `Anthropic content type "tool_reference" is unsupported`)
}

// Claude Code's WebSearch sends Anthropic's hosted web_search tool, which no
// backend behind the Router can run. The refusal names the tool type.
func TestAnthropicUnsupportedToolNamesTheType(t *testing.T) {
	body := []byte(`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"Perform a web search for the query: vllm"}],
		"tools":[{"type":"web_search_20250305","name":"web_search","max_uses":8}]}`)
	_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, body)
	require.ErrorContains(t, err, `Anthropic tool type "web_search_20250305" is unsupported`)
}
