package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A coding agent that reads an image file returns it in a tool result. The
// Responses API requires detail on every input_image, so a request that
// leaves it out fails at the backend.
func TestResponsesToolResultImageCarriesDefaultDetail(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"m","max_tokens":64,"messages":[
		{"role":"user","content":"look"},
		{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Read","input":{"file_path":"a.png"}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[
			{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}}]}]}]}`)
	request, _, _, err := engine.DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, body)
	require.NoError(t, err)
	upstream, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, llmprotocol.Envelope{})
	require.NoError(t, err)
	var wire struct {
		Input []struct {
			Type   string `json:"type"`
			Output []struct {
				Type   string `json:"type"`
				Detail string `json:"detail"`
			} `json:"output"`
		} `json:"input"`
	}
	require.NoError(t, json.Unmarshal(upstream.Body, &wire))
	for _, item := range wire.Input {
		if item.Type == "function_call_output" {
			require.Len(t, item.Output, 1)
			require.Equal(t, "input_image", item.Output[0].Type)
			require.Equal(t, "auto", item.Output[0].Detail)
			return
		}
	}
	t.Fatalf("no function_call_output in %s", upstream.Body)
}
