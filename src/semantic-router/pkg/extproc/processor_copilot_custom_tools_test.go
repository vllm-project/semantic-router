package extproc

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Both requests of a captured Copilot CLI tool loop with a GPT model ID must
// pass Router ingress and reach a Chat backend with the custom tool intact.
func TestExtProcDispatchesCopilotCustomToolLoopToChatBackend(t *testing.T) {
	raw, err := os.ReadFile("../protocolcodec/testdata/clients/copilot-cli-1.0.88-gpt-tool-loop.json")
	if err != nil {
		t.Fatal(err)
	}
	var turns []json.RawMessage
	if err := json.Unmarshal(raw, &turns); err != nil {
		t.Fatal(err)
	}
	for turn, body := range turns {
		router := &OpenAIRouter{}
		ctx := &RequestContext{
			SourceFormat: llmprotocol.OpenAIChatV1,
			RequestID:    "request_copilot_tool_loop",
			TraceContext: t.Context(),
		}
		request, immediate := router.prepareProtocolRequest(body, ctx)
		if immediate != nil || request == nil {
			t.Fatalf("turn %d rejected at ingress: %+v", turn+1, ctx.ImmediateProtocolError)
		}
		request.Model = "routed-model"
		request.Generation++
		ctx.TargetFormat = llmprotocol.OpenAIChatV1

		dispatch, err := router.encodeDispatchRequest(ctx)
		if err != nil {
			t.Fatalf("turn %d dispatch: %v", turn+1, err)
		}
		var chat struct {
			Tools []struct {
				Type   string          `json:"type"`
				Custom json.RawMessage `json:"custom"`
			} `json:"tools"`
		}
		if err := json.Unmarshal(dispatch, &chat); err != nil {
			t.Fatal(err)
		}
		custom := 0
		for _, tool := range chat.Tools {
			if tool.Type == "custom" && len(tool.Custom) > 0 {
				custom++
			}
		}
		if custom != 1 {
			t.Fatalf("turn %d dispatch carried %d custom tools, want 1: %s", turn+1, custom, dispatch)
		}
	}
}
