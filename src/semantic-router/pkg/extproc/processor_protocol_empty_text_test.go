package extproc

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestChatEmptyTextToolHistoryDispatchesToAnthropic(t *testing.T) {
	router := routingTestRouter("test-model")
	body := []byte(`{"model":"test-model","max_tokens":20,"messages":[
  {"role":"system","content":""},
  {"role":"user","content":"hi"},
  {"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},
  {"role":"tool","tool_call_id":"c1","content":""}]}`)
	engine := protocolcodec.NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatal(err)
	}
	before, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, &request)
	ctx.ProtocolEnvelope = envelope
	dispatch, err := router.prepareProviderDispatch(&request, "test-model", "", false, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if dispatch.targetFormat != llmprotocol.AnthropicMessagesV1 {
		t.Fatalf("target = %s", dispatch.targetFormat)
	}
	encoded, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		System   json.RawMessage
		Messages []struct {
			Role    string
			Content []struct {
				Type      string
				ID        string
				ToolUseID string `json:"tool_use_id"`
				Content   json.RawMessage
			}
		}
	}
	if err = json.Unmarshal(encoded, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.System) != 0 || len(wire.Messages) != 3 || bytes.Contains(encoded, []byte(`"text":""`)) {
		t.Fatalf("invalid dispatched history: %s", encoded)
	}
	assistant, tool := wire.Messages[1].Content, wire.Messages[2].Content
	if len(assistant) != 1 || assistant[0].Type != "tool_use" || assistant[0].ID != "c1" || len(tool) != 1 || tool[0].Type != "tool_result" || tool[0].ToolUseID != "c1" || string(tool[0].Content) != "[]" {
		t.Fatalf("tool history changed: %s", encoded)
	}
	after, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("provider encoding mutated the neutral request")
	}
}

func TestEmptyInstructionNormalizationPreservesPromptCacheMarker(t *testing.T) {
	request := promptCacheTestRequest()
	request.Instructions[0].Content = append(request.Instructions[0].Content,
		llmprotocol.Content{Kind: llmprotocol.ContentText})
	request.Tools = nil
	ctx := &RequestContext{
		SourceFormat:        llmprotocol.OpenAIChatV1,
		TargetFormat:        llmprotocol.AnthropicMessagesV1,
		SemanticRequest:     request,
		VSRSelectedDecision: promptCacheTestDecision(map[string]interface{}{"enabled": true, "ttl": "1h"}),
	}
	body, err := (&OpenAIRouter{}).encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var wire promptCacheWireRequest
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.System) != 2 || wire.System[0].Text != "stable preface" ||
		wire.System[1].Text != "reusable instructions" || wire.System[0].CacheControl != nil ||
		wire.System[1].CacheControl == nil || wire.System[1].CacheControl.Type != "ephemeral" ||
		wire.System[1].CacheControl.TTL != "1h" {
		t.Fatalf("normalized instructions or cache boundary changed: %s", body)
	}
	assertPromptCacheReceipt(t, ctx, promptCacheActionInserted, 1, 0)
	if len(request.Instructions[0].Content) != 3 || countPromptCacheMarkers(*request) != 0 {
		t.Fatal("provider normalization changed the retained request")
	}
}
