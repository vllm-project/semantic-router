package protocolcodec

import (
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Each decoded message keeps the position of the client message it came from,
// so a refusal can name the client's own field: the parts of a user message
// split at its tool results share one position, and a system message that only
// added a tool leaves a gap.
func TestAnthropicDecodingRecordsClientMessagePositions(t *testing.T) {
	request, _, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(`{
		"model":"m","max_tokens":64,"tools":[{"name":"Bash","input_schema":{"type":"object"}}],
		"messages":[
			{"role":"user","content":"run it"},
			{"role":"system","content":[{"type":"tool_addition","tool":{"type":"tool_definition","definition":{"name":"Read","input_schema":{"type":"object"}}}}]},
			{"role":"assistant","content":[{"type":"thinking","thinking":"plan","signature":"EqQBCkYIBxgCKkBexample"},{"type":"tool_use","id":"toolu_1","name":"Bash","input":{}}]},
			{"role":"user","content":[{"type":"text","text":"output:"},{"type":"tool_result","tool_use_id":"toolu_1","content":"ok"},{"type":"text","text":"go on"}]}
		]}`))
	if err != nil {
		t.Fatal(err)
	}
	var positions []int
	for _, message := range request.Messages {
		positions = append(positions, message.SourcePosition)
	}
	if want := []int{1, 3, 4, 4, 4}; !slices.Equal(positions, want) {
		t.Fatalf("positions = %v, want %v", positions, want)
	}
}

// A thinking block may carry an empty signature, as a reply without one
// renders it. It decodes without one and needs no signature capability, so
// any backend accepts it.
func TestAnthropicThinkingWithAnEmptySignatureNeedsNoCapability(t *testing.T) {
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(`{
		"model":"m","max_tokens":64,"messages":[
			{"role":"user","content":"What is 2+2?"},
			{"role":"assistant","content":[{"type":"thinking","thinking":"simple arithmetic","signature":""},{"type":"text","text":"4"}]},
			{"role":"user","content":"And 3+3?"}
		]}`))
	if err != nil {
		t.Fatal(err)
	}
	if thinking := request.Messages[1].Content[0]; thinking.Kind != llmprotocol.ContentReasoning || thinking.Text != "simple arithmetic" || thinking.Signature != "" {
		t.Fatalf("thinking = %+v", thinking)
	}
	if llmprotocol.RequiredCapabilities(request).Supports(llmprotocol.CapabilityReasoningSignature) {
		t.Fatal("an empty signature requires the signature capability")
	}
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		if _, err := engine.EncodeRequest(target, request, envelope); err != nil {
			t.Fatalf("%s refused thinking with an empty signature: %v", target, err)
		}
	}
}
