package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// Behind a decision, Claude Code's adaptive thinking on a Responses backend
// follows the decision's use_reasoning, as it already does on Chat and
// Messages backends. An explicit thinking-off from the client is still kept.
func TestResponsesBackendFollowsTheDecisionForAdaptiveThinking(t *testing.T) {
	router := routingTestRouter("chat")
	model := router.Config.ModelConfig["chat"]
	model.APIFormat = config.APIFormatResponses
	model.ReasoningFamily = "reasoning"
	router.Config.ModelConfig["chat"] = model
	router.Config.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{
		"reasoning": {Type: config.ReasoningFamilyTypeReasoningEffort, Modes: []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled}},
	}
	decode := func(body string) llmprotocol.Request {
		request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(body))
		if err != nil {
			t.Fatal(err)
		}
		return request
	}
	for _, enabled := range []bool{false, true} {
		useReasoning := enabled
		decision := &config.Decision{Name: "agent", ModelRefs: []config.ModelRef{{Model: "chat", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: &useReasoning}}}}
		request := decode(adaptiveClaudeCodeRequest)
		if !router.applySemanticReasoningMode(&request, "chat", llmprotocol.OpenAIResponsesV1, enabled, decision) {
			t.Fatalf("use_reasoning=%v: adaptive thinking was kept instead of the decision's", enabled)
		}
		// The OpenAI-compatible contract carries the decision as an effort:
		// "none" for off, the decision's effort for on.
		want := "none"
		if enabled {
			want = router.getReasoningEffort(decision, "chat")
		}
		if request.ReasoningMode == llmprotocol.ReasoningModeAdaptive || request.ReasoningEffort != want {
			t.Fatalf("use_reasoning=%v: mode %q effort %q, want effort %q", enabled, request.ReasoningMode, request.ReasoningEffort, want)
		}
	}
	useReasoning := true
	decision := &config.Decision{Name: "agent", ModelRefs: []config.ModelRef{{Model: "chat", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: &useReasoning}}}}
	off := decode(`{"model":"chat","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"disabled"}}`)
	if router.applySemanticReasoningMode(&off, "chat", llmprotocol.OpenAIResponsesV1, true, decision) || off.ReasoningMode != llmprotocol.ReasoningModeDisabled {
		t.Fatalf("explicit thinking-off was overridden: %q", off.ReasoningMode)
	}
}
