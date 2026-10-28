package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// anthropicToolErrorHistory is an Anthropic Messages history whose tool result
// carries the failure flag an agentic client sends after a failed tool call.
func anthropicToolErrorHistory() []byte {
	return []byte(`{"model":"m","max_tokens":64,"messages":[
		{"role":"user","content":"Read the record"},
		{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"read_record","input":{}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"[]","is_error":true}]}]}`)
}

func TestRouterProtocolPolicyDerivation(t *testing.T) {
	if got := routerProtocolPolicy(nil); got.LossyFeatures != llmprotocol.LossyReject {
		t.Fatalf("nil config must keep the strict default, got %q", got.LossyFeatures)
	}
	if got := routerProtocolPolicy(&config.RouterConfig{}); got.LossyFeatures != llmprotocol.LossyReject {
		t.Fatalf("empty knob must keep the strict default, got %q", got.LossyFeatures)
	}
	permissive := &config.RouterConfig{ProtocolLossyFeatures: config.ProtocolLossyAllowWithDiagnostic}
	if got := routerProtocolPolicy(permissive); got.LossyFeatures != llmprotocol.LossyAllowWithDiagnostic {
		t.Fatalf("permissive knob not honored, got %q", got.LossyFeatures)
	}
	strict := &config.RouterConfig{ProtocolLossyFeatures: config.ProtocolLossyReject}
	if got := routerProtocolPolicy(strict); got.LossyFeatures != llmprotocol.LossyReject {
		t.Fatalf("explicit reject must stay strict, got %q", got.LossyFeatures)
	}
}

func TestRouterProtocolPolicyDrivesToolResultErrorTranslation(t *testing.T) {
	history := anthropicToolErrorHistory()

	strict := &OpenAIRouter{ProtocolCodecs: protocolcodec.NewBuiltinRegistry()}
	engine, err := strict.protocolEngine()
	if err != nil {
		t.Fatal(err)
	}
	_, err = engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, history, nil)
	if err == nil {
		t.Fatal("default router must reject tool_result.is_error translation")
	}

	permissive := &OpenAIRouter{
		ProtocolCodecs: protocolcodec.NewBuiltinRegistry(),
		protocolPolicy: routerProtocolPolicy(&config.RouterConfig{ProtocolLossyFeatures: config.ProtocolLossyAllowWithDiagnostic}),
	}
	engine, err = permissive.protocolEngine()
	if err != nil {
		t.Fatal(err)
	}
	translated, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, history, nil)
	if err != nil {
		t.Fatalf("permissive router must translate the history: %v", err)
	}
	if len(translated.Diagnostics) != 1 || translated.Diagnostics[0].Field != "tool_result.is_error" {
		t.Fatalf("want one is_error diagnostic, got %+v", translated.Diagnostics)
	}
	if translated.Request.Messages[len(translated.Request.Messages)-1].Content[0].ToolResult.IsError == nil {
		t.Fatal("neutral failure flag must survive permissive translation")
	}
}

func TestDirectlyAssembledRouterKeepsStrictDefault(t *testing.T) {
	router := &OpenAIRouter{}
	engine, err := router.protocolEngine()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1,
		anthropicToolErrorHistory(), nil,
	); err == nil {
		t.Fatal("a router without a built policy must keep the strict default")
	}
}
