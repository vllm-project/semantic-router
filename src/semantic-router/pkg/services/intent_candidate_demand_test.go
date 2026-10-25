package services

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestPreviewCandidateDemandUsesEffectiveNeutralRequest(t *testing.T) {
	req := IntentRequest{Messages: []IntentMessage{{Role: "user", Content: json.RawMessage(`[{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"https://example.invalid/image"}}]`)}}, Tools: []json.RawMessage{json.RawMessage(`{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}`)}, MaxCompletionTokens: json.RawMessage(`128`)}
	input, err := req.resolveSignalInput()
	if err != nil {
		t.Fatal(err)
	}
	if input.semanticRequest == nil {
		t.Fatal("structured selection facts unavailable")
	}
	payload, _ := config.NewStructuredPayload(map[string]any{"default_max_tokens": 64})
	d := &config.Decision{Plugins: []config.DecisionPlugin{{Type: "request_params", Configuration: payload}}}
	stub := &evalModelSelectorStub{}
	service := &ClassificationService{}
	service.SetEvalModelSelector(stub)
	service.populateEvalModelSelection(&EvalResponse{}, input, &decision.DecisionResult{Decision: d}, nil)
	body, err := intentRequestEnvelope(req, "")
	if err != nil {
		t.Fatal(err)
	}
	actual, _, _, err := (protocolcodec.OpenAIChatCodec{}).DecodeRequest(body, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	demand, _ := selection.EffectiveCandidateDemand(&actual, d)
	got := stub.input.Demand
	if !got.Known || got.InputTokens != demand.InputTokens || got.MaxOutputTokens == nil || *got.MaxOutputTokens != 128 || !got.ModelCapabilities.Supports(llmprotocol.CapabilityImageInput) || !got.ModelCapabilities.Supports(llmprotocol.CapabilityTools) {
		t.Fatalf("selection demand=%+v expected=%+v", got, demand)
	}
	// Legacy context contains the caller's output reserve. The new demand never does.
	if stub.input.ContextTokenCount <= got.InputTokens {
		t.Fatalf("legacy and neutral accounting lost distinction: legacy=%d input=%d", stub.input.ContextTokenCount, got.InputTokens)
	}
}

func TestPreviewUnknownNeutralFactsRemainUnknown(t *testing.T) {
	req := IntentRequest{Text: "hello", Functions: []json.RawMessage{json.RawMessage(`{"name":"legacy"}`)}}
	input, err := req.resolveSignalInput()
	if err != nil {
		t.Fatal(err)
	}
	if input.semanticRequest != nil {
		t.Fatal("unsupported legacy function semantics must not masquerade as known neutral demand")
	}
}
