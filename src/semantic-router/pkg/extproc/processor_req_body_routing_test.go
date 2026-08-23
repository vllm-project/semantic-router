package extproc

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dispatchauthority"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

func TestHandleAutoModelRoutingEmitsLogicalModelAndDispatchCapability(t *testing.T) {
	router := routingTestRouter("qwen14b-dev")
	request := testNeutralRequest("MoM", "hello from routed model")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.ModalityClassification = &ModalityClassificationResult{
		Modality: ModalityBoth, Confidence: 0.97, Method: "signal",
	}

	response, err := router.handleEntrypointModelRouting(
		request, "MoM", "", entropy.ReasoningDecision{}, "qwen14b-dev", ctx,
	)
	if err != nil {
		t.Fatalf("handleEntrypointModelRouting returned error: %v", err)
	}
	common := response.GetRequestBody().GetResponse()
	headersByName := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())
	if got := headersByName[headers.SelectedModel]; got != "qwen14b-dev" {
		t.Fatalf("selected model header = %q", got)
	}
	if got := headersByName[backendinvoker.DispatchCapabilityHeader]; got != "capability" {
		t.Fatalf("dispatch capability = %q", got)
	}
	if got := headersByName["x-vsr-destination-endpoint"]; got != "" {
		t.Fatalf("ExtProc leaked a physical destination: %q", got)
	}

	var body map[string]any
	if err := json.Unmarshal(common.GetBodyMutation().GetBody(), &body); err != nil {
		t.Fatalf("decode routed request: %v", err)
	}
	if got := body["model"]; got != "qwen14b-dev" {
		t.Fatalf("logical model = %#v", got)
	}
	if request.Model != "qwen14b-dev" || ctx.SemanticRequest != request {
		t.Fatalf("neutral request was not updated in place: %+v", request)
	}
	if ctx.ModalityClassification.Modality != ModalityBoth {
		t.Fatalf("backend dispatch lost modality evidence: %+v", ctx.ModalityClassification)
	}
}

func TestHandleAutoModelRoutingSameModelEncodesCurrentSemanticRequest(t *testing.T) {
	router := routingTestRouter("auto")
	request := testNeutralRequest("auto", "short semantic content")
	request.Generation = 4
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)

	response, err := router.handleEntrypointModelRouting(
		request, "auto", "", entropy.ReasoningDecision{}, "auto", ctx,
	)
	if err != nil {
		t.Fatalf("handleEntrypointModelRouting returned error: %v", err)
	}
	common := response.GetRequestBody().GetResponse()
	var body struct {
		Model    string `json:"model"`
		Messages []struct {
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(common.GetBodyMutation().GetBody(), &body); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	if body.Model != "auto" || len(body.Messages) != 1 || body.Messages[0].Content != "short semantic content" {
		t.Fatalf("unexpected encoded semantic request: %+v", body)
	}
}

func TestSpecifiedModelPreservesAnthropicWireAtExtProcBoundary(t *testing.T) {
	router := routingTestRouter("test-model")
	request := testNeutralRequest("test-model", "Explain the incident.")
	request.Stream = true
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(256)
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, request)
	ctx.ExpectStreamingResponse = true

	response, err := router.handleSpecifiedModelRouting(request, "test-model", "", ctx)
	if err != nil {
		t.Fatalf("handleSpecifiedModelRouting: %v", err)
	}
	common := response.GetRequestBody().GetResponse()
	var body map[string]json.RawMessage
	if err := json.Unmarshal(common.GetBodyMutation().GetBody(), &body); err != nil {
		t.Fatalf("decode Anthropic request: %v", err)
	}
	if _, ok := body["messages"]; !ok {
		t.Fatalf("Anthropic request omitted messages: %s", common.GetBodyMutation().GetBody())
	}
	if string(body["stream"]) != "true" || string(body["max_tokens"]) != "256" {
		t.Fatalf("Anthropic options were not preserved: %s", common.GetBodyMutation().GetBody())
	}
	if got := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())[":path"]; got != "/v1/messages" {
		t.Fatalf("ExtProc source-format path = %q", got)
	}
}

func routingTestRouter(model string) *OpenAIRouter {
	return &OpenAIRouter{
		Config: &config.RouterConfig{BackendModels: config.BackendModels{
			DefaultModel: model,
			ModelConfig: map[string]config.ModelParams{
				model: {ResourceID: "model-id", ResourceRevision: 1},
			},
		}},
		DispatchCapabilities: dispatchCapabilityRuntimeStub{},
	}
}

func routingTestContext(format llmprotocol.WireFormat, request *llmprotocol.Request) *RequestContext {
	return &RequestContext{
		Headers:         map[string]string{},
		SourceFormat:    format,
		SemanticRequest: request,
		TraceContext: withVerifiedDispatchGrant(
			context.Background(), dispatchauthority.VerifiedGrant{},
		),
	}
}
