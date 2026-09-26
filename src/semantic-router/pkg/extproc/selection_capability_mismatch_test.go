package extproc

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Explicitly disabling reasoning needs a configured backend off control;
// unannotated Chat candidates cannot promise that semantic behavior.
func unsupportedThinkingRequest(model string) *llmprotocol.Request {
	request := testNeutralRequest(model, "Create hello.txt containing hi.")
	request.ReasoningMode = llmprotocol.ReasoningModeDisabled
	request.Trusted.SourceFormat = llmprotocol.AnthropicMessagesV1
	return request
}

func TestAutoRoutingReportsUnsupportedCapabilityLikeNamedModel(t *testing.T) {
	router, chatModel := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	router.Config.Decisions = []config.Decision{{
		Name: "default-route", Priority: 1, ModelRefs: []config.ModelRef{{Model: chatModel}},
	}}
	classifier, err := classification.NewClassifier(router.Config, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = classifier.Close() })
	router.Classifier = classifier

	request := unsupportedThinkingRequest("auto")
	ctx := &RequestContext{
		RequestID: "capability-mismatch", RequestModel: "auto", TraceContext: context.Background(),
		SemanticRequest: request, SourceFormat: llmprotocol.AnthropicMessagesV1,
	}
	_, response := router.runRequestPreRoutingStages("auto", extractSemanticRequestSignals(request), ctx)
	if got := response.GetImmediateResponse().GetStatus().GetCode(); got != 400 {
		t.Fatalf("auto-routed status = %d, body %s; want the 400 a named model gets", got, response.GetImmediateResponse().GetBody())
	}
	autoError := ctx.ImmediateProtocolError
	if autoError == nil || autoError.Category != llmprotocol.ErrorUnsupportedFeature ||
		!strings.Contains(autoError.Message, "reasoning-off control") {
		t.Fatalf("client error = %v, want the unsupported reasoning-off control", autoError)
	}

	named := unsupportedThinkingRequest(chatModel)
	_, err = router.prepareProviderDispatch(named, chatModel, "", false, routingTestContext(llmprotocol.AnthropicMessagesV1, named))
	var namedError *llmprotocol.ProtocolError
	if !errors.As(err, &namedError) || namedError.Message != autoError.Message {
		t.Fatalf("named-model error = %v, want the auto-routed message %q", err, autoError.Message)
	}
}

func TestAutoRoutingKeepsMixedContextAndWireExclusionsUnavailable(t *testing.T) {
	router, chatModel := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	chatParams := router.Config.ModelConfig[chatModel]
	chatParams.ContextWindowSize = 32_768
	router.Config.ModelConfig[chatModel] = chatParams
	messagesModel := "messages-small-context"
	messagesParams := chatParams
	messagesParams.APIFormat = config.APIFormatAnthropic
	messagesParams.ContextWindowSize = 1
	router.Config.ModelConfig[messagesModel] = messagesParams
	router.Config.Decisions = []config.Decision{{
		Name: "default-route", Priority: 1,
		ModelRefs: []config.ModelRef{{Model: messagesModel}, {Model: chatModel}},
	}}
	classifier, err := classification.NewClassifier(router.Config, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = classifier.Close() })
	router.Classifier = classifier

	request := unsupportedThinkingRequest("auto")
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, request)
	ctx.RequestModel = "auto"
	_, response := router.runRequestPreRoutingStages("auto", extractSemanticRequestSignals(request), ctx)
	if ctx.VSRContextTokenCount <= messagesParams.ContextWindowSize {
		t.Fatalf("context estimate %d did not exercise the context-window exclusion", ctx.VSRContextTokenCount)
	}
	if got := response.GetImmediateResponse().GetStatus().GetCode(); got != 503 {
		t.Fatalf("mixed context and wire exclusions returned %d, body %s; want fail-closed 503", got, response.GetImmediateResponse().GetBody())
	}
	if ctx.ImmediateProtocolError != nil {
		t.Fatalf("mixed exclusions were attributed solely to the caller: %v", ctx.ImmediateProtocolError)
	}
}
