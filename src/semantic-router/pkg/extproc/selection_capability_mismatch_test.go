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

// Claude Code sends adaptive thinking with a display setting on every request,
// which only the Messages codec can express.
func claudeCodeThinkingRequest(model string) *llmprotocol.Request {
	request := testNeutralRequest(model, "Create hello.txt containing hi.")
	request.ReasoningMode = llmprotocol.ReasoningModeAdaptive
	request.ReasoningDisplay = "omitted"
	request.ReasoningEffort = "medium"
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

	request := claudeCodeThinkingRequest("auto")
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
		!strings.Contains(autoError.Message, "reasoning_adaptive") {
		t.Fatalf("client error = %v, want the unsupported reasoning_adaptive capability", autoError)
	}

	named := claudeCodeThinkingRequest(chatModel)
	_, err = router.prepareProviderDispatch(named, chatModel, "", false, routingTestContext(llmprotocol.AnthropicMessagesV1, named))
	var namedError *llmprotocol.ProtocolError
	if !errors.As(err, &namedError) || namedError.Message != autoError.Message {
		t.Fatalf("named-model error = %v, want the auto-routed message %q", err, autoError.Message)
	}
}
