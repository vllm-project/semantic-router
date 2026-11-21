package extproc

import (
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A dispatch failure reaches the tools plugin as an immediate response. The
// plugin must neither rewrite it into a request-body continue, or the
// client's error is lost and the request is forwarded without credentials,
// nor ask the decision model about a request that is already refused.
func TestToolsPluginKeepsImmediateErrorResponse(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	router.CredentialResolver = nil
	judge := &toolsJudge{need: 0.9}
	router.decisionDecider = judge
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, toolsRequest("a", "b"))
	ctx.Headers = map[string]string{headers.VSRDebug: "true"}
	ctx.VSRSelectedDecision = toolsContext(config.ToolsPluginConfig{Enabled: true, Select: &config.ToolsSelectConfig{Mode: config.ToolsSelectModeFilter}}).VSRSelectedDecision

	dispatch, err := router.prepareProviderDispatch(ctx.SemanticRequest, model, "", false, ctx)
	if err != nil {
		t.Fatalf("prepareProviderDispatch: %v", err)
	}
	response := router.buildProviderDispatchResponse(dispatch, ctx)
	if response.GetImmediateResponse() == nil {
		t.Fatalf("expected a credential failure, got %+v", response)
	}

	router.applyToolsPlugin(ctx.SemanticRequest, response, ctx)

	immediate := response.GetImmediateResponse()
	if immediate == nil {
		t.Fatalf("the tools plugin replaced the credential failure with %+v", response)
	}
	if immediate.GetStatus().GetCode() != typev3.StatusCode_InternalServerError {
		t.Fatalf("status = %v, want %v", immediate.GetStatus().GetCode(), typev3.StatusCode_InternalServerError)
	}
	if len(judge.asked()) != 0 {
		t.Fatal("a refused request is not selected for")
	}
}
