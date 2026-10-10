package extproc

import (
	"testing"
	"time"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A dispatch failure reaches tool selection as an immediate response. The debug
// observability headers must not rewrite it into a request-body continue, or the
// client's error is lost and the request is forwarded without credentials.
func TestEmitToolObservabilityKeepsImmediateErrorResponse(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	router.CredentialResolver = nil
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, testNeutralRequest(model, "hello"))
	ctx.Headers = map[string]string{headers.VSRDebug: "true"}

	dispatch, err := router.prepareProviderDispatch(ctx.SemanticRequest, model, "", false, ctx)
	if err != nil {
		t.Fatalf("prepareProviderDispatch: %v", err)
	}
	response := router.buildProviderDispatchResponse(dispatch, ctx)
	if response.GetImmediateResponse() == nil {
		t.Fatalf("expected a credential failure, got %+v", response)
	}

	emitToolObservability(&response, ctx, "default", 0.87, 4*time.Millisecond)

	immediate := response.GetImmediateResponse()
	if immediate == nil {
		t.Fatalf("tool observability replaced the credential failure with %+v", response)
	}
	if immediate.GetStatus().GetCode() != typev3.StatusCode_InternalServerError {
		t.Fatalf("status = %v, want %v", immediate.GetStatus().GetCode(), typev3.StatusCode_InternalServerError)
	}
}
