package extproc

import (
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// A request the router rejects at dispatch (here an unresolved provider
// credential) is answered with an immediate error and never reaches the primary
// backend. It must not be mirrored to the shadow model either.
func TestShadowIsNotDispatchedForRejectedRequest(t *testing.T) {
	tests := []struct {
		name string
		// originalModel is what the client asked for; it equals the routed model when
		// the request is pinned to a concrete model.
		originalModel func(primary string) string
	}{
		{name: "request routed to another model", originalModel: func(string) string { return "virtual" }},
		{name: "request pinned to the routed model", originalModel: func(primary string) string { return primary }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backend := newShadowTestBackend(t)
			router, primaryModel := newShadowTestRouter(t, backend)
			router.CredentialResolver = nil

			originalModel := tt.originalModel(primaryModel)
			request := testNeutralRequest(originalModel, "please shadow this prompt")
			ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
			decision := &config.Decision{Name: shadowTestDecision, ModelRefs: []config.ModelRef{{Model: primaryModel}}}
			ctx.VSRSelectedDecision = decision
			ctx.VSRSelectedDecisionName = decision.Name
			recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
			replayID, err := recorder.AddRecord(routerreplay.RoutingRecord{RequestID: ctx.RequestID, Decision: decision.Name})
			if err != nil {
				t.Fatalf("add replay record: %v", err)
			}
			ctx.RouterReplayID = replayID
			ctx.RouterReplayRecorder = recorder
			ctx.ShadowDispatchPluginConfig = shadowTestPluginConfig()

			response, err := router.handleEntrypointModelRouting(
				request, originalModel, decision.Name, entropy.ReasoningDecision{}, primaryModel, ctx,
			)
			if err != nil {
				t.Fatalf("handleEntrypointModelRouting: %v", err)
			}
			immediate := response.GetImmediateResponse()
			if immediate == nil || immediate.GetStatus().GetCode() != typev3.StatusCode_InternalServerError {
				t.Fatalf("expected the credential failure as an immediate 500, got %+v", response)
			}

			waitForShadow(t, router)
			if got := backend.requestCount(); got != 0 {
				t.Fatalf("shadow backend received %d request(s) although the client was rejected", got)
			}
		})
	}
}
