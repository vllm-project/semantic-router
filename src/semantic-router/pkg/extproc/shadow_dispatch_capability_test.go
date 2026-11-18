package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func withImageContent(ctx *RequestContext) {
	message := &ctx.SemanticRequest.Messages[0]
	message.Content = append(message.Content, llmprotocol.Content{
		Kind: llmprotocol.ContentImage, URL: "https://example.invalid/a.png",
	})
}

func shadowDroppedFor(reason string) float64 {
	return shadowCounter(shadowTestDecision, metrics.ShadowDispatchResultDropped, reason)
}

func TestShadowDispatchSkipsModelThatLacksARequestedModality(t *testing.T) {
	const reason = "backend_lacks_capability"
	for _, tc := range []struct {
		name         string
		capabilities []string
		image        bool
		sent         int
	}{
		{"image request, text-only card", []string{"text"}, true, 0},
		{"image request, card declares image input", []string{"text", "vision"}, true, 1},
		{"image request, no recognized declaration", nil, true, 1},
		{"text request, text-only card", []string{"text"}, false, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			backend := newShadowTestBackend(t)
			router, primaryModel := newShadowTestRouter(t, backend)
			params := router.Config.ModelConfig[shadowTestModel]
			params.Capabilities = tc.capabilities
			router.Config.ModelConfig[shadowTestModel] = params
			before := shadowDroppedFor(reason)

			var mutate func(*RequestContext)
			if tc.image {
				mutate = withImageContent
			}
			runShadowRequest(t, router, primaryModel, shadowTestPluginConfig(), mutate)
			waitForShadow(t, router)

			if got := backend.requestCount(); got != tc.sent {
				t.Fatalf("shadow backend requests = %d, want %d", got, tc.sent)
			}
			wantDropped := 0.0
			if tc.sent == 0 {
				wantDropped = 1
			}
			if got := shadowDroppedFor(reason) - before; got != wantDropped {
				t.Fatalf("dropped(%s) delta = %v, want %v", reason, got, wantDropped)
			}
		})
	}
}
