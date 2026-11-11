package extproc

import (
	"context"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// answerThroughFallback records a request routed to model-primary whose answer
// came from model-fallback-1, through the response-phase or the native chain.
func answerThroughFallback(t *testing.T, native bool, recorder *routerreplay.Recorder) *RequestContext {
	t.Helper()
	replay := &config.RouterReplayPluginConfig{Enabled: true}
	if native {
		session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-1")
		session.router.ReplayRecorder = recorder
		session.ctx.RouterReplayPluginConfig = replay
		session.router.startRouterReplay(session.ctx, "auto", "model-primary", "chat_fallback_decision")
		_, err := session.Fallback().Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
		require.NoError(t, err)
		session.fallback.succeeded(http.StatusOK)
		return session.ctx
	}
	router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
	router.ReplayRecorder = recorder
	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.RouterReplayPluginConfig = replay
	router.startRouterReplay(ctx, "auto", "model-primary", "chat_fallback_decision")
	ctx.UpstreamStatusCode = http.StatusServiceUnavailable
	router.fallbackCaller = func(context.Context, string, []byte, map[string]string) ([]byte, int, error) {
		return []byte(`{"id":"c1","object":"chat.completion","created":1,"model":"model-fallback-1",
			"choices":[{"index":0,"message":{"role":"assistant","content":"answer from the fallback"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":5,"completion_tokens":4,"total_tokens":9}}`), http.StatusOK, nil
	}
	require.NotNil(t, router.handleUpstreamTransportError([]byte("upstream 503"), ctx).GetImmediateResponse())
	return ctx
}

func TestFeedbackAfterFallbackCreditsTheModelThatAnswered(t *testing.T) {
	for _, native := range []bool{false, true} {
		name := "response_phase"
		if native {
			name = "native"
		}
		t.Run(name, func(t *testing.T) {
			recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
			t.Cleanup(func() { _ = recorder.Close() })
			ctx := answerThroughFallback(t, native, recorder)
			require.Equal(t, "model-fallback-1", ctx.VSRSelectedModel, "the client is told the fallback answered")

			rt := newRouterLearningRuntime(nil, recorder, nil)
			feedback := func(targetRef string) routerruntime.RouterOutcomeResult {
				return rt.UpdateOutcome(context.Background(), &routerruntime.RouterOutcome{
					ReplayID: ctx.RouterReplayID, Source: routerruntime.RouterOutcomeSourceAgent,
					Target: routerruntime.RouterOutcomeTargetModel, TargetRef: targetRef,
					Verdict: routerruntime.RouterOutcomeVerdictGoodFit, Score: 1,
				})
			}
			named := feedback("model-fallback-1")
			require.Equal(t, routerruntime.RouterOutcomeCodeOK, named.Code, named.Message)
			require.Equal(t, 1, named.Updated)
			require.Equal(t, 1, feedback("").Updated)
			primary := feedback("model-primary")
			require.Equal(t, routerruntime.RouterOutcomeCodeOwnershipMismatch, primary.Code)

			require.Equal(t, 2, rt.experienceSnapshot("", 0, "model-fallback-1").GoodFitCount)
			require.Zero(t, rt.experienceSnapshot("", 0, "model-primary").GoodFitCount, "the failed primary earns no credit")
		})
	}
}

// Only a candidate whose answer reached the client owns the feedback. A chain
// that failed, or whose answer a response policy blocked, leaves the routed model.
func TestAnsweringReplayModelFollowsTheFallbackVerdict(t *testing.T) {
	for _, tc := range []struct{ verdict, want string }{
		{"completed", "model-fallback-1"}, {"blocked", "model-primary"}, {"failed", "model-primary"}, {"", "model-primary"},
	} {
		record := routerreplay.RoutingRecord{SelectedModel: "model-primary"}
		if tc.verdict != "" {
			record.Outcomes = []routerreplay.Outcome{{Source: "fallback", Target: "request", TargetRef: "model-fallback-1", Verdict: tc.verdict}}
		}
		require.Equal(t, tc.want, answeringReplayModel(record), "verdict %q", tc.verdict)
	}
}
