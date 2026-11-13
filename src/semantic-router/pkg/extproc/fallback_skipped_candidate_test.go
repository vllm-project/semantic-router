package extproc

import (
	"context"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
)

func observeSkippedCandidates(t *testing.T) *observer.ObservedLogs {
	t.Helper()
	core, logs := observer.New(zapcore.WarnLevel)
	t.Cleanup(zap.ReplaceGlobals(zap.New(core)))
	return logs
}

func requireOneSkippedCandidate(t *testing.T, logs *observer.ObservedLogs, model string) {
	t.Helper()
	entries := logs.FilterMessage("fallback_candidate_skipped").All()
	require.Len(t, entries, 1)
	fields := entries[0].ContextMap()
	require.Equal(t, model, fields["candidate_model"])
	require.Equal(t, true, fields["can_fallback"])
	require.Contains(t, fields["error"], model)
}

// A candidate that cannot be prepared is passed over for the next one, in
// both gateways, and the log says which candidate and why.
func TestFallbackLogsASkippedCandidate(t *testing.T) {
	eligible := []string{"model-primary", "model-unknown", "model-fallback-1"}

	t.Run("native", func(t *testing.T) {
		logs := observeSkippedCandidates(t)
		session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), eligible...)
		step, err := session.Fallback().Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
		require.NoError(t, err)
		require.NotNil(t, step.Call)
		require.Equal(t, "model-fallback-1", step.Call.Route)
		requireOneSkippedCandidate(t, logs, "model-unknown")
	})

	t.Run("response phase", func(t *testing.T) {
		logs := observeSkippedCandidates(t)
		router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
		ctx := testFallbackRequestContext("model-primary", eligible)
		ctx.UpstreamStatusCode = http.StatusServiceUnavailable
		router.fallbackCaller = func(_ context.Context, model string, _ []byte, _ map[string]string) ([]byte, int, error) {
			require.Equal(t, "model-fallback-1", model)
			return []byte(`{"id":"chatcmpl-next","object":"chat.completion","created":1700000000,` +
				`"model":"model-fallback-1","choices":[{"index":0,"message":{"role":"assistant",` +
				`"content":"from the next candidate"},"finish_reason":"stop"}]}`), http.StatusOK, nil
		}
		resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
		require.NotNil(t, resp.GetImmediateResponse())
		require.Contains(t, string(resp.GetImmediateResponse().GetBody()), "from the next candidate")
		requireOneSkippedCandidate(t, logs, "model-unknown")
	})
}
