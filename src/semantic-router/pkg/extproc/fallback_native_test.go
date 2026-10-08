package extproc

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// nativeFallbackSession is a session whose primary dispatch went to
// model-primary, with the given models eligible in order.
func nativeFallbackSession(t *testing.T, policy fallback.FallbackPolicy, eligible ...string) *routingSession {
	t.Helper()
	router, _ := setupFallbackTestRouter(t, policy)
	return &routingSession{
		router: router,
		ctx:    testFallbackRequestContext("model-primary", eligible),
		clientHeader: routing.Header{
			{Name: ":method", Value: "POST"},
			{Name: ":path", Value: "/v1/chat/completions"},
			{Name: "content-type", Value: "application/json"},
			{Name: "x-client-trace", Value: "kept"},
		},
	}
}

func failedWith(status int, body string) routing.Outcome {
	return routing.Outcome{Route: "model-primary", Status: status, Body: []byte(body), Duration: time.Millisecond}
}

func TestNativeFallbackTakesTheResponsePhaseFallbackOff(t *testing.T) {
	session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-1")
	ctx := session.ctx
	require.True(t, session.router.shouldAttemptFallback(ctx))

	session.Fallback()
	require.False(t, session.router.shouldAttemptFallback(ctx))
	ctx.UpstreamStatusCode = http.StatusServiceUnavailable
	require.Nil(t, session.router.maybeExecuteFallback([]byte("unavailable"), ctx))
	require.Nil(t, ctx.FallbackRecord, "the response phase must not start a second chain")
}

func TestNativeFallbackPreparesTheNextCandidate(t *testing.T) {
	session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-1")
	ctx := session.ctx
	primaryRequest := ctx.SemanticRequest

	step, err := session.Fallback().Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
	require.NoError(t, err)
	require.NotNil(t, step.Call)
	require.Nil(t, step.Immediate)
	require.Equal(t, "model-fallback-1", step.Call.Route)
	require.Equal(t, "POST", step.Call.Request.Header.Get(":method"))
	require.Equal(t, "kept", step.Call.Request.Header.Get("x-client-trace"))
	require.Equal(t, "model-fallback-1", step.Call.Request.Header.Get(routing.RouteHeader))
	require.Contains(t, string(step.Call.Request.Body), `"model":"model-fallback-1"`)

	// The context switches to the candidate only when the candidate succeeds.
	require.Equal(t, "model-primary", ctx.VSRSelectedModel)
	require.Same(t, primaryRequest, ctx.SemanticRequest)
	require.Len(t, ctx.FallbackRecord.Attempts, 1)
	require.Equal(t, "upstream returned status 503: unavailable", ctx.FallbackRecord.Attempts[0].ErrorMessage)
	require.False(t, ctx.FallbackAuditRecorded)
}

func TestNativeFallbackSwitchesToTheCandidateOnItsSuccess(t *testing.T) {
	session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-1")
	ctx := session.ctx
	step, err := session.Fallback().Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
	require.NoError(t, err)
	require.NotNil(t, step.Call)

	session.fallback.succeeded(http.StatusOK)
	require.Equal(t, "model-fallback-1", ctx.VSRSelectedModel)
	require.Equal(t, "model-fallback-1", ctx.RequestModel)
	require.Equal(t, headers.ResponsePathFallback, ctx.ResponsePath)
	require.Equal(t, "succeeded", ctx.FallbackRecord.FinalStatus)
	require.Len(t, ctx.FallbackRecord.Attempts, 2)
	require.True(t, ctx.FallbackAuditRecorded)

	// A later success report, such as the response phases seeing the same
	// response again, changes nothing.
	session.fallback.succeeded(http.StatusOK)
	require.Len(t, ctx.FallbackRecord.Attempts, 2)
}

func TestNativeFallbackEndsWithThePrimaryContext(t *testing.T) {
	session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-1")
	ctx := session.ctx
	primaryRequest, primaryCandidate := ctx.SemanticRequest, ctx.VSRSelectedCandidate
	chain := session.Fallback()

	step, err := chain.Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
	require.NoError(t, err)
	require.NotNil(t, step.Call)
	step, err = chain.Next(context.Background(), failedWith(http.StatusBadGateway, "bad gateway"))
	require.NoError(t, err)
	require.Nil(t, step.Call, "every eligible candidate was tried")
	require.Nil(t, step.Immediate)

	require.Equal(t, "model-primary", ctx.VSRSelectedModel)
	require.Same(t, primaryRequest, ctx.SemanticRequest)
	require.Equal(t, primaryCandidate, ctx.VSRSelectedCandidate)
	require.Len(t, ctx.FallbackRecord.Attempts, 2)
	require.Equal(t, "model-fallback-1", ctx.FallbackRecord.Attempts[1].Model)
	require.Equal(t, "bad gateway", ctx.FallbackRecord.Attempts[1].ErrorMessage)
	require.True(t, ctx.FallbackAuditRecorded)

	session.fallback.succeeded(http.StatusOK)
	require.Equal(t, "model-primary", ctx.VSRSelectedModel, "no candidate is in flight once the chain ended")
}

func TestNativeFallbackBoundsACandidateLikeTheResponsePhase(t *testing.T) {
	candidatePerTry := func(t *testing.T, session *routingSession) *time.Duration {
		t.Helper()
		step, err := session.Fallback().Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
		require.NoError(t, err)
		require.NotNil(t, step.Call)
		if len(step.Call.Reliability) == 0 {
			return nil
		}
		return step.Call.Reliability[0].PerTryTimeout
	}
	policy := fallback.DefaultEnabledPolicy()
	policy.PerAttemptTimeout = 2 * time.Second

	t.Run("per-attempt timeout", func(t *testing.T) {
		session := nativeFallbackSession(t, policy, "model-primary", "model-fallback-1")
		require.Equal(t, 2*time.Second, *candidatePerTry(t, session))
	})
	t.Run("a tighter decision per-try timeout stays", func(t *testing.T) {
		session := nativeFallbackSession(t, policy, "model-primary", "model-fallback-1")
		session.ctx.VSRSelectedDecision = &config.Decision{
			Name: "d", Reliability: &config.DecisionReliability{PerTryTimeout: "1s", RetryOn: "reset"},
		}
		require.Equal(t, time.Second, *candidatePerTry(t, session))
	})
	t.Run("a looser provider per-try timeout tightens", func(t *testing.T) {
		session := nativeFallbackSession(t, policy, "model-primary", "model-fallback-1")
		params := session.router.Config.ModelConfig["model-fallback-1"]
		params.Reliability.PerTryTimeout = "5s"
		session.router.Config.ModelConfig["model-fallback-1"] = params
		require.Equal(t, 2*time.Second, *candidatePerTry(t, session))
	})
	t.Run("what is left of the total timeout", func(t *testing.T) {
		bounded := policy
		bounded.TotalTimeout = 3 * time.Second
		session := nativeFallbackSession(t, bounded, "model-primary", "model-fallback-1")
		session.ctx.ProcessingStartTime = time.Now().Add(-2500 * time.Millisecond)
		got := *candidatePerTry(t, session)
		require.True(t, got > 0 && got <= 500*time.Millisecond, "per-try = %v", got)
	})
	t.Run("no budget, no bound", func(t *testing.T) {
		unbounded := policy
		unbounded.PerAttemptTimeout, unbounded.TotalTimeout = 0, 0
		session := nativeFallbackSession(t, unbounded, "model-primary", "model-fallback-1")
		require.Nil(t, candidatePerTry(t, session))
	})
}

func TestNativeFallbackStopsWhereThePolicyStops(t *testing.T) {
	t.Run("non-retryable status", func(t *testing.T) {
		session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-1")
		step, err := session.Fallback().Next(context.Background(), failedWith(http.StatusBadRequest, "bad request"))
		require.NoError(t, err)
		require.Nil(t, step.Call)
		require.Len(t, session.ctx.FallbackRecord.Attempts, 1)
	})
	t.Run("total timeout spent", func(t *testing.T) {
		policy := fallback.DefaultEnabledPolicy()
		policy.TotalTimeout = 50 * time.Millisecond
		session := nativeFallbackSession(t, policy, "model-primary", "model-fallback-1")
		session.ctx.ProcessingStartTime = time.Now().Add(-time.Second)
		step, err := session.Fallback().Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
		require.NoError(t, err)
		require.Nil(t, step.Call)
		require.Equal(t, "total_deadline_exceeded", session.ctx.FallbackRecord.FinalStatus)
	})
	t.Run("one eligible model", func(t *testing.T) {
		session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary")
		step, err := session.Fallback().Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
		require.NoError(t, err)
		require.Nil(t, step.Call)
		require.Nil(t, session.ctx.FallbackRecord)
	})
	t.Run("ended session", func(t *testing.T) {
		session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-1")
		chain := session.Fallback()
		session.closed = true
		_, err := chain.Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
		require.ErrorIs(t, err, errSessionEnded)
	})
}
