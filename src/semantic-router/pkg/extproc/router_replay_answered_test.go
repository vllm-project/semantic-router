package extproc

import (
	"context"
	"errors"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/principal"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// openReplay opens the replay record a routed request has once its decision
// is known, as provider dispatch does before encoding.
func openReplay(t *testing.T, router *OpenAIRouter, ctx *RequestContext, model, decision string) {
	t.Helper()
	router.ReplayRecorder = routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	replayConfig := config.DefaultRouterReplayPluginConfig()
	replayConfig.Enabled = true
	replayConfig.PromptCapture = config.PromptCaptureRaw
	ctx.RouterReplayPluginConfig = &replayConfig
	router.startRouterReplay(ctx, "vllm-sr/auto", model, decision)
	require.NotEmpty(t, ctx.RouterReplayID)
}

// startAnsweredReplay opens the replay record of a routed Anthropic request.
func startAnsweredReplay(t *testing.T, ctx *RequestContext) *OpenAIRouter {
	t.Helper()
	router := &OpenAIRouter{}
	ctx.RequestID = "answered-request"
	ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
	ctx.SemanticRequest = testNeutralRequest("vllm-sr/auto", "say ok")
	ctx.VSRSelectedDecision = &config.Decision{Name: "all"}
	openReplay(t, router, ctx, "qwen-msg", "all")
	return router
}

func requireReplayEnd(t *testing.T, router *OpenAIRouter, ctx *RequestContext, state, reason string, status int) {
	t.Helper()
	record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, state, record.LifecycleState)
	require.Equal(t, reason, record.TerminalReason)
	require.Equal(t, status, record.ResponseStatus)
}

// A request the Router refuses after its replay record exists, here for the
// provider encoder's protocol error, ends as its client saw it: a failure
// with the 400 it got, not the client abort the end of its stream reads as.
func TestRefusedRequestReplayKeepsTheClientsStatus(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{}}
	router := startAnsweredReplay(t, ctx)
	refusal := llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "conflicting_reasoning_display",
		"reasoning display requires enabled or adaptive reasoning", nil)

	response, err := router.requestBodyReply(nil, refusal, ctx)
	require.NoError(t, err)
	require.EqualValues(t, http.StatusBadRequest, response.GetImmediateResponse().GetStatus().GetCode())
	// The ext_proc stream and the routing session both end this way after an
	// immediate response.
	require.NoError(t, router.handleProcessReceiveError(ctx, io.EOF))

	requireReplayEnd(t, router, ctx, routerreplay.LifecycleFailed, "request_rejected", http.StatusBadRequest)
	record, _ := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	require.Contains(t, record.ResponseBody, "reasoning display requires enabled or adaptive reasoning")
}

// When no backend answers, the standalone gateway writes its own local reply
// without the response phases and closes the session with it: the record
// keeps the status the client got. A client that leaves is still an abort.
func TestLocalReplyReplayKeepsTheClientsStatus(t *testing.T) {
	for name, tc := range map[string]struct {
		closeErr error
		state    string
		reason   string
		status   int
	}{
		"connection reset":   {&routing.LocalReply{Status: http.StatusServiceUnavailable, Failure: "reset"}, routerreplay.LifecycleFailed, "upstream_reset", http.StatusServiceUnavailable},
		"unclassified error": {&routing.LocalReply{Status: http.StatusServiceUnavailable}, routerreplay.LifecycleFailed, "upstream_unavailable", http.StatusServiceUnavailable},
		"client gone":        {context.Canceled, routerreplay.LifecycleAborted, "client_canceled", 0},
	} {
		t.Run(name, func(t *testing.T) {
			session := (&OpenAIRouter{}).newRoutingSession(context.Background(), nil)
			session.router = startAnsweredReplay(t, session.ctx)

			session.Close(tc.closeErr)

			requireReplayEnd(t, session.router, session.ctx, tc.state, tc.reason, tc.status)
		})
	}
}

// A record the Router already ended keeps what it holds when a reply that
// would fail it arrives later: a fast response that completed is not turned
// into a 500, and a refusal the Router recorded is not written again.
func TestAnsweredReplayLeavesAnEndedRecord(t *testing.T) {
	for name, end := range map[string]func(*OpenAIRouter, *RequestContext){
		"completed": func(router *OpenAIRouter, ctx *RequestContext) {
			router.updateRouterReplayStatus(ctx, http.StatusOK, false)
			router.attachRouterReplayResponse(ctx, []byte(`{"ok":true}`), true)
		},
		"refused": func(router *OpenAIRouter, ctx *RequestContext) {
			router.updateRouterReplayStatus(ctx, http.StatusForbidden, false)
			router.attachRouterReplayResponse(ctx, []byte(`{"denied":true}`), false)
			router.finalizeRouterReplay(ctx, routerreplay.LifecycleFailed, "decision_rejected")
		},
	} {
		t.Run(name, func(t *testing.T) {
			ctx := &RequestContext{Headers: map[string]string{}}
			router := startAnsweredReplay(t, ctx)
			end(router, ctx)
			ended, _ := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)

			_, err := router.requestBodyReply(router.createErrorResponse(http.StatusInternalServerError, "Fast response could not be encoded"), nil, ctx)
			require.NoError(t, err)

			requireReplayEnd(t, router, ctx, ended.LifecycleState, ended.TerminalReason, ended.ResponseStatus)
			record, _ := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			require.Equal(t, ended.ResponseBody, record.ResponseBody)
		})
	}
}

// A fallback chain that ends without a success answers the client itself,
// and no response phase sees that answer, so the record keeps it: a failure
// with the status the client got, in both gateways.
func TestExhaustedFallbackReplayKeepsTheClientsStatus(t *testing.T) {
	eligible := []string{"model-primary", "model-fallback-1"}
	noAnswer := routing.Outcome{
		Route: "model-primary", Status: http.StatusServiceUnavailable, Local: true,
		Failure: "connect_failure", Duration: time.Millisecond,
	}
	for name, tc := range map[string]struct {
		candidate routing.Outcome
		reason    string
		status    int
	}{
		"no backend answers": {noAnswer, "upstream_connect_failure", http.StatusServiceUnavailable},
		"the candidate answers with an error": {
			failedWith(http.StatusInternalServerError, `{"error":{"message":"boom","type":"server_error"}}`),
			"upstream_error_response", http.StatusInternalServerError,
		},
	} {
		t.Run("native gateway, "+name, func(t *testing.T) {
			session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), eligible...)
			openReplay(t, session.router, session.ctx, "model-primary", "")
			chain := session.Fallback()
			step, err := chain.Next(context.Background(), noAnswer)
			require.NoError(t, err)
			require.NotNil(t, step.Call)
			step, err = chain.Next(context.Background(), tc.candidate)
			require.NoError(t, err)
			require.NotNil(t, step.Immediate)
			require.Equal(t, tc.status, step.Immediate.Status)
			// The gateway serves the chain's answer and ends the session cleanly.
			session.Close(nil)

			requireReplayEnd(t, session.router, session.ctx, routerreplay.LifecycleFailed, tc.reason, tc.status)
		})
	}

	t.Run("response phase, no candidate backend answers", func(t *testing.T) {
		router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
		ctx := testFallbackRequestContext("model-primary", eligible)
		openReplay(t, router, ctx, "model-primary", "")
		ctx.UpstreamStatusCode = http.StatusServiceUnavailable
		router.fallbackCaller = func(context.Context, string, []byte, map[string]string) ([]byte, int, error) {
			return nil, 0, errors.New("dial tcp 127.0.0.1:8002: connect: connection refused")
		}
		resp := router.handleUpstreamTransportError([]byte(`{"error":{"message":"unavailable","type":"server_error"}}`), ctx)
		require.NotNil(t, resp.GetImmediateResponse())
		require.NoError(t, router.handleProcessReceiveError(ctx, io.EOF))

		requireReplayEnd(t, router, ctx, routerreplay.LifecycleFailed, "upstream_unavailable", http.StatusServiceUnavailable)
	})
}

// A chain that ends at a candidate refused for want of the request's
// delegated credential answers 401, after a failed candidate too, and the
// record is that refusal.
func TestRefusedFallbackCandidateReplayKeepsTheClientsStatus(t *testing.T) {
	eligible := []string{"model-primary", "model-fallback-1", "model-fallback-2"}
	delegatedLast := func(router *OpenAIRouter, ctx *RequestContext) {
		router.Config.ProviderProfiles["prof-delegated"] = config.ProviderProfile{
			Type: "anthropic", BaseURL: "http://127.0.0.1:8003", AuthCredential: config.CredentialDelegated,
		}
		router.Config.VLLMEndpoints[2].ProviderProfileName = "prof-delegated"
		ctx.TraceContext = principal.ForRequest(context.Background(), "")
		openReplay(t, router, ctx, "model-primary", "")
	}

	t.Run("native gateway", func(t *testing.T) {
		session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), eligible...)
		delegatedLast(session.router, session.ctx)
		noAnswer := routing.Outcome{
			Route: "model-primary", Status: http.StatusServiceUnavailable, Local: true,
			Failure: "connect_failure", Duration: time.Millisecond,
		}
		chain := session.Fallback()
		step, err := chain.Next(context.Background(), noAnswer)
		require.NoError(t, err)
		require.NotNil(t, step.Call)
		step, err = chain.Next(context.Background(), noAnswer)
		require.NoError(t, err)
		require.NotNil(t, step.Immediate)
		require.Equal(t, http.StatusUnauthorized, step.Immediate.Status)
		session.Close(nil)

		requireReplayEnd(t, session.router, session.ctx, routerreplay.LifecycleFailed, "request_rejected", http.StatusUnauthorized)
	})

	t.Run("response phase", func(t *testing.T) {
		router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
		ctx := testFallbackRequestContext("model-primary", eligible)
		delegatedLast(router, ctx)
		ctx.UpstreamStatusCode = http.StatusServiceUnavailable
		router.fallbackCaller = func(context.Context, string, []byte, map[string]string) ([]byte, int, error) {
			return nil, 0, errors.New("dial tcp 127.0.0.1:8002: connect: connection refused")
		}
		resp := router.handleUpstreamTransportError([]byte(`{"error":{"message":"unavailable","type":"server_error"}}`), ctx)
		require.EqualValues(t, http.StatusUnauthorized, resp.GetImmediateResponse().GetStatus().GetCode())
		require.NoError(t, router.handleProcessReceiveError(ctx, io.EOF))

		requireReplayEnd(t, router, ctx, routerreplay.LifecycleFailed, "request_rejected", http.StatusUnauthorized)
	})
}
