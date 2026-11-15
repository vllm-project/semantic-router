package extproc

import (
	"context"
	"io"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// startAnsweredReplay opens the replay record a routed request has once its
// decision is known, as provider dispatch does before encoding.
func startAnsweredReplay(t *testing.T, ctx *RequestContext) *OpenAIRouter {
	t.Helper()
	router := &OpenAIRouter{ReplayRecorder: routerreplay.NewRecorder(store.NewMemoryStore(10, 0))}
	replayConfig := config.DefaultRouterReplayPluginConfig()
	replayConfig.Enabled = true
	replayConfig.PromptCapture = config.PromptCaptureRaw
	ctx.RequestID = "answered-request"
	ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
	ctx.SemanticRequest = testNeutralRequest("vllm-sr/auto", "say ok")
	ctx.RouterReplayPluginConfig = &replayConfig
	ctx.VSRSelectedDecision = &config.Decision{Name: "all"}
	router.startRouterReplay(ctx, "vllm-sr/auto", "qwen-msg", "all")
	require.NotEmpty(t, ctx.RouterReplayID)
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
