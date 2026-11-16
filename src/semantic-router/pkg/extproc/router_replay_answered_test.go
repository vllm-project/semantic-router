package extproc

import (
	"context"
	"errors"
	"io"
	"net/http"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/principal"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// enableReplay turns Router Replay on for the request.
func enableReplay(router *OpenAIRouter, ctx *RequestContext) {
	router.ReplayRecorder = routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	replayConfig := config.DefaultRouterReplayPluginConfig()
	replayConfig.Enabled = true
	replayConfig.PromptCapture = config.PromptCaptureRaw
	ctx.RouterReplayPluginConfig = &replayConfig
}

// openReplay opens the replay record a routed request has once its decision
// is known, as provider dispatch does before encoding.
func openReplay(t *testing.T, router *OpenAIRouter, ctx *RequestContext, model, decision string) {
	t.Helper()
	enableReplay(router, ctx)
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

// A request the Router refuses once it has routed it, before any dispatch
// opened its record, still leaves one: here the selected model's wire cannot
// carry the request, on each routing flow and however the gateway ends it.
func TestRoutedRefusalReplayKeepsTheClientsStatus(t *testing.T) {
	flows := map[string]func(*OpenAIRouter, *llmprotocol.Request, string, *RequestContext) error{
		"entrypoint": func(router *OpenAIRouter, request *llmprotocol.Request, model string, ctx *RequestContext) error {
			_, err := router.handleEntrypointModelRouting(request, "vllm-sr/auto", "dispatch", entropy.ReasoningDecision{}, model, ctx)
			return err
		},
		"entrypoint named as its model": func(router *OpenAIRouter, request *llmprotocol.Request, model string, ctx *RequestContext) error {
			_, err := router.handleEntrypointModelRouting(request, model, "dispatch", entropy.ReasoningDecision{}, model, ctx)
			return err
		},
		"specified model": func(router *OpenAIRouter, request *llmprotocol.Request, model string, ctx *RequestContext) error {
			ctx.VSRSelectedDecision = nil
			_, err := router.handleSpecifiedModelRouting(request, model, "", ctx)
			return err
		},
	}
	ends := map[string]func(*testing.T, *OpenAIRouter, *RequestContext, error){
		"ext_proc stream": func(t *testing.T, router *OpenAIRouter, ctx *RequestContext, refusal error) {
			response, err := router.requestBodyReply(nil, refusal, ctx)
			require.NoError(t, err)
			require.EqualValues(t, http.StatusBadRequest, response.GetImmediateResponse().GetStatus().GetCode())
			// The client can look the refusal up, as any routed answer.
			require.NotEmpty(t, ctx.RouterReplayID)
			require.Equal(t, ctx.RouterReplayID, headerValueForTest(response.GetImmediateResponse().GetHeaders(), headers.RouterReplayID))
			require.NoError(t, router.handleProcessReceiveError(ctx, io.EOF))
		},
		"routing session": func(t *testing.T, router *OpenAIRouter, ctx *RequestContext, refusal error) {
			session := router.newRoutingSession(context.Background(), nil)
			session.ctx = ctx
			effect, err := session.phase(func() (*ext_proc.ProcessingResponse, error) {
				return router.requestBodyReply(nil, refusal, ctx)
			}, func(*ext_proc.ProcessingResponse) {})
			require.NoError(t, err)
			require.Equal(t, http.StatusBadRequest, effect.Immediate.Status)
			session.Close(nil)
		},
	}
	for flowName, flow := range flows {
		for endName, end := range ends {
			t.Run(flowName+", "+endName, func(t *testing.T) {
				router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
				request := testNeutralRequest(model, "draw a cat")
				request.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceImageGeneration}
				ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
				// What decision evaluation leaves for an Entrypoint request.
				ctx.VSRSelectedDecision = &config.Decision{Name: "dispatch"}
				ctx.VSRSelectedModel = model
				enableReplay(router, ctx)

				err := flow(router, request, model, ctx)
				require.Error(t, err)
				end(t, router, ctx, err)

				require.NotEmpty(t, ctx.RouterReplayID, "the refusal left no replay record")
				requireReplayEnd(t, router, ctx, routerreplay.LifecycleFailed, "request_rejected", http.StatusBadRequest)
				record, _ := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
				require.Equal(t, model, record.SelectedModel)
				require.Contains(t, record.ResponseBody, "image_generation")
			})
		}
	}
}

// Every answer the Router gives a request it routed, from any stage after the
// decision, is recorded with the status the client got. One it gives before
// routing, such as a malformed request, is not a routed request.
func TestRoutedAnswerReplayNeedsARoute(t *testing.T) {
	for name, tc := range map[string]struct {
		decision *config.Decision
		model    string
		recorded bool
	}{
		"decision":        {&config.Decision{Name: "all"}, "", true},
		"specified model": {nil, "qwen-msg", true},
		"not routed":      {nil, "", false},
	} {
		t.Run(name, func(t *testing.T) {
			router := &OpenAIRouter{}
			ctx := &RequestContext{
				Headers: map[string]string{}, RequestID: "rate-limited", RequestModel: "vllm-sr/auto",
				SourceFormat: llmprotocol.OpenAIChatV1, SemanticRequest: testNeutralRequest("vllm-sr/auto", "hi"),
				VSRSelectedDecision: tc.decision, VSRSelectedModel: tc.model,
			}
			enableReplay(router, ctx)

			_, err := router.requestBodyReply(router.createErrorResponse(http.StatusTooManyRequests, "rate limit exceeded"), nil, ctx)
			require.NoError(t, err)
			require.NoError(t, router.handleProcessReceiveError(ctx, io.EOF))

			if !tc.recorded {
				require.Empty(t, ctx.RouterReplayID)
				return
			}
			requireReplayEnd(t, router, ctx, routerreplay.LifecycleFailed, "request_rejected", http.StatusTooManyRequests)
			record, _ := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			require.Equal(t, "vllm-sr/auto", record.OriginalModel)
			require.Equal(t, tc.model, record.SelectedModel)
		})
	}
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
// would fail it arrives later: an answer that completed is not turned into a
// 500, and a refusal the Router recorded is not written again.
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

			_, err := router.requestBodyReply(router.createErrorResponse(http.StatusInternalServerError, "internal error"), nil, ctx)
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

// answeredUpstream delivers an upstream answer through the response phases
// the way each gateway does, then ends the request.
var answeredUpstream = map[string]func(t *testing.T, prepare func(*OpenAIRouter, *RequestContext), status, contentType string, chunks ...string) (*OpenAIRouter, *RequestContext){
	"ext_proc stream": func(t *testing.T, prepare func(*OpenAIRouter, *RequestContext), status, contentType string, chunks ...string) (*OpenAIRouter, *RequestContext) {
		ctx := &RequestContext{Headers: map[string]string{}}
		router := startAnsweredReplay(t, ctx)
		prepare(router, ctx)
		stream := NewMockStream(nil)
		require.NoError(t, router.processResponseHeaders(stream, nil, &ext_proc.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
				{Key: ":status", RawValue: []byte(status)}, {Key: "content-type", RawValue: []byte(contentType)},
			}}},
		}, ctx))
		for i, chunk := range chunks {
			require.NoError(t, router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{Body: []byte(chunk), EndOfStream: i == len(chunks)-1},
			}, ctx))
		}
		require.NoError(t, router.handleProcessReceiveError(ctx, io.EOF))
		return router, ctx
	},
	"routing session": func(t *testing.T, prepare func(*OpenAIRouter, *RequestContext), status, contentType string, chunks ...string) (*OpenAIRouter, *RequestContext) {
		session := (&OpenAIRouter{}).newRoutingSession(context.Background(), nil)
		session.router = startAnsweredReplay(t, session.ctx)
		prepare(session.router, session.ctx)
		_, err := session.ResponseHeaders(routing.Header{{Name: ":status", Value: status}, {Name: "content-type", Value: contentType}}, false)
		require.NoError(t, err)
		for i, chunk := range chunks {
			_, err = session.ResponseBody([]byte(chunk), i == len(chunks)-1)
			require.NoError(t, err)
		}
		session.Close(nil)
		return session.router, session.ctx
	},
}

// An upstream answer the Router cannot read reaches the client as a 502, not
// the 200 the upstream sent, and the record is that failure, not an abort.
func TestInvalidUpstreamAnswerReplayKeepsTheClientsStatus(t *testing.T) {
	for name, body := range map[string]string{
		"cut body":  `{"id":"msg_bad","type":"message","content":[`,
		"no output": `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":0}}`,
	} {
		for gateway, deliver := range answeredUpstream {
			t.Run(name+", "+gateway, func(t *testing.T) {
				router, ctx := deliver(t, func(_ *OpenAIRouter, ctx *RequestContext) {
					ctx.TargetFormat = llmprotocol.AnthropicMessagesV1
				}, "200", "application/json", body)

				requireReplayEnd(t, router, ctx, routerreplay.LifecycleFailed, "upstream_invalid_response", http.StatusBadGateway)
				record, _ := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
				require.Contains(t, record.ResponseBody, "The selected model returned an invalid response")
			})
		}
	}
}

// A stream that reached its terminal event is complete for its client even
// when the Router cannot rebuild the answer from it: here a Chat backend's
// answer that is all reasoning, for a Messages client that omits thinking.
func TestCompleteStreamReplayIsNotAnAbort(t *testing.T) {
	chunk := func(delta, finish string) string {
		return `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"qwen36","choices":[{"index":0,"delta":` + delta + `,"finish_reason":` + finish + `}]}` + "\n\n"
	}
	for gateway, deliver := range answeredUpstream {
		t.Run(gateway, func(t *testing.T) {
			router, ctx := deliver(t, func(_ *OpenAIRouter, ctx *RequestContext) {
				ctx.TargetFormat = llmprotocol.OpenAIChatV1
				ctx.ExpectStreamingResponse = true
				ctx.SemanticRequest.Stream = true
				ctx.SemanticRequest.ReasoningMode = llmprotocol.ReasoningModeAdaptive
				ctx.SemanticRequest.ReasoningDisplay = "omitted"
			}, "200", "text/event-stream",
				chunk(`{"role":"assistant","content":""}`, "null"),
				chunk(`{"reasoning":"Here"}`, `"length"`),
				"data: [DONE]\n\n")

			require.True(t, ctx.StreamingComplete)
			requireReplayEnd(t, router, ctx, routerreplay.LifecycleCompleted, "response_complete", http.StatusOK)
		})
	}
}

// A fast response the Router cannot encode reaches its client as a 500, and
// the record keeps that, not the 200 a delivered fast response has.
func TestUnencodableFastResponseReplayKeepsTheClientsStatus(t *testing.T) {
	cfg := immediateResponseTestConfig()
	classifier, err := classification.NewClassifier(cfg)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, classifier.Close()) })
	// No codec can encode the fast response.
	router := &OpenAIRouter{Config: cfg, Classifier: classifier, ProtocolCodecs: &protocolcodec.Registry{}}
	request := testNeutralRequest("vllm-sr/auto", "Synthetic routing contract.")
	ctx := &RequestContext{
		Headers: map[string]string{}, RequestID: "fast-unencodable", RequestModel: "vllm-sr/auto",
		TraceContext: context.Background(), SemanticRequest: request, SourceFormat: llmprotocol.OpenAIChatV1,
	}
	enableReplay(router, ctx)

	_, response := router.runRequestPreRoutingStages("vllm-sr/auto", request, extractSemanticRequestSignals(request), ctx)
	require.EqualValues(t, http.StatusInternalServerError, response.GetImmediateResponse().GetStatus().GetCode())
	_, err = router.requestBodyReply(response, nil, ctx)
	require.NoError(t, err)
	require.NoError(t, router.handleProcessReceiveError(ctx, io.EOF))

	requireReplayEnd(t, router, ctx, routerreplay.LifecycleFailed, "request_processing_failed", http.StatusInternalServerError)
}

// A buffered answer the decision's response_guard blocks reaches the client
// as a 403 refusal, and the record is that refusal, not the 200 the upstream
// sent followed by an abort.
func TestBlockedAnswerReplayKeepsTheClientsStatus(t *testing.T) {
	router, ctx := newResponseStageRouter(t, newJailbreakDecider(0.95), config.ResponseGuardBlock, config.ResponseGuardBlock)
	ctx.RequestID = "blocked-answer"
	ctx.SourceFormat, ctx.TargetFormat = llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1
	ctx.SemanticRequest = testNeutralRequest("vllm-sr/auto", "hi")
	openReplay(t, router, ctx, "qwen-chat", ctx.VSRSelectedDecisionName)
	_, err := router.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":status", RawValue: []byte("200")}, {Key: "content-type", RawValue: []byte("application/json")},
		}}},
	}, ctx)
	require.NoError(t, err)

	response, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: extProcResponseFixture(llmprotocol.OpenAIChatV1), EndOfStream: true},
	}, ctx)
	require.NoError(t, err)
	require.EqualValues(t, http.StatusForbidden, response.GetImmediateResponse().GetStatus().GetCode())
	require.NoError(t, router.handleProcessReceiveError(ctx, io.EOF))

	requireReplayEnd(t, router, ctx, routerreplay.LifecycleFailed, "request_rejected", http.StatusForbidden)
}
