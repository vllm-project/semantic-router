package extproc

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
	"testing/iotest"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/parity"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

// chatChunk is one Chat Completions stream event of the upstream answer.
func chatChunk(delta, finish string) string {
	return `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"qwen36","choices":[{"index":0,"delta":` + delta + `,"finish_reason":` + finish + `}]}` + "\n\n"
}

const chatStreamError = `data: {"error":{"message":"upstream overloaded","type":"server_error","code":"overloaded"}}` + "\n\n"

// The Router's own answer to a routed request names a refusal for a 4xx and
// a failure of the Router for a 5xx. A guard overload is a refusal its client
// retries, whether its protocol carries it as 429 or 529.
func TestRoutedAnswerReplayNamesARouterFailure(t *testing.T) {
	overloaded := func(router *OpenAIRouter, ctx *RequestContext) *ext_proc.ProcessingResponse {
		ctx.VSRGuardOverload = []string{"jailbreak"}
		return router.guardOverloadResponse(ctx)
	}
	answer := func(status int, message string) func(*OpenAIRouter, *RequestContext) *ext_proc.ProcessingResponse {
		return func(router *OpenAIRouter, _ *RequestContext) *ext_proc.ProcessingResponse {
			return router.createErrorResponse(status, message)
		}
	}
	for name, tc := range map[string]struct {
		source llmprotocol.WireFormat
		answer func(*OpenAIRouter, *RequestContext) *ext_proc.ProcessingResponse
		status int
		reason string
	}{
		"rate limit":               {llmprotocol.OpenAIChatV1, answer(http.StatusTooManyRequests, "rate limit exceeded"), http.StatusTooManyRequests, "request_rejected"},
		"RAG retrieval":            {llmprotocol.OpenAIChatV1, answer(http.StatusServiceUnavailable, "RAG retrieval failed"), http.StatusServiceUnavailable, "request_processing_failed"},
		"compression":              {llmprotocol.OpenAIChatV1, answer(http.StatusInternalServerError, "Context compression failed under fail_closed policy"), http.StatusInternalServerError, "request_processing_failed"},
		"invalid cached answer":    {llmprotocol.OpenAIChatV1, answer(http.StatusBadGateway, "The cached response is invalid"), http.StatusBadGateway, "request_processing_failed"},
		"guard overload, Chat":     {llmprotocol.OpenAIChatV1, overloaded, http.StatusTooManyRequests, "request_rejected"},
		"guard overload, Messages": {llmprotocol.AnthropicMessagesV1, overloaded, anthropicOverloadedStatus, "request_rejected"},
	} {
		t.Run(name, func(t *testing.T) {
			router := &OpenAIRouter{Config: &config.RouterConfig{}}
			ctx := &RequestContext{
				Headers: map[string]string{}, RequestID: "router-answer", RequestModel: "vllm-sr/auto",
				SourceFormat: tc.source, SemanticRequest: testNeutralRequest("vllm-sr/auto", "hi"),
				VSRSelectedDecision: &config.Decision{Name: "all"},
			}
			enableReplay(router, ctx)

			_, err := router.requestBodyReply(tc.answer(router, ctx), nil, ctx)
			require.NoError(t, err)
			require.NoError(t, router.handleProcessReceiveError(ctx, io.EOF))

			requireReplayEnd(t, router, ctx, routerreplay.LifecycleFailed, tc.reason, tc.status)
		})
	}
}

// A response the Router fails to process itself, in place of what the
// upstream sent, reaches the client as its own 5xx, and the record is that
// failure of the Router, not an abort.
func TestRouterFailureInPlaceOfUpstreamReplay(t *testing.T) {
	recovery := &config.Decision{
		Name: "recoverable",
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginContext,
			Configuration: compressPayload(map[string]interface{}{
				"failure_mode": "fail_closed",
				"recovery":     map[string]interface{}{"enabled": true, "store": "redis"},
			}),
		}},
	}
	for name, tc := range map[string]struct {
		prepare func(*OpenAIRouter, *RequestContext)
		status  string
		body    string
		answer  int
	}{
		// The model asks to retrieve context under a key the request was not issued.
		"context recovery fails closed": {
			prepare: func(router *OpenAIRouter, ctx *RequestContext) {
				router.Config = &config.RouterConfig{}
				router.CompressionRecovery = &recoveryStoreStub{}
				ctx.VSRSelectedDecision = recovery
				ctx.ContextCompressionRecoveryKeys = []string{"issued-key"}
				ctx.Headers["x-authz-user-id"] = "user"
			},
			status: "200",
			body: `{"id":"initial","object":"chat.completion","created":1,"model":"model","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{` +
				`"id":"retrieve","type":"function","function":{"name":"vsr_context_retrieve","arguments":"{\"key\":\"forged-key\"}"}}]}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
			answer: http.StatusBadGateway,
		},
		"no protocol runtime for an upstream error": {
			prepare: func(router *OpenAIRouter, _ *RequestContext) {
				router.ProtocolCodecs = &protocolcodec.Registry{}
			},
			status: "500",
			body:   `{"error":{"message":"boom","type":"server_error"}}`,
			answer: http.StatusServiceUnavailable,
		},
	} {
		for gateway, deliver := range answeredUpstream {
			t.Run(name+", "+gateway, func(t *testing.T) {
				router, ctx := deliver(t, func(router *OpenAIRouter, ctx *RequestContext) {
					ctx.SourceFormat, ctx.TargetFormat = llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1
					tc.prepare(router, ctx)
				}, tc.status, "application/json", tc.body)

				requireReplayEnd(t, router, ctx, routerreplay.LifecycleFailed, "request_processing_failed", tc.answer)
			})
		}
	}
}

// A stream that carries the upstream's error event is a failure its client
// receives under the 200 the stream began with, not an abort.
func TestUpstreamStreamErrorReplayFails(t *testing.T) {
	for gateway, deliver := range answeredUpstream {
		t.Run(gateway, func(t *testing.T) {
			router, ctx := deliver(t, func(_ *OpenAIRouter, ctx *RequestContext) {
				streamingClient(ctx, llmprotocol.OpenAIChatV1)
			}, "200", "text/event-stream",
				chatChunk(`{"role":"assistant","content":"partial"}`, "null"), chatStreamError)

			requireReplayEnd(t, router, ctx, routerreplay.LifecycleFailed, "upstream_error_response", http.StatusOK)
		})
	}
}

// A stream the decision's response_guard holds reaches its client as the
// answer or as one error event in its place, and the record says which: a
// refusal of the answer, the upstream's own failure, or an answer the Router
// could not read.
func TestHeldStreamReplayKeepsWhatTheClientGot(t *testing.T) {
	limit := responseGuardHoldLimit
	t.Cleanup(func() { responseGuardHoldLimit = limit })
	answered := whole(streamOf(llmprotocol.OpenAIChatV1, benignAnswer))
	for name, tc := range map[string]struct {
		jailbreak float32
		prepare   func(*RequestContext)
		parts     [][]byte
		state     string
		reason    string
	}{
		"blocked": {jailbreak: 0.95, parts: whole(streamOf(llmprotocol.OpenAIChatV1, unsafeAnswer)), state: routerreplay.LifecycleFailed, reason: "request_rejected"},
		"over the hold limit": {
			jailbreak: 0.01, parts: answered, state: routerreplay.LifecycleFailed, reason: "request_rejected",
			prepare: func(*RequestContext) { responseGuardHoldLimit = len(answered[0]) - 1 },
		},
		"upstream error event": {
			jailbreak: 0.01, state: routerreplay.LifecycleFailed, reason: "upstream_error_response",
			parts: [][]byte{[]byte(chatChunk(`{"role":"assistant","content":"partial"}`, "null") + chatStreamError)},
		},
		// Every output item is thinking a Messages client asked to omit.
		"answer the Router cannot rebuild": {
			jailbreak: 0.01, state: routerreplay.LifecycleFailed, reason: "upstream_invalid_response",
			prepare: func(ctx *RequestContext) {
				ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
				ctx.SemanticRequest.ReasoningMode = llmprotocol.ReasoningModeAdaptive
				ctx.SemanticRequest.ReasoningDisplay = "omitted"
			},
			parts: [][]byte{[]byte(chatChunk(`{"role":"assistant","content":""}`, "null") + chatChunk(`{"reasoning":"Here"}`, `"length"`) + "data: [DONE]\n\n")},
		},
		"released": {jailbreak: 0.01, parts: answered, state: routerreplay.LifecycleCompleted, reason: "response_complete"},
	} {
		t.Run(name, func(t *testing.T) {
			responseGuardHoldLimit = limit
			router, ctx := guardedStreamingRouter(t, tc.jailbreak, config.ResponseGuardBlock, llmprotocol.OpenAIChatV1)
			startResponseStageReplay(t, router, ctx)
			streamingClient(ctx, llmprotocol.OpenAIChatV1)
			if tc.prepare != nil {
				tc.prepare(ctx)
			}

			answerHeaders(t, router, ctx, "text/event-stream")
			require.NotNil(t, ctx.ResponseGuardHold, "the guard does not hold the stream")
			answerBody(t, router, ctx, tc.parts)
			require.NoError(t, router.handleProcessReceiveError(ctx, io.EOF))

			requireReplayEnd(t, router, ctx, tc.state, tc.reason, http.StatusOK)
		})
	}
}

// A held Responses stream whose backend answered one body the Router cannot
// read gets the 502 a buffered client gets, since nothing reached the client.
func TestHeldStreamUnreadableBodyReplay(t *testing.T) {
	router, ctx := guardedStreamingRouter(t, 0.01, config.ResponseGuardBlock, llmprotocol.OpenAIResponsesV1)
	startResponseStageReplay(t, router, ctx)
	streamingClient(ctx, llmprotocol.OpenAIResponsesV1)

	answerHeaders(t, router, ctx, "application/json")
	require.NotNil(t, ctx.ResponseGuardHold, "the guard does not hold the stream")
	client, _ := answerBody(t, router, ctx, [][]byte{[]byte("this is not json")})
	require.NoError(t, router.handleProcessReceiveError(ctx, io.EOF))

	require.Contains(t, client, "The selected model returned an invalid response")
	requireReplayEnd(t, router, ctx, routerreplay.LifecycleFailed, "upstream_invalid_response", http.StatusBadGateway)
}

// A Looper hop's stream ends its record as the main stream does: complete
// when it reached its terminal event, though the Router could not rebuild
// its answer, and failed when it carried the upstream's error event.
func TestLooperStreamReplayKeepsWhatTheClientGot(t *testing.T) {
	for name, tc := range map[string]struct {
		chunks []string
		state  string
		reason string
	}{
		"complete": {
			chunks: []string{chatChunk(`{"role":"assistant","content":"ok"}`, `"stop"`), "data: [DONE]\n\n"},
			state:  routerreplay.LifecycleCompleted, reason: "response_complete",
		},
		"no answer to rebuild": {
			chunks: []string{chatChunk(`{"role":"assistant","content":""}`, "null"), chatChunk(`{}`, `"stop"`), "data: [DONE]\n\n"},
			state:  routerreplay.LifecycleCompleted, reason: "response_complete",
		},
		"upstream error event": {
			chunks: []string{chatChunk(`{"role":"assistant","content":"partial"}`, "null"), chatStreamError},
			state:  routerreplay.LifecycleFailed, reason: "upstream_error_response",
		},
	} {
		for gateway, deliver := range answeredUpstream {
			t.Run(name+", "+gateway, func(t *testing.T) {
				router, ctx := deliver(t, func(_ *OpenAIRouter, ctx *RequestContext) {
					streamingClient(ctx, llmprotocol.OpenAIChatV1)
					ctx.LooperRequest = true
				}, "200", "text/event-stream", tc.chunks...)

				requireReplayEnd(t, router, ctx, tc.state, tc.reason, http.StatusOK)
			})
		}
	}
}

// A routed request whose processing fails with an error that is no protocol
// refusal, here a backend the Router cannot resolve once it picked the model,
// leaves a record of that failure. The standalone gateway answers it with 500;
// behind Envoy the ext_proc filter's failure mode decides what the client gets.
func TestRoutedProcessingErrorReplayFails(t *testing.T) {
	_, cfg := loadParityCorpus(t)
	request := parity.Case{Name: "unresolvable-backend", Request: parity.CaseRequest{
		Method: http.MethodPost, Path: "/v1/chat/completions",
		Headers: [][]string{{"content-type", "application/json"}},
		Body:    `{"model":"general-small","messages":[{"role":"user","content":"Hello."}]}`,
	}}
	for gateway, tc := range map[string]struct {
		processor func(*OpenAIRouter) routing.Processor
		status    int
	}{
		"routing session": {func(router *OpenAIRouter) routing.Processor { return router }, http.StatusInternalServerError},
		"ext_proc stream": {func(router *OpenAIRouter) routing.Processor { return &extprocStreamProcessor{router: router} }, 0},
	} {
		t.Run(gateway, func(t *testing.T) {
			router := newParityRouter(t, cfg)
			params := router.Config.ModelConfig["general-small"]
			params.APIFormat = "no-such-wire-format"
			router.Config.ModelConfig["general-small"] = params

			run := parity.NewRecorder(tc.processor(router), routing.DefaultOptions).Run(context.Background(), request)
			require.NotEmpty(t, run.Error, "the request was not failed")

			records := router.ReplayRecorder.ListAllRecords()
			require.Len(t, records, 1, "the failed request left no replay record")
			require.Equal(t, routerreplay.LifecycleFailed, records[0].LifecycleState)
			require.Equal(t, "request_processing_failed", records[0].TerminalReason)
			require.Equal(t, tc.status, records[0].ResponseStatus)
			require.Equal(t, "general-small", records[0].SelectedModel)
		})
	}
}

// A buffered upstream body that ends before its declared length reaches the
// standalone client as the 502 of an answer the Router cannot read, and one a
// timeout stopped as the 504 Envoy gives it, with the replay id, and the
// record is that failure. Behind Envoy the processor never sees the cut: Envoy
// answers it with its own local reply, and the record ends as an abort when
// the stream is reset.
func TestCutUpstreamBodyReplay(t *testing.T) {
	_, cfg := loadParityCorpus(t)
	request := parity.Case{Name: "cut-body", Request: parity.CaseRequest{
		Method: http.MethodPost, Path: "/v1/chat/completions",
		Headers: [][]string{{"content-type", "application/json"}},
		Body:    `{"model":"general-small","messages":[{"role":"user","content":"Hello."}]}`,
	}}
	cut := func(err error) *routing.UpstreamResponse {
		return &routing.UpstreamResponse{
			Status: http.StatusOK, Header: routing.Header{{Name: "content-type", Value: "application/json"}},
			Body: io.MultiReader(strings.NewReader(`{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assi`),
				iotest.ErrReader(err)),
		}
	}
	for name, tc := range map[string]struct {
		err     error
		status  int
		reason  string
		message string
	}{
		"closed early": {io.ErrUnexpectedEOF, http.StatusBadGateway, "upstream_invalid_response", "returned an invalid response"},
		"timed out": {
			&upstream.Error{Kind: upstream.KindTimeout, Err: context.DeadlineExceeded},
			http.StatusGatewayTimeout, "upstream_timeout", "did not answer in time",
		},
	} {
		t.Run("routing session, "+name, func(t *testing.T) {
			router := newParityRouter(t, cfg)
			engine := routing.NewEngine(router, routing.DefaultOptions)
			plan, err := engine.Plan(context.Background(), request.GatewayRequest())
			require.NoError(t, err)

			response, err := engine.Respond(context.Background(), plan, cut(tc.err))
			require.NoError(t, err)
			plan.Finish(nil)

			require.Equal(t, tc.status, response.Status)
			require.Contains(t, string(response.Body), tc.message)
			records := router.ReplayRecorder.ListAllRecords()
			require.Len(t, records, 1)
			require.Equal(t, records[0].ID, response.Header.Get("x-vsr-replay-id"))
			require.Equal(t, routerreplay.LifecycleFailed, records[0].LifecycleState)
			require.Equal(t, tc.reason, records[0].TerminalReason)
			require.Equal(t, tc.status, records[0].ResponseStatus)
		})
	}
	t.Run("ext_proc stream", func(t *testing.T) {
		router := newParityRouter(t, cfg)
		engine := routing.NewEngine(&extprocStreamProcessor{router: router}, routing.DefaultOptions)
		plan, err := engine.Plan(context.Background(), request.GatewayRequest())
		require.NoError(t, err)

		_, err = engine.Respond(context.Background(), plan, cut(io.ErrUnexpectedEOF))
		require.ErrorIs(t, err, io.ErrUnexpectedEOF)
		plan.Finish(err)

		records := router.ReplayRecorder.ListAllRecords()
		require.Len(t, records, 1)
		require.Equal(t, routerreplay.LifecycleAborted, records[0].LifecycleState)
		require.Equal(t, http.StatusOK, records[0].ResponseStatus)
	})
}
