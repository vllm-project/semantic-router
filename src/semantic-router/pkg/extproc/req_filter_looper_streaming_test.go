package extproc

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func newConfiguredFusionStreamingRouter(
	t *testing.T,
	endpoint string,
	judgeProvider string,
	judgeAPIKeyYAML string,
) (*OpenAIRouter, *routerreplay.Recorder) {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.5
providers:
  defaults:
    model: judge
  models:
    - name: panel-a
      provider_model_id: provider/panel-a
      api_format: openai
      backend_refs:
        - provider: vllm
          endpoint: ` + endpoint + `
          api_key: sk-panel-a
    - name: panel-b
      provider_model_id: provider/panel-b
      api_format: openai
      backend_refs:
        - provider: vllm
          endpoint: ` + endpoint + `
          api_key: sk-panel-b
    - name: judge
      provider_model_id: provider/judge-v2
      api_format: openai
      backend_refs:
        - provider: ` + judgeProvider + `
          endpoint: ` + endpoint + `
` + judgeAPIKeyYAML + `
    - name: backup-model
      provider_model_id: provider/backup-v2
      api_format: openai
      backend_refs:
        - provider: vllm
          endpoint: ` + endpoint + `
          api_key: sk-backup-secret
recipes:
  - name: default
`))
	require.NoError(t, err)
	enabled := true
	cfg.Looper.Streaming.Enabled = &enabled
	cfg.ClearRouteCache = true
	cfg.Looper.Endpoint = endpoint

	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	router := &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: authz.NewCredentialResolver(authz.NewStaticConfigProvider(cfg)),
		ReplayRecorder:     recorder,
	}
	return router, recorder
}

func TestLooperStreamingDispatch_NativeContinue(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		writeFusionReplayCompletion(w, request.Model, []map[string]interface{}{{
			"index": 0,
			"message": map[string]interface{}{
				"role":    "assistant",
				"content": "Panel answer from " + request.Model,
			},
			"finish_reason": "stop",
		}}, map[string]int64{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
	}))
	defer server.Close()

	for _, tc := range []struct {
		name              string
		setupRouter       func() (*OpenAIRouter, *RequestContext, *llmprotocol.Request)
		wantBodyModel     string
		wantAuthHeader    string
		wantUpstreamRoute string
	}{
		{
			name: "configured backend_refs primary path",
			setupRouter: func() (*OpenAIRouter, *RequestContext, *llmprotocol.Request) {
				router, recorder := newConfiguredFusionStreamingRouter(t, server.URL, "vllm", "          api_key: sk-judge-secret")
				replayConfig := config.DefaultRouterReplayPluginConfig()
				req := testNeutralRequest("router-entrypoint", "hello")
				req.Stream = true
				ctx := &RequestContext{
					RequestID:                "req-native-primary",
					SourceFormat:             llmprotocol.OpenAIChatV1,
					SemanticRequest:          req,
					ExpectStreamingResponse:  true,
					RouterReplayRecorder:     recorder,
					RouterReplayPluginConfig: &replayConfig,
				}
				return router, ctx, req
			},
			wantBodyModel:     "provider/judge-v2",
			wantAuthHeader:    "Bearer sk-judge-secret",
			wantUpstreamRoute: "judge",
		},
		{
			name: "legacy looper endpoint fixture path",
			setupRouter: func() (*OpenAIRouter, *RequestContext, *llmprotocol.Request) {
				router, ctx, req := quorumFallbackRouter(server.URL)
				enabled := true
				router.Config.Looper.Streaming.Enabled = &enabled
				router.Config.ClearRouteCache = true
				ctx.ExpectStreamingResponse = true
				req.Stream = true
				return router, ctx, req
			},
			wantBodyModel:     "judge",
			wantUpstreamRoute: "judge",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router, reqCtx, request := tc.setupRouter()
			decision := &config.Decision{
				Name: "fusion-stream",
				Algorithm: &config.AlgorithmConfig{
					Type: config.DecisionAlgorithmFusion,
					Fusion: &config.FusionAlgorithmConfig{
						Model:          "judge",
						AnalysisModels: []string{"panel-a", "panel-b"},
						AnalysisMode:   config.FusionAnalysisModeOneCall,
					},
				},
			}

			response, err := router.handleLooperExecution(context.Background(), request, decision, reqCtx)
			require.NoError(t, err)
			require.NotNil(t, response)

			// In Native Streaming, RequestBody returns a CONTINUE response (not ImmediateResponse)
			require.Nil(t, response.GetImmediateResponse())
			reqBodyResp := response.GetRequestBody()
			require.NotNil(t, reqBodyResp)
			common := reqBodyResp.GetResponse()
			require.NotNil(t, common)
			require.Equal(t, ext_proc.CommonResponse_CONTINUE, common.GetStatus())
			require.True(t, common.GetClearRouteCache(), "ClearRouteCache must be true to re-match cluster in Envoy")

			setHeaders := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())
			require.Equal(t, tc.wantUpstreamRoute, setHeaders[headers.VSRRouteModel])
			if tc.wantAuthHeader != "" {
				require.Equal(t, tc.wantAuthHeader, setHeaders["Authorization"])
			}

			// Verify request body mutation carries the synthesis prompt and stream=true
			require.NotNil(t, common.GetBodyMutation())
			mutatedBody := common.GetBodyMutation().GetBody()
			require.NotEmpty(t, mutatedBody)

			var parsedReq map[string]interface{}
			require.NoError(t, json.Unmarshal(mutatedBody, &parsedReq))
			require.Equal(t, true, parsedReq["stream"])
			require.Equal(t, tc.wantBodyModel, parsedReq["model"])
			streamOpts, ok := parsedReq["stream_options"].(map[string]interface{})
			require.True(t, ok, "encodeDispatchRequest must inject stream_options onto synthesis dispatch")
			require.Equal(t, true, streamOpts["include_usage"])

			// Verify context state
			require.True(t, reqCtx.LooperStreamingActive)
			require.Equal(t, "judge", reqCtx.VSRSelectedModel)
			require.Equal(t, "judge", reqCtx.LooperTargetModel)
			require.Equal(t, "router-entrypoint", reqCtx.RequestModel)
			require.Equal(t, int64(20), reqCtx.LooperIntermediateUsage.PromptTokens) // 2 panel models * 10
			require.Equal(t, int64(10), reqCtx.LooperIntermediateUsage.CompletionTokens)
			require.Equal(t, int64(30), reqCtx.LooperIntermediateUsage.TotalTokens)
			require.Equal(t, []string{"panel-a", "panel-b", "judge"}, reqCtx.LooperModelsUsed)
			require.Equal(t, 3, reqCtx.LooperIterations)
			require.NotEmpty(t, reqCtx.RouterReplayID)
		})
	}
}

func TestLooperStreamingDispatch_QuorumFallbackStreams(t *testing.T) {
	// Panel server where panel-a and panel-b fail (500)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	router, recorder := newConfiguredFusionStreamingRouter(t, server.URL, "vllm", "          api_key: sk-judge-secret")
	replayConfig := config.DefaultRouterReplayPluginConfig()
	request := testNeutralRequest("router-entrypoint", "preserve client user prompt")
	request.Stream = true
	reqCtx := &RequestContext{
		RequestID:                "req-quorum-fallback-stream",
		SourceFormat:             llmprotocol.OpenAIChatV1,
		SemanticRequest:          request,
		ExpectStreamingResponse:  true,
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
	}

	decision := &config.Decision{
		Name: "fusion-quorum-fallback",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model:                "judge",
				AnalysisModels:       []string{"panel-a", "panel-b"},
				AnalysisMode:         config.FusionAnalysisModeOneCall,
				QuorumFailurePolicy:  config.FusionQuorumFailurePolicyFallback,
				QuorumFallbackTarget: "backup-model",
			},
		},
	}

	response, err := router.handleLooperExecution(context.Background(), request, decision, reqCtx)
	require.NoError(t, err)
	require.NotNil(t, response)

	// Falls back to direct streaming to backup-model via primary backend_refs dispatch
	require.Nil(t, response.GetImmediateResponse())
	reqBodyResp := response.GetRequestBody()
	require.NotNil(t, reqBodyResp)
	common := reqBodyResp.GetResponse()
	require.Equal(t, ext_proc.CommonResponse_CONTINUE, common.GetStatus())

	setHeaders := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())
	require.Equal(t, "backup-model", setHeaders[headers.VSRRouteModel])
	require.Equal(t, "Bearer sk-backup-secret", setHeaders["Authorization"])

	var fallbackPayload map[string]interface{}
	require.NoError(t, json.Unmarshal(common.GetBodyMutation().GetBody(), &fallbackPayload))
	require.Equal(t, "provider/backup-v2", fallbackPayload["model"])
	require.Equal(t, true, fallbackPayload["stream"])
	fallbackBytes, _ := json.Marshal(fallbackPayload["messages"])
	require.Contains(t, string(fallbackBytes), "preserve client user prompt")

	require.Equal(t, "backup-model", reqCtx.VSRSelectedModel)
	require.True(t, reqCtx.LooperStreamingActive)
	require.Nil(t, reqCtx.LooperExecutionTrace)
	require.Equal(t, []string{"panel-a", "panel-b", "backup-model"}, reqCtx.LooperModelsUsed)
	require.Equal(t, 3, reqCtx.LooperIterations)
	require.NotNil(t, reqCtx.VSRFusionQuorum)
	require.Equal(t, string(looper.FusionQuorumFallbackServed), reqCtx.VSRFusionQuorum.Disposition)
	require.Equal(t, "backup-model", reqCtx.VSRFusionQuorum.FallbackTarget)
}

func TestLooperStreamingDispatch_FailFastOnQuorumFailure(t *testing.T) {
	// Server where panel fails with 500 and policy is abort
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	router, reqCtx, request := quorumFallbackRouter(server.URL)
	enabled := true
	router.Config.Looper.Streaming.Enabled = &enabled
	reqCtx.ExpectStreamingResponse = true
	request.Stream = true

	decision := &config.Decision{
		Name: "fusion-quorum-abort",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model:               "judge",
				AnalysisModels:      []string{"panel-a", "panel-b"},
				QuorumFailurePolicy: config.FusionQuorumFailurePolicyFail,
			},
		},
	}

	response, err := router.handleLooperExecution(context.Background(), request, decision, reqCtx)
	require.NoError(t, err)
	require.NotNil(t, response)

	// Must return ImmediateResponse (502) and NOT forward to Envoy
	require.NotNil(t, response.GetImmediateResponse())
	require.EqualValues(t, 502, response.GetImmediateResponse().GetStatus().GetCode())
	require.Nil(t, response.GetRequestBody())
	require.False(t, reqCtx.LooperStreamingActive)

	// Verify replay failure recorded
	record, ok := router.ReplayRecorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
}

func TestLooperStreamingResponse_AuthoritativeUsageSettlement(t *testing.T) {
	router := &OpenAIRouter{}
	reqCtx := &RequestContext{
		LooperStreamingActive: true,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     25,
			CompletionTokens: 15,
			TotalTokens:      40,
		},
		SemanticResponse: &llmprotocol.Response{
			Usage: llmprotocol.Usage{
				State: llmprotocol.UsageAvailable,
				InputTotal: llmprotocol.TokenCount{
					Value:      replayInt64Ptr(10),
					Provenance: llmprotocol.UsageAuthoritative,
				},
				OutputTotal: llmprotocol.TokenCount{
					Value:      replayInt64Ptr(5),
					Provenance: llmprotocol.UsageAuthoritative,
				},
				Total: llmprotocol.TokenCount{
					Value:      replayInt64Ptr(15),
					Provenance: llmprotocol.UsageAuthoritative,
				},
			},
		},
	}

	usage := router.takeNeutralResponseUsage(reqCtx)
	require.False(t, usage.invalid)
	// Authoritative settlement combines intermediate panel tokens + synthesis upstream tokens
	require.Equal(t, 35, usage.promptTokens)     // 25 + 10
	require.Equal(t, 20, usage.completionTokens) // 15 + 5
	require.Equal(t, 55, usage.totalTokens)      // 40 + 15
}

func TestLooperStreamingResponse_ClientCancelledEstimatedSettlement(t *testing.T) {
	coreObs, observedLogs := observer.New(zapcore.InfoLevel)
	restoreLogger := zap.ReplaceGlobals(zap.New(coreObs))
	t.Cleanup(restoreLogger)

	memStore := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(memStore)
	replayConfig := config.DefaultRouterReplayPluginConfig()
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			RouterReplay: config.RouterReplayConfig{},
		},
		ReplayRecorder: recorder,
	}

	reqCtx := &RequestContext{
		RequestID:                 "req-stream-cancel",
		RouterReplayRecorder:      recorder,
		RouterReplayPluginConfig:  &replayConfig,
		IsStreamingResponse:       true,
		StreamingComplete:         false,
		LooperStreamingActive:     true,
		LooperAuditedOutputTokens: 8,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     20,
			CompletionTokens: 10,
			TotalTokens:      30,
		},
	}

	router.startRouterReplay(reqCtx, "client-model", "judge", "fusion-stream")
	require.NotEmpty(t, reqCtx.RouterReplayID)

	// Trigger client cancel error in handleProcessReceiveError
	receiveErr := context.Canceled
	retErr := router.handleProcessReceiveError(reqCtx, receiveErr)
	require.NoError(t, retErr)
	require.True(t, reqCtx.StreamingAborted)
	require.True(t, reqCtx.LooperUsageSettled)

	// Verify replay record has estimated settlement without fabricated aggregate cost
	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState)
	require.Equal(t, "client_canceled", record.TerminalReason)
	require.NotNil(t, record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.NotNil(t, record.TotalTokens)
	require.Equal(t, 20, *record.PromptTokens)
	require.Equal(t, 18, *record.CompletionTokens) // 10 (intermediate) + 8 (audited deltas)
	require.Equal(t, 38, *record.TotalTokens)
	require.Nil(t, record.ActualCost)
	require.Nil(t, record.BaselineCost)
	require.Nil(t, record.CostSavings)

	// Verify structured llm_usage audit log explicitly marks usage as estimated and unpriced
	var usageLogs []observer.LoggedEntry
	for _, entry := range observedLogs.All() {
		if entry.ContextMap()["event"] == "llm_usage" {
			usageLogs = append(usageLogs, entry)
		}
	}
	require.Len(t, usageLogs, 1)
	fields := usageLogs[0].ContextMap()
	require.Equal(t, "req-stream-cancel", fields["request_id"])
	require.Equal(t, "estimated", fields["usage_provenance"])
	require.Equal(t, true, fields["stream_aborted"])
	require.Equal(t, "response_stream", fields["abort_stage"])
	require.Equal(t, "client_canceled", fields["abort_reason"])
	require.Equal(t, int64(20), fields["prompt_tokens"])
	require.Equal(t, int64(20), fields["intermediate_prompt_tokens"])
	require.Equal(t, int64(10), fields["intermediate_completion_tokens"])
	require.Equal(t, int64(8), fields["audited_stream_output_tokens"])
	require.Equal(t, int64(18), fields["completion_tokens"])
	require.Equal(t, int64(38), fields["total_tokens"])
	require.Equal(t, "looper_aborted_unpriced", fields["pricing"])
	require.Equal(t, float64(0), fields["cost"])
	require.Equal(t, "unknown", fields["currency"])
}

func TestLooperStreamingPipeline_EndToEndIncludeUsageAuthAndModelMapping(t *testing.T) {
	panelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		writeFusionReplayCompletion(w, request.Model, []map[string]interface{}{{
			"index": 0,
			"message": map[string]interface{}{
				"role":    "assistant",
				"content": "Panel insight from " + request.Model,
			},
			"finish_reason": "stop",
		}}, map[string]int64{"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18})
	}))
	defer panelServer.Close()

	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.5
providers:
  defaults:
    model: judge
  models:
    - name: panel-a
      provider_model_id: provider/panel-a
      api_format: openai
      backend_refs:
        - provider: vllm
          endpoint: ` + panelServer.URL + `
          api_key: sk-panel-a
    - name: panel-b
      provider_model_id: provider/panel-b
      api_format: openai
      backend_refs:
        - provider: vllm
          endpoint: ` + panelServer.URL + `
          api_key: sk-panel-b
    - name: judge
      provider_model_id: provider/judge-v2
      api_format: openai
      backend_refs:
        - provider: vllm
          endpoint: ` + panelServer.URL + `
          api_key: sk-judge-secret
recipes:
  - name: default
`))
	require.NoError(t, err)
	enabled := true
	cfg.Looper.Streaming.Enabled = &enabled
	cfg.ClearRouteCache = true
	cfg.Looper.Endpoint = panelServer.URL

	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	replayConfig := config.DefaultRouterReplayPluginConfig()
	replayConfig.PromptCapture = config.PromptCaptureRaw
	router := &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: authz.NewCredentialResolver(authz.NewStaticConfigProvider(cfg)),
		ReplayRecorder:     recorder,
	}

	// Client does NOT pass stream_options.include_usage; router must inject it upstream.
	request := testNeutralRequest("router-entrypoint", "Compare distributed consensus protocols")
	request.Stream = true
	reqCtx := &RequestContext{
		RequestID:                "req-e2e-looper-stream",
		Headers:                  map[string]string{headers.VSRDebug: "true"},
		SourceFormat:             llmprotocol.OpenAIChatV1,
		SemanticRequest:          request,
		ExpectStreamingResponse:  true,
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
	}

	decision := &config.Decision{
		Name: "fusion-e2e-stream",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model:          "judge",
				AnalysisModels: []string{"panel-a", "panel-b"},
				AnalysisMode:   config.FusionAnalysisModeNone,
			},
		},
	}
	reqCtx.VSRSelectedDecision = decision
	reqCtx.VSRSelectedDecisionName = decision.Name

	// Phase 1: RequestBody executes intermediate Panel and returns CONTINUE for Synthesis
	reqResp, err := router.handleLooperExecution(context.Background(), request, decision, reqCtx)
	require.NoError(t, err)
	require.Nil(t, reqResp.GetImmediateResponse())
	common := reqResp.GetRequestBody().GetResponse()
	require.Equal(t, ext_proc.CommonResponse_CONTINUE, common.GetStatus())
	require.True(t, common.GetClearRouteCache())

	setHeaders := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())
	require.Equal(t, "judge", setHeaders[headers.VSRRouteModel])
	require.Equal(t, "Bearer sk-judge-secret", setHeaders["Authorization"])
	require.Equal(t, "/v1/chat/completions", setHeaders[":path"])

	var dispatchedPayload map[string]interface{}
	require.NoError(t, json.Unmarshal(common.GetBodyMutation().GetBody(), &dispatchedPayload))
	require.Equal(t, "provider/judge-v2", dispatchedPayload["model"], "external provider_model_id mapping must be applied")
	require.Equal(t, true, dispatchedPayload["stream"])
	streamOpts, ok := dispatchedPayload["stream_options"].(map[string]interface{})
	require.True(t, ok, "stream_options must be injected by encodeDispatchRequest")
	require.Equal(t, true, streamOpts["include_usage"])
	// Original client preference must stay false/unset so public SSE filter strips usage frame
	require.False(t, streamUsageRequestedByClient(reqCtx))

	// Phase 2: Upstream ResponseHeaders arrives with 200 text/event-stream
	resHeadersReq := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "200"},
					{Key: "content-type", Value: "text/event-stream"},
					{Key: "content-length", Value: "999"},
				},
			},
		},
	}
	resHeadersResp, err := router.handleResponseHeaders(resHeadersReq, reqCtx)
	require.NoError(t, err)
	require.Equal(t, http_ext.ProcessingMode_STREAMED, resHeadersResp.GetModeOverride().GetResponseBodyMode())
	resHeaderMut := resHeadersResp.GetResponseHeaders().GetResponse().GetHeaderMutation()
	require.Contains(t, resHeaderMut.GetRemoveHeaders(), "content-length")
	resHeaderMap := headerValuesByName(resHeaderMut.GetSetHeaders())
	require.Equal(t, headers.ResponsePathLooper, resHeaderMap[headers.VSRResponsePath])
	looperTraceMember := traceMembersOf(t, resHeaderMap[headers.VSRTrace])[headers.TraceLooper]
	require.Contains(t, looperTraceMember, `"panel-a" "panel-b" "judge"`)
	require.Contains(t, looperTraceMember, `algorithm=fusion`)

	// Phase 3: Upstream streams SSE chunks ending with include_usage terminal chunk + [DONE]
	sseChunks := []string{
		"data: {\"id\":\"chatcmpl-synth-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"provider/judge-v2\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Raft and \"},\"finish_reason\":null}]}\n\n",
		"data: {\"id\":\"chatcmpl-synth-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"provider/judge-v2\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Paxos converge.\"},\"finish_reason\":\"stop\"}]}\n\n",
		"data: {\"id\":\"chatcmpl-synth-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"provider/judge-v2\",\"choices\":[],\"usage\":{\"prompt_tokens\":18,\"completion_tokens\":7,\"total_tokens\":25}}\n\n" +
			"data: [DONE]\n\n",
	}

	var clientVisibleSSE strings.Builder
	for idx, chunk := range sseChunks {
		bodyResp, bodyErr := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{
				Body:        []byte(chunk),
				EndOfStream: idx == len(sseChunks)-1,
			},
		}, reqCtx)
		require.NoError(t, bodyErr)
		if mut := bodyResp.GetResponseBody().GetResponse().GetBodyMutation(); mut != nil {
			clientVisibleSSE.Write(mut.GetBody())
		}
	}

	// Client stream gets virtualized public model and stripped forced usage frame
	renderedSSE := clientVisibleSSE.String()
	require.Contains(t, renderedSSE, `"model":"router-entrypoint"`)
	require.Contains(t, renderedSSE, `"Raft and "`)
	require.Contains(t, renderedSSE, `"Paxos converge."`)
	require.Contains(t, renderedSSE, `"finish_reason":"stop"`)
	require.Contains(t, renderedSSE, "data: [DONE]")
	require.NotContains(t, renderedSSE, `"provider/judge-v2"`)
	require.NotContains(t, renderedSSE, `"prompt_tokens":18`)

	// Verify takeNeutralResponseUsage combined Panel (24 prompt + 12 completion = 36) + Synthesis (18 prompt + 7 completion = 25)
	settledUsage := router.takeNeutralResponseUsage(reqCtx)
	require.False(t, settledUsage.invalid)
	require.Equal(t, 42, settledUsage.promptTokens)     // 24 + 18
	require.Equal(t, 19, settledUsage.completionTokens) // 12 + 7
	require.Equal(t, 61, settledUsage.totalTokens)      // 36 + 25

	// Verify Replay record persisted authoritative total, structured Fusion trace, and unpriced cascade cost
	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.NotNil(t, record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.NotNil(t, record.TotalTokens)
	require.Equal(t, 42, *record.PromptTokens)
	require.Equal(t, 19, *record.CompletionTokens)
	require.Equal(t, 61, *record.TotalTokens)
	require.Nil(t, record.ActualCost)
	require.Nil(t, record.BaselineCost)
	require.Nil(t, record.CostSavings)
	var replayBody map[string]interface{}
	require.NoError(t, json.Unmarshal([]byte(record.ResponseBody), &replayBody))
	require.Contains(t, replayBody, "choices")
	require.Contains(t, replayBody, "fusion")
}

func TestLooperStreamingDisconnect_PrefillBeforeResponseHeadersPersistsAuditAndReplay(t *testing.T) {
	coreObs, observedLogs := observer.New(zapcore.InfoLevel)
	restoreLogger := zap.ReplaceGlobals(zap.New(coreObs))
	t.Cleanup(restoreLogger)

	memStore := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(memStore)
	replayConfig := config.DefaultRouterReplayPluginConfig()
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			RouterReplay: config.RouterReplayConfig{},
		},
		ReplayRecorder: recorder,
	}

	// Simulate client disconnecting during upstream prefill/queueing:
	// RequestBody returned CONTINUE (LooperStreamingActive == true), but
	// upstream ResponseHeaders has NOT arrived yet (IsStreamingResponse == false).
	reqCtx := &RequestContext{
		RequestID:                "req-prefill-disconnect",
		RequestModel:             "router-entrypoint",
		VSRSelectedModel:         "judge",
		LooperTargetModel:        "judge",
		VSRSelectedDecisionName:  "fusion-stream",
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
		IsStreamingResponse:      false,
		StreamingComplete:        false,
		LooperStreamingActive:    true,
		LooperModelsUsed:         []string{"panel-a", "panel-b"},
		LooperIterations:         2,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     34,
			CompletionTokens: 16,
			TotalTokens:      50,
		},
	}

	router.startRouterReplay(reqCtx, "router-entrypoint", "judge", "fusion-stream")
	require.NotEmpty(t, reqCtx.RouterReplayID)

	retErr := router.handleProcessReceiveError(reqCtx, context.Canceled)
	require.NoError(t, retErr)
	require.True(t, reqCtx.StreamingAborted)
	require.True(t, reqCtx.LooperUsageSettled)

	// Calling handleProcessReceiveError a second time must be idempotent
	require.NoError(t, router.handleProcessReceiveError(reqCtx, context.Canceled))

	// 1. Verify Replay record preserved intermediate tokens
	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState)
	require.Equal(t, "client_canceled", record.TerminalReason)
	require.NotNil(t, record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.NotNil(t, record.TotalTokens)
	require.Equal(t, 34, *record.PromptTokens)
	require.Equal(t, 16, *record.CompletionTokens)
	require.Equal(t, 50, *record.TotalTokens)

	// 2. Verify structured llm_usage audit log emitted exactly once with prefill stage
	var usageLogs []observer.LoggedEntry
	for _, entry := range observedLogs.All() {
		if entry.ContextMap()["event"] == "llm_usage" {
			usageLogs = append(usageLogs, entry)
		}
	}
	require.Len(t, usageLogs, 1)
	fields := usageLogs[0].ContextMap()
	require.Equal(t, "req-prefill-disconnect", fields["request_id"])
	require.Equal(t, "judge", fields["target_model"])
	require.Equal(t, "upstream_prefill", fields["abort_stage"])
	require.Equal(t, int64(34), fields["prompt_tokens"])
	require.Equal(t, int64(16), fields["completion_tokens"])
	require.Equal(t, int64(50), fields["total_tokens"])
}

func TestLooperStreamingTokenCalibration_UsesPhysicalTargetModelNeverAlias(t *testing.T) {
	classifier, err := classification.BuildClassifier(&config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				ContextRules: []config.ContextRule{{
					Name:      "window",
					MinTokens: config.TokenCount("100K"),
				}},
			},
		},
	})
	require.NoError(t, err)

	router := &OpenAIRouter{
		Config:     &config.RouterConfig{},
		Classifier: classifier,
	}

	const synthesisBytes = 8000
	const synthesisPromptTokens = 2000 // exact 4.0 bytes/token for physical model "judge"

	for i := 0; i < 12; i++ {
		streamState := &semanticResponseStreamState{
			responseID: "chatcmpl-cal",
			model:      "judge",
			stop:       llmprotocol.StopEndTurn,
			terminal:   true,
			items: map[int]*semanticStreamItem{
				0: {role: llmprotocol.RoleAssistant, text: "synthesized", completed: true},
			},
			order: []int{0},
			usage: llmprotocol.Usage{
				State: llmprotocol.UsageAvailable,
				InputTotal: llmprotocol.TokenCount{
					Value:      replayInt64Ptr(synthesisPromptTokens),
					Provenance: llmprotocol.UsageAuthoritative,
				},
				OutputTotal: llmprotocol.TokenCount{
					Value:      replayInt64Ptr(50),
					Provenance: llmprotocol.UsageAuthoritative,
				},
			},
		}
		ctx := &RequestContext{
			RequestID:                "req-cal",
			SourceFormat:             llmprotocol.OpenAIChatV1,
			RequestModel:             "auto", // abstract client alias
			VSRSelectedModel:         "judge",
			LooperStreamingActive:    true,
			LooperTargetModel:        "judge",
			LooperSynthesisTextBytes: synthesisBytes,
			// Large intermediate panel tokens must NOT poison judge's ratio
			LooperIntermediateUsage: looper.TokenUsage{
				PromptTokens:     20000,
				CompletionTokens: 5000,
				TotalTokens:      25000,
			},
			SemanticStreamState: streamState,
		}
		ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})
		router.finalizeSemanticStreamingResponse(ctx, nil)
	}

	judgeMean, _, judgeSamples, judgeCalibrated := classifier.TokenCalibrationRatio("model:judge")
	require.True(t, judgeCalibrated)
	require.Equal(t, int64(12), judgeSamples)
	require.InDelta(t, 4.0, judgeMean, 0.05, "calibration must use synthesis-only prompt tokens and bytes")

	_, _, autoSamples, _ := classifier.TokenCalibrationRatio("model:auto")
	require.Zero(t, autoSamples, "abstract client alias 'auto' must never be calibrated")
}

func TestObserveProtocolStream_CJKAbortAccumulatesEstimatedTokens(t *testing.T) {
	cjkSentence := "这是一段用于测试流式中断计费保护的中文输出内容，包含多个分句。"
	expectedTokens := contextcompression.EstimateTokens(cjkSentence)
	require.Greater(t, expectedTokens, 1, "space-free Chinese sentence must not collapse to 1 token")

	memStore := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(memStore)
	replayConfig := config.DefaultRouterReplayPluginConfig()
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			RouterReplay: config.RouterReplayConfig{},
		},
		ReplayRecorder: recorder,
	}

	reqCtx := &RequestContext{
		RequestID:                "req-cjk-stream-abort",
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
		IsStreamingResponse:      true,
		LooperStreamingActive:    true,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     20,
			CompletionTokens: 10,
			TotalTokens:      30,
		},
		SemanticStreamState: &semanticResponseStreamState{
			items: make(map[int]*semanticStreamItem),
		},
	}
	router.startRouterReplay(reqCtx, "router-entrypoint", "judge", "fusion-stream")

	observeProtocolStream(reqCtx, []llmprotocol.Event{
		{Type: llmprotocol.EventOutputTextDelta, Delta: cjkSentence},
	}, nil)

	require.NoError(t, router.handleProcessReceiveError(reqCtx, context.Canceled))
	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.NotNil(t, record.CompletionTokens)
	require.Equal(t, 10+expectedTokens, *record.CompletionTokens)
}

func TestSettleAbortedLooperStreamingUsage_IncludesToolCallsAndAvoidsChunkInflation(t *testing.T) {
	memStore := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(memStore)
	replayConfig := config.DefaultRouterReplayPluginConfig()
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			RouterReplay: config.RouterReplayConfig{},
		},
		ReplayRecorder: recorder,
	}

	toolName := "get_weather"
	toolArgs := `{"location": "Hangzhou, Zhejiang", "unit": "celsius"}`
	combinedText := toolName + toolArgs
	expectedTokens := contextcompression.EstimateTokens(combinedText)

	reqCtx := &RequestContext{
		RequestID:                "req-tool-stream-abort",
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
		IsStreamingResponse:      true,
		LooperStreamingActive:    true,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     20,
			CompletionTokens: 5,
			TotalTokens:      25,
		},
		SemanticStreamState: &semanticResponseStreamState{
			items: make(map[int]*semanticStreamItem),
		},
	}
	router.startRouterReplay(reqCtx, "router-entrypoint", "judge", "fusion-stream")

	// Emit tool call across multiple small delta chunks
	observeProtocolStream(reqCtx, []llmprotocol.Event{
		{
			Type:      llmprotocol.EventToolCallDelta,
			ItemIndex: 0,
			ToolCall:  &llmprotocol.ToolCall{Name: toolName},
		},
		{
			Type:      llmprotocol.EventToolCallDelta,
			ItemIndex: 0,
			ToolCall:  &llmprotocol.ToolCall{Arguments: `{"location":`},
		},
		{
			Type:      llmprotocol.EventToolCallDelta,
			ItemIndex: 0,
			ToolCall:  &llmprotocol.ToolCall{Arguments: ` "Hangzhou, Zhejiang",`},
		},
		{
			Type:      llmprotocol.EventToolCallDelta,
			ItemIndex: 0,
			ToolCall:  &llmprotocol.ToolCall{Arguments: ` "unit": "celsius"}`},
		},
	}, nil)

	require.NoError(t, router.handleProcessReceiveError(reqCtx, context.Canceled))
	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.NotNil(t, record.CompletionTokens)
	require.Equal(t, 5+expectedTokens, *record.CompletionTokens)
}

func replayInt64Ptr(v int64) *int64 {
	return &v
}

func TestBuildLooperStreamingDispatchResponse_PreservesClientSemanticRequestContext(t *testing.T) {
	for _, tc := range []struct {
		name        string
		buildRouter func() *OpenAIRouter
	}{
		{
			name: "configured backend_refs primary path",
			buildRouter: func() *OpenAIRouter {
				router, _ := newConfiguredFusionStreamingRouter(t, "http://127.0.0.1:18080", "vllm", "          api_key: sk-judge-secret")
				return router
			},
		},
		{
			name: "legacy looper endpoint path",
			buildRouter: func() *OpenAIRouter {
				return &OpenAIRouter{
					Config: &config.RouterConfig{
						Looper: config.LooperConfig{Endpoint: "http://127.0.0.1:18080"},
						BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
							"judge": {Capabilities: []string{"chat"}},
						}},
					},
				}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := tc.buildRouter()
			originalSemanticReq := &llmprotocol.Request{
				Model: "original-user-model",
				Messages: []llmprotocol.Message{
					{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "client prompt"}}},
				},
			}
			originalEnvelope := llmprotocol.Envelope{Request: []byte(`{"client":"raw"}`)}

			reqCtx := &RequestContext{
				RequestID:        "req-anti-pollution",
				SourceFormat:     llmprotocol.OpenAIChatV1,
				SemanticRequest:  originalSemanticReq,
				ProtocolEnvelope: originalEnvelope,
			}
			decision := &config.Decision{Name: "test-decision"}

			synthesisBody := []byte(`{"model":"judge","messages":[{"role":"system","content":"synthesis prompt"}]}`)
			resp, textBytes, err := router.buildLooperStreamingDispatchResponse(synthesisBody, "judge", decision, reqCtx)
			require.NoError(t, err)
			require.NotNil(t, resp)
			require.Nil(t, resp.GetImmediateResponse())
			require.Equal(t, ext_proc.CommonResponse_CONTINUE, resp.GetRequestBody().GetResponse().GetStatus())
			require.Positive(t, textBytes)

			require.Same(t, originalSemanticReq, reqCtx.SemanticRequest, "reqCtx.SemanticRequest must be restored to client's original request")
			require.Equal(t, "original-user-model", reqCtx.SemanticRequest.Model)
			require.False(t, reqCtx.SemanticRequest.Stream)
			require.Zero(t, reqCtx.SemanticRequest.Generation)
			require.Len(t, reqCtx.SemanticRequest.Messages, 1)
			require.Equal(t, "client prompt", reqCtx.SemanticRequest.Messages[0].Content[0].Text)
			require.Equal(t, originalEnvelope, reqCtx.ProtocolEnvelope, "reqCtx.ProtocolEnvelope must be restored to client's original envelope")
		})
	}
}

func TestHandleLooperStreamingDispatch_ImmediateResponseRecordsIntermediateUsage(t *testing.T) {
	panelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		writeFusionReplayCompletion(w, request.Model, []map[string]interface{}{{
			"index":         0,
			"message":       map[string]interface{}{"role": "assistant", "content": "panel ok"},
			"finish_reason": "stop",
		}}, map[string]int64{"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18})
	}))
	defer panelServer.Close()

	// judge uses provider "openai" with no API key, triggering a 401 ImmediateResponse during dispatch
	router, recorder := newConfiguredFusionStreamingRouter(t, panelServer.URL, "openai", "")
	replayConfig := config.DefaultRouterReplayPluginConfig()
	request := testNeutralRequest("entrypoint-model", "test immediate dispatch failure")
	request.Stream = true
	reqCtx := &RequestContext{
		RequestID:                "req-immediate-fail",
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
		ExpectStreamingResponse:  true,
		SourceFormat:             llmprotocol.OpenAIChatV1,
		SemanticRequest:          request,
		RequestModel:             "entrypoint-model",
	}

	decision := &config.Decision{
		Name: "fusion-stream",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model:          "judge",
				AnalysisModels: []string{"panel-a", "panel-b"},
				AnalysisMode:   config.FusionAnalysisModeOneCall,
			},
		},
	}

	resp, err := router.handleLooperExecution(context.Background(), request, decision, reqCtx)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.GetImmediateResponse())
	require.EqualValues(t, 401, resp.GetImmediateResponse().GetStatus().GetCode())
	require.False(t, reqCtx.LooperStreamingActive)

	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, 401, record.ResponseStatus)
	require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
	require.Equal(t, "looper_stream_dispatch_rejected", record.TerminalReason)
	require.NotNil(t, record.PromptTokens)
	require.Equal(t, 24, *record.PromptTokens) // 2 panels * 12
	require.NotNil(t, record.CompletionTokens)
	require.Equal(t, 12, *record.CompletionTokens) // 2 panels * 6
	require.NotNil(t, record.TotalTokens)
	require.Equal(t, 36, *record.TotalTokens)
}

func TestFinalizeSemanticStreamingResponse_AbnormalStreamSetsLifecycleFailed(t *testing.T) {
	t.Run("responseErr sets LifecycleFailed and resists EOF", func(t *testing.T) {
		memStore := store.NewMemoryStore(10, 0)
		recorder := routerreplay.NewRecorder(memStore)
		replayConfig := config.DefaultRouterReplayPluginConfig()

		router := &OpenAIRouter{
			Config: &config.RouterConfig{
				RouterReplay: config.RouterReplayConfig{},
			},
			ReplayRecorder: recorder,
		}

		reqCtx := &RequestContext{
			RequestID:                "req-abnormal-stream",
			RouterReplayRecorder:     recorder,
			RouterReplayPluginConfig: &replayConfig,
			IsStreamingResponse:      true,
			LooperStreamingActive:    true,
			LooperIntermediateUsage: looper.TokenUsage{
				PromptTokens:     30,
				CompletionTokens: 15,
				TotalTokens:      45,
			},
			SemanticStreamState: &semanticResponseStreamState{
				terminal: false,
			},
		}
		router.startRouterReplay(reqCtx, "router-entrypoint", "judge", "fusion-stream")

		router.finalizeSemanticStreamingResponse(reqCtx, nil)

		require.True(t, reqCtx.StreamingAborted)
		require.True(t, reqCtx.LooperUsageSettled)
		record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.Equal(t, "upstream_stream_incomplete", record.TerminalReason)
		require.NotNil(t, record.PromptTokens)
		require.NotNil(t, record.CompletionTokens)
		require.NotNil(t, record.TotalTokens)
		require.Equal(t, 30, *record.PromptTokens)
		require.Equal(t, 15, *record.CompletionTokens)
		require.Equal(t, 45, *record.TotalTokens)

		require.NoError(t, router.handleProcessReceiveError(reqCtx, io.EOF))
		recordAfterEOF, ok := recorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleFailed, recordAfterEOF.LifecycleState)
		require.Equal(t, "upstream_stream_incomplete", recordAfterEOF.TerminalReason)
		require.Equal(t, 30, *recordAfterEOF.PromptTokens)
		require.Equal(t, 15, *recordAfterEOF.CompletionTokens)
		require.Equal(t, 45, *recordAfterEOF.TotalTokens)
	})

	t.Run("usage invalid sets LifecycleFailed and resists client canceled", func(t *testing.T) {
		memStore := store.NewMemoryStore(10, 0)
		recorder := routerreplay.NewRecorder(memStore)
		replayConfig := config.DefaultRouterReplayPluginConfig()

		router := &OpenAIRouter{
			Config: &config.RouterConfig{
				RouterReplay: config.RouterReplayConfig{},
			},
			ReplayRecorder: recorder,
		}

		streamState := &semanticResponseStreamState{
			responseID: "resp-1",
			model:      "judge",
			stop:       llmprotocol.StopEndTurn,
			terminal:   true,
			items: map[int]*semanticStreamItem{
				0: {completed: true, text: "completed text"},
			},
			order: []int{0},
			usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
		}

		reqCtx := &RequestContext{
			RequestID:                "req-invalid-usage-stream",
			RouterReplayRecorder:     recorder,
			RouterReplayPluginConfig: &replayConfig,
			IsStreamingResponse:      true,
			LooperStreamingActive:    true,
			LooperIntermediateUsage: looper.TokenUsage{
				PromptTokens:     10,
				CompletionTokens: 5,
				TotalTokens:      15,
			},
			SemanticStreamState: streamState,
		}
		router.startRouterReplay(reqCtx, "router-entrypoint", "judge", "fusion-stream")

		router.finalizeSemanticStreamingResponse(reqCtx, nil)

		expectedEstimatedOutput := contextcompression.EstimateTokens("completed text")
		require.True(t, reqCtx.StreamingAborted)
		require.True(t, reqCtx.LooperUsageSettled)
		record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.Equal(t, "upstream_stream_incomplete", record.TerminalReason)
		require.NotNil(t, record.PromptTokens)
		require.NotNil(t, record.CompletionTokens)
		require.NotNil(t, record.TotalTokens)
		require.Equal(t, 10, *record.PromptTokens)
		require.Equal(t, 5+expectedEstimatedOutput, *record.CompletionTokens)
		require.Equal(t, 15+expectedEstimatedOutput, *record.TotalTokens)

		require.NoError(t, router.handleProcessReceiveError(reqCtx, context.Canceled))
		recordAfterCancel, ok := recorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleFailed, recordAfterCancel.LifecycleState)
		require.Equal(t, "upstream_stream_incomplete", recordAfterCancel.TerminalReason)
		require.Equal(t, 10, *recordAfterCancel.PromptTokens)
		require.Equal(t, 5+expectedEstimatedOutput, *recordAfterCancel.CompletionTokens)
		require.Equal(t, 15+expectedEstimatedOutput, *recordAfterCancel.TotalTokens)
	})
}

func TestLooperStreaming_ShouldStreamLooperFinalStageGate(t *testing.T) {
	enabledFalse := false
	enabledTrue := true
	routerDefault := &OpenAIRouter{
		Config: &config.RouterConfig{
			Looper: config.LooperConfig{
				Streaming: config.LooperStreamingConfig{Enabled: nil}, // default false per RFC #3474
			},
		},
	}
	routerDisabled := &OpenAIRouter{
		Config: &config.RouterConfig{
			Looper: config.LooperConfig{
				Streaming: config.LooperStreamingConfig{Enabled: &enabledFalse},
			},
		},
	}
	routerEnabled := &OpenAIRouter{
		Config: &config.RouterConfig{
			Looper: config.LooperConfig{
				Streaming: config.LooperStreamingConfig{Enabled: &enabledTrue},
			},
		},
	}

	fusionDecision := &config.Decision{
		Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmFusion},
	}
	escalateDecision := &config.Decision{
		Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmEscalate},
	}
	eligible := looper.StageOwnership{Eligibility: looper.StreamingEligible}
	ineligible := looper.StageOwnership{Eligibility: looper.StreamingIneligibleBufferingRequired}

	tests := []struct {
		name      string
		router    *OpenAIRouter
		ownership looper.StageOwnership
		reqCtx    *RequestContext
		request   *llmprotocol.Request
		decision  *config.Decision
		want      bool
	}{
		{
			name:      "default unset gate returns false",
			router:    routerDefault,
			ownership: eligible,
			reqCtx:    &RequestContext{ExpectStreamingResponse: true, SourceFormat: llmprotocol.OpenAIChatV1},
			request:   &llmprotocol.Request{Stream: true},
			decision:  fusionDecision,
			want:      false,
		},
		{
			name:      "gate disabled returns false",
			router:    routerDisabled,
			ownership: eligible,
			reqCtx:    &RequestContext{ExpectStreamingResponse: true, SourceFormat: llmprotocol.OpenAIChatV1},
			request:   &llmprotocol.Request{Stream: true},
			decision:  fusionDecision,
			want:      false,
		},
		{
			name:      "non-fusion algorithm returns false",
			router:    routerEnabled,
			ownership: eligible,
			reqCtx:    &RequestContext{ExpectStreamingResponse: true, SourceFormat: llmprotocol.OpenAIChatV1},
			request:   &llmprotocol.Request{Stream: true},
			decision:  escalateDecision,
			want:      false,
		},
		{
			name:      "client did not expect streaming response",
			router:    routerEnabled,
			ownership: eligible,
			reqCtx:    &RequestContext{ExpectStreamingResponse: false, SourceFormat: llmprotocol.OpenAIChatV1},
			request:   &llmprotocol.Request{Stream: true},
			decision:  fusionDecision,
			want:      false,
		},
		{
			name:      "request stream false",
			router:    routerEnabled,
			ownership: eligible,
			reqCtx:    &RequestContext{ExpectStreamingResponse: true, SourceFormat: llmprotocol.OpenAIChatV1},
			request:   &llmprotocol.Request{Stream: false},
			decision:  fusionDecision,
			want:      false,
		},
		{
			name:      "non-OpenAIChat source format (Anthropic)",
			router:    routerEnabled,
			ownership: eligible,
			reqCtx:    &RequestContext{ExpectStreamingResponse: true, SourceFormat: llmprotocol.AnthropicMessagesV1},
			request:   &llmprotocol.Request{Stream: true},
			decision:  fusionDecision,
			want:      false,
		},
		{
			name:      "OpenAI Responses API request",
			router:    routerEnabled,
			ownership: eligible,
			reqCtx:    &RequestContext{ExpectStreamingResponse: true, SourceFormat: llmprotocol.OpenAIResponsesV1},
			request:   &llmprotocol.Request{Stream: true},
			decision:  fusionDecision,
			want:      false,
		},
		{
			name:      "stage ownership requires buffering",
			router:    routerEnabled,
			ownership: ineligible,
			reqCtx:    &RequestContext{ExpectStreamingResponse: true, SourceFormat: llmprotocol.OpenAIChatV1},
			request:   &llmprotocol.Request{Stream: true},
			decision:  fusionDecision,
			want:      false,
		},
		{
			name:      "fusion enabled and eligible returns true",
			router:    routerEnabled,
			ownership: eligible,
			reqCtx:    &RequestContext{ExpectStreamingResponse: true, SourceFormat: llmprotocol.OpenAIChatV1},
			request:   &llmprotocol.Request{Stream: true},
			decision:  fusionDecision,
			want:      true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, tc.router.shouldStreamLooperFinalStage(tc.ownership, tc.reqCtx, tc.request, tc.decision))
		})
	}
}

func TestLooperStreaming_BlocksOuterFallbackAndSettlesUpstreamTransportError(t *testing.T) {
	memStore := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(memStore)
	replayConfig := config.DefaultRouterReplayPluginConfig()
	policy := fallback.FallbackPolicy{
		Enabled:              true,
		MaxAttempts:          2,
		RetryableStatusCodes: []int{502, 503},
	}
	orchestrator := fallback.NewOrchestrator(policy, fallback.NewBackendCircuitBreaker(policy.CircuitBreaker))

	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			RouterReplay: config.RouterReplayConfig{},
		},
		ReplayRecorder:       recorder,
		FallbackOrchestrator: orchestrator,
	}

	reqCtx := &RequestContext{
		RequestID:                "req-transport-err",
		SourceFormat:             llmprotocol.OpenAIChatV1,
		SemanticRequest:          testNeutralRequest("entrypoint", "hello"),
		VSREligibleModelRefs:     []config.ModelRef{{Model: "judge"}, {Model: "backup-model"}},
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
		LooperStreamingActive:    false,
		UpstreamStatusCode:       502,
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		},
	}

	// 1. Pre-check: with real orchestrator + eligible candidates, non-looper-stream allows fallback
	require.True(t, router.shouldAttemptFallback(reqCtx))

	// Enabling LooperStreamingActive specifically blocks outer fallback
	reqCtx.LooperStreamingActive = true
	require.False(t, router.shouldAttemptFallback(reqCtx))

	// 2. Upstream transport error returns translated CONTINUE JSON body and settles intermediate usage
	router.startRouterReplay(reqCtx, "entrypoint", "judge", "fusion-stream")
	errResp := router.handleUpstreamTransportError([]byte(`{"error":{"message":"bad gateway","type":"server_error"}}`), reqCtx)
	require.NotNil(t, errResp)
	require.Nil(t, errResp.GetImmediateResponse())
	common := errResp.GetResponseBody().GetResponse()
	require.NotNil(t, common)
	require.Equal(t, ext_proc.CommonResponse_CONTINUE, common.GetStatus())
	setHeaders := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())
	require.Equal(t, "application/json", setHeaders["content-type"])
	var errJSON map[string]interface{}
	require.NoError(t, json.Unmarshal(common.GetBodyMutation().GetBody(), &errJSON))
	require.Contains(t, errJSON, "error")

	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
	require.Equal(t, "upstream_error_response", record.TerminalReason)
	require.NotNil(t, record.PromptTokens)
	require.Equal(t, 100, *record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.Equal(t, 50, *record.CompletionTokens)
	require.NotNil(t, record.TotalTokens)
	require.Equal(t, 150, *record.TotalTokens)
}

func TestLooperStreaming_PrepareCancelledRecordsLifecycleAborted(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	secondStarted := make(chan struct{})
	var callSeq atomic.Int32

	panelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if callSeq.Add(1) == 1 {
			writeFusionReplayCompletion(w, request.Model, []map[string]interface{}{{
				"index":         0,
				"message":       map[string]interface{}{"role": "assistant", "content": "first panel done"},
				"finish_reason": "stop",
			}}, map[string]int64{"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60})
			return
		}
		close(secondStarted)
		<-r.Context().Done()
	}))
	defer panelServer.Close()

	go func() {
		<-secondStarted
		cancel()
	}()

	router, recorder := newConfiguredFusionStreamingRouter(t, panelServer.URL, "vllm", "          api_key: sk-judge-secret")
	replayConfig := config.DefaultRouterReplayPluginConfig()
	request := testNeutralRequest("entrypoint", "cancel during looper prepare")
	request.Stream = true
	reqCtx := &RequestContext{
		RequestID:                "req-prepare-cancel",
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
		ExpectStreamingResponse:  true,
		SourceFormat:             llmprotocol.OpenAIChatV1,
		SemanticRequest:          request,
	}

	decision := &config.Decision{
		Name: "fusion-stream",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model:                  "judge",
				AnalysisModels:         []string{"panel-a", "panel-b"},
				MaxConcurrent:          1,
				MinSuccessfulResponses: 2,
				QuorumFailurePolicy:    config.FusionQuorumFailurePolicyFallback,
				QuorumFallbackTarget:   "backup-model",
			},
		},
	}

	resp, err := router.handleLooperExecution(ctx, request, decision, reqCtx)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.GetImmediateResponse())
	// Envoy HttpStatus enum maps non-standard 499 to 500 on ImmediateResponse,
	// while Replay preserves the logical 499 client-cancelled status.
	require.EqualValues(t, 500, resp.GetImmediateResponse().GetStatus().GetCode())
	require.False(t, reqCtx.LooperStreamingActive)

	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, 499, record.ResponseStatus)
	require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState)
	require.Equal(t, "client_cancelled_during_looper_prepare", record.TerminalReason)
	require.NotNil(t, record.PromptTokens)
	require.Equal(t, 40, *record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.Equal(t, 20, *record.CompletionTokens)
	require.NotNil(t, record.TotalTokens)
	require.Equal(t, 60, *record.TotalTokens)
}

func TestLooperStreamingPipeline_ClientIncludeUsagePreservesSynthesisUsageFrame(t *testing.T) {
	panelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		writeFusionReplayCompletion(w, request.Model, []map[string]interface{}{{
			"index":         0,
			"message":       map[string]interface{}{"role": "assistant", "content": "insight from " + request.Model},
			"finish_reason": "stop",
		}}, map[string]int64{"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15})
	}))
	defer panelServer.Close()

	router, recorder := newConfiguredFusionStreamingRouter(t, panelServer.URL, "vllm", "          api_key: sk-judge-secret")
	replayConfig := config.DefaultRouterReplayPluginConfig()
	includeUsage := true
	request := testNeutralRequest("router-entrypoint", "explain paxos")
	request.Stream = true
	request.StreamOptions = llmprotocol.StreamOptions{IncludeUsage: &includeUsage}
	reqCtx := &RequestContext{
		RequestID:                "req-client-include-usage",
		SourceFormat:             llmprotocol.OpenAIChatV1,
		SemanticRequest:          request,
		ExpectStreamingResponse:  true,
		RouterReplayRecorder:     recorder,
		RouterReplayPluginConfig: &replayConfig,
	}
	decision := &config.Decision{
		Name: "fusion-client-include-usage",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model:          "judge",
				AnalysisModels: []string{"panel-a", "panel-b"},
				AnalysisMode:   config.FusionAnalysisModeNone,
			},
		},
	}

	reqResp, err := router.handleLooperExecution(context.Background(), request, decision, reqCtx)
	require.NoError(t, err)
	require.Nil(t, reqResp.GetImmediateResponse(), "unexpected immediate response: %s", string(reqResp.GetImmediateResponse().GetBody()))
	require.NotNil(t, reqResp.GetRequestBody())
	require.Equal(t, ext_proc.CommonResponse_CONTINUE, reqResp.GetRequestBody().GetResponse().GetStatus())
	require.True(t, streamUsageRequestedByClient(reqCtx))
	require.True(t, reqCtx.LooperStreamingActive)
	require.Equal(t, "router-entrypoint", reqCtx.RequestModel)
	require.Equal(t, llmprotocol.OpenAIChatV1, reqCtx.TargetFormat)

	_, err = router.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "200"},
					{Key: "content-type", Value: "text/event-stream"},
				},
			},
		},
	}, reqCtx)
	require.NoError(t, err)
	require.Equal(t, "router-entrypoint", reqCtx.RequestModel)
	require.True(t, reqCtx.IsStreamingResponse)

	sseChunks := []string{
		"data: {\"id\":\"chatcmpl-synth-iu\",\"object\":\"chat.completion.chunk\",\"created\":1700000001,\"model\":\"provider/judge-v2\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Paxos answer\"},\"finish_reason\":\"stop\"}]}\n\n",
		"data: {\"id\":\"chatcmpl-synth-iu\",\"object\":\"chat.completion.chunk\",\"created\":1700000001,\"model\":\"provider/judge-v2\",\"choices\":[],\"usage\":{\"prompt_tokens\":17,\"completion_tokens\":6,\"total_tokens\":23}}\n\n" +
			"data: [DONE]\n\n",
	}
	var clientVisibleSSE strings.Builder
	for idx, chunk := range sseChunks {
		bodyResp, bodyErr := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{
				Body:        []byte(chunk),
				EndOfStream: idx == len(sseChunks)-1,
			},
		}, reqCtx)
		require.NoError(t, bodyErr)
		if mut := bodyResp.GetResponseBody().GetResponse().GetBodyMutation(); mut != nil {
			clientVisibleSSE.Write(mut.GetBody())
		}
	}
	require.NotNil(t, reqCtx.PublicChatUsageFilter)
	require.False(t, reqCtx.StreamingAborted)

	rendered := clientVisibleSSE.String()
	// Client explicitly asked for include_usage: synthesis usage frame must be preserved
	// (with rewritten public model), while Router accounting accumulates Panel+Synthesis.
	require.Contains(t, rendered, `"model":"router-entrypoint"`)
	require.Contains(t, rendered, `"Paxos answer"`)
	require.Contains(t, rendered, `"prompt_tokens":17`)
	require.Contains(t, rendered, `"completion_tokens":6`)
	require.Contains(t, rendered, "data: [DONE]")
	require.NotContains(t, rendered, `"provider/judge-v2"`)

	record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
	require.True(t, ok)
	require.NotNil(t, record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.NotNil(t, record.TotalTokens)
	require.Equal(t, 22+17, *record.PromptTokens)
	require.Equal(t, 8+6, *record.CompletionTokens)
	require.Equal(t, 30+23, *record.TotalTokens)
}

func TestLooperStreamingSSE_ToolCallsDeltaAbortAndMidStreamError(t *testing.T) {
	t.Run("real SSE tool_calls delta followed by client abort settles estimated tokens", func(t *testing.T) {
		router, recorder := newConfiguredFusionStreamingRouter(t, "http://127.0.0.1:18080", "vllm", "          api_key: sk-judge")
		replayConfig := config.DefaultRouterReplayPluginConfig()
		request := testNeutralRequest("router-entrypoint", "check weather")
		request.Stream = true
		reqCtx := &RequestContext{
			RequestID:                "req-sse-tool-abort",
			SourceFormat:             llmprotocol.OpenAIChatV1,
			TargetFormat:             llmprotocol.OpenAIChatV1,
			SemanticRequest:          request,
			ExpectStreamingResponse:  true,
			LooperStreamingActive:    true,
			LooperTargetModel:        "judge",
			VSRSelectedModel:         "judge",
			RequestModel:             "router-entrypoint",
			RouterReplayRecorder:     recorder,
			RouterReplayPluginConfig: &replayConfig,
			LooperIntermediateUsage: looper.TokenUsage{
				PromptTokens:     25,
				CompletionTokens: 10,
				TotalTokens:      35,
			},
		}
		router.startRouterReplay(reqCtx, "router-entrypoint", "judge", "fusion-stream")

		_, err := router.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &ext_proc.HttpHeaders{
				Headers: &core.HeaderMap{
					Headers: []*core.HeaderValue{
						{Key: ":status", Value: "200"},
						{Key: "content-type", Value: "text/event-stream"},
					},
				},
			},
		}, reqCtx)
		require.NoError(t, err)

		toolChunks := []string{
			"data: {\"id\":\"chatcmpl-tc-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000002,\"model\":\"provider/judge-v2\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
			"data: {\"id\":\"chatcmpl-tc-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000002,\"model\":\"provider/judge-v2\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\\\"Hangzhou\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
		}
		for _, chunk := range toolChunks {
			_, bodyErr := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body:        []byte(chunk),
					EndOfStream: false,
				},
			}, reqCtx)
			require.NoError(t, bodyErr)
		}

		require.NoError(t, router.handleProcessReceiveError(reqCtx, context.Canceled))

		expectedEstimated := contextcompression.EstimateTokens(`get_weather{"city":"Hangzhou"}`)
		require.Positive(t, expectedEstimated)
		record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, 200, record.ResponseStatus)
		require.Equal(t, routerreplay.LifecycleAborted, record.LifecycleState)
		require.NotNil(t, record.PromptTokens)
		require.NotNil(t, record.CompletionTokens)
		require.Equal(t, 25, *record.PromptTokens)
		require.Equal(t, 10+expectedEstimated, *record.CompletionTokens)
		require.Equal(t, 35+expectedEstimated, *record.TotalTokens)
	})

	t.Run("mid-stream OpenAI SSE error event forwards frame and settles LifecycleFailed once", func(t *testing.T) {
		router, recorder := newConfiguredFusionStreamingRouter(t, "http://127.0.0.1:18080", "vllm", "          api_key: sk-judge")
		replayConfig := config.DefaultRouterReplayPluginConfig()
		request := testNeutralRequest("router-entrypoint", "stream failure midflight")
		request.Stream = true
		reqCtx := &RequestContext{
			RequestID:                "req-sse-mid-err",
			SourceFormat:             llmprotocol.OpenAIChatV1,
			TargetFormat:             llmprotocol.OpenAIChatV1,
			SemanticRequest:          request,
			ExpectStreamingResponse:  true,
			LooperStreamingActive:    true,
			LooperTargetModel:        "judge",
			VSRSelectedModel:         "judge",
			RequestModel:             "router-entrypoint",
			RouterReplayRecorder:     recorder,
			RouterReplayPluginConfig: &replayConfig,
			LooperIntermediateUsage: looper.TokenUsage{
				PromptTokens:     30,
				CompletionTokens: 12,
				TotalTokens:      42,
			},
		}
		router.startRouterReplay(reqCtx, "router-entrypoint", "judge", "fusion-stream")

		_, err := router.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &ext_proc.HttpHeaders{
				Headers: &core.HeaderMap{
					Headers: []*core.HeaderValue{
						{Key: ":status", Value: "200"},
						{Key: "content-type", Value: "text/event-stream"},
					},
				},
			},
		}, reqCtx)
		require.NoError(t, err)

		var clientSSE strings.Builder
		chunks := []string{
			"data: {\"id\":\"chatcmpl-err-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000003,\"model\":\"provider/judge-v2\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"partial\"},\"finish_reason\":null}]}\n\n",
			"data: {\"error\":{\"message\":\"Backend engine overloaded mid-stream\",\"type\":\"vendor_overloaded\",\"param\":null,\"code\":\"provider_overloaded\"}}\n\n",
		}
		for idx, chunk := range chunks {
			bodyResp, bodyErr := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body:        []byte(chunk),
					EndOfStream: idx == len(chunks)-1,
				},
			}, reqCtx)
			require.NoError(t, bodyErr)
			if mut := bodyResp.GetResponseBody().GetResponse().GetBodyMutation(); mut != nil {
				clientSSE.Write(mut.GetBody())
			}
		}

		require.Contains(t, clientSSE.String(), "partial")
		require.Contains(t, clientSSE.String(), "Backend engine overloaded mid-stream")

		// Subsequent io.EOF must not overwrite LifecycleFailed or double-settle usage
		require.NoError(t, router.handleProcessReceiveError(reqCtx, io.EOF))

		expectedPartial := contextcompression.EstimateTokens("partial")
		record, ok := recorder.GetRecord(reqCtx.RouterReplayID)
		require.True(t, ok)
		require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
		require.True(t, reqCtx.LooperUsageSettled)
		require.NotNil(t, record.PromptTokens)
		require.Equal(t, 30, *record.PromptTokens)
		require.NotNil(t, record.CompletionTokens)
		require.Equal(t, 12+expectedPartial, *record.CompletionTokens)
		require.NotNil(t, record.TotalTokens)
		require.Equal(t, 42+expectedPartial, *record.TotalTokens)
	})
}

func TestLooperStreaming_CacheWriteAndLatencyIsolation(t *testing.T) {
	mockCache, router, decision := statusCacheRouter()
	isolatedModel := "looper-stream-latency-isolated-judge"
	latency.RemoveModelFromLatencyCache(isolatedModel)
	t.Cleanup(func() { latency.RemoveModelFromLatencyCache(isolatedModel) })

	router.Config.BackendModels.ModelConfig = map[string]config.ModelParams{
		isolatedModel: {APIFormat: "openai.chat.v1"},
	}

	req := testNeutralRequest("router-entrypoint", "do not cache or update latency")
	req.Stream = true
	reqCtx := withSelectedDecision(&RequestContext{
		RequestID:                "req-stream-cache-latency-isolation",
		SourceFormat:             llmprotocol.OpenAIChatV1,
		TargetFormat:             llmprotocol.OpenAIChatV1,
		SemanticRequest:          req,
		RequestModel:             isolatedModel,
		VSRSelectedModel:         isolatedModel,
		RequestQuery:             "do not cache or update latency",
		ExpectStreamingResponse:  true,
		LooperStreamingActive:    true,
		LooperTargetModel:        isolatedModel,
		LooperSynthesisTextBytes: len("do not cache or update latency"),
		StartTime:                time.Now().Add(-60 * time.Millisecond),
		ProcessingStartTime:      time.Now().Add(-50 * time.Millisecond),
		LooperIntermediateUsage: looper.TokenUsage{
			PromptTokens:     30,
			CompletionTokens: 10,
			TotalTokens:      40,
		},
	}, decision)

	_, err := router.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "200"},
					{Key: "content-type", Value: "text/event-stream"},
				},
			},
		},
	}, reqCtx)
	require.NoError(t, err)
	require.True(t, reqCtx.IsStreamingResponse)

	sseChunks := []string{
		"data: {\"id\":\"chatcmpl-iso-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000005,\"model\":\"" + isolatedModel + "\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Isolated answer\"},\"finish_reason\":\"stop\"}]}\n\n",
		"data: {\"id\":\"chatcmpl-iso-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000005,\"model\":\"" + isolatedModel + "\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n" +
			"data: [DONE]\n\n",
	}
	for idx, chunk := range sseChunks {
		_, bodyErr := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{
				Body:        []byte(chunk),
				EndOfStream: idx == len(sseChunks)-1,
			},
		}, reqCtx)
		require.NoError(t, bodyErr)
	}

	require.False(t, mockCache.addEntryCalled, "Looper streaming final stage must not write to response cache")
	_, hasTTFT := latency.GetTTFT(isolatedModel)
	_, hasTPOT := latency.GetTPOT(isolatedModel)
	require.False(t, hasTTFT, "Looper streaming final stage must not update TTFT cache")
	require.False(t, hasTPOT, "Looper streaming final stage must not update TPOT cache")
}
