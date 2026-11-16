package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/graph"
)

func TestOpenAIHostedResponsesPredicate(t *testing.T) {
	for _, tc := range []struct {
		provider, endpoint string
		want               bool
	}{
		{"openai", "https://api.openai.com/v1", true},
		{"openai", "https://chatgpt.com/backend-api/codex", true},
		{"openai", "https://resource.openai.azure.com/openai/v1", true},
		{"openai", "https://API.OPENAI.COM:443/v1", true},
		{"openai", "http://127.0.0.1:1234", false},
		{"vllm", "https://api.openai.com/v1", false},
		{"openai", "https://api.openai.com.example.org/v1", false},
		{"openai", "https://notopenai.azure.com/v1", false},
		{"openai", "https://openai.azure.com/v1", false},
		{"openai", "https://api.openai.com@localhost/v1", false},
	} {
		t.Run(tc.provider+tc.endpoint, func(t *testing.T) {
			profile := &config.ProviderProfile{Type: tc.provider, BaseURL: tc.endpoint}
			require.Equal(t, tc.want, isOpenAIHostedResponses(profile, "127.0.0.1:1234", llmprotocol.OpenAIResponsesV1))
			require.False(t, isOpenAIHostedResponses(profile, tc.endpoint, llmprotocol.AnthropicMessagesV1))
		})
	}
	require.True(t, isOpenAIHostedResponses(&config.ProviderProfile{Type: "openai"}, "api.openai.com:443", llmprotocol.OpenAIResponsesV1))
	require.False(t, isOpenAIHostedResponses(nil, "api.openai.com:443", llmprotocol.OpenAIResponsesV1))
}

const responsesReplayInput = `{"model":"worker","include":["reasoning.encrypted_content"],"input":[{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"cipher","content":[{"type":"reasoning_text","text":"private thought"}]},{"type":"reasoning","id":"rs_plain","summary":[{"type":"summary_text","text":"plain-only"}]},{"type":"message","role":"user","content":[{"type":"input_text","text":"next"}]}]}`

func TestResponsesReasoningDestinationDispatch(t *testing.T) {
	for _, tc := range []struct {
		name, provider, endpoint string
		format                   llmprotocol.WireFormat
		hosted                   bool
	}{
		{"hosted", "openai", "https://api.openai.com/v1", llmprotocol.OpenAIResponsesV1, true},
		{"compatible", "vllm", "http://127.0.0.1:8000/v1", llmprotocol.OpenAIResponsesV1, false},
		{"messages", "anthropic", "http://127.0.0.1:8000", llmprotocol.AnthropicMessagesV1, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := routingTestRouter("worker")
			params := router.Config.ModelConfig["worker"]
			params.APIFormat = config.APIFormatResponses
			if tc.format == llmprotocol.AnthropicMessagesV1 {
				params.APIFormat = config.APIFormatAnthropic
			}
			router.Config.ModelConfig["worker"] = params
			profile := config.ProviderProfile{Type: tc.provider, BaseURL: tc.endpoint}
			router.Config.ProviderProfiles["provider"] = profile
			engine := protocolcodec.NewBuiltinEngine()
			request, envelope, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, []byte(responsesReplayInput))
			require.NoError(t, err)
			ctx := routingTestContext(llmprotocol.OpenAIResponsesV1, &request)
			ctx.ProtocolEnvelope = envelope
			dispatch, err := router.prepareProviderDispatch(&request, "worker", "", false, ctx)
			require.NoError(t, err)
			require.NoError(t, router.candidateCapabilityMismatch(config.ModelRef{Model: "worker"}, &request, nil, nil, nil, true))
			body, err := router.encodeDispatchRequest(ctx)
			require.NoError(t, err)
			assertResponsesReplayDestination(t, body, tc.hosted)
			require.NotEmpty(t, ctx.ProtocolDiagnostics)
			droppedCounter := metrics.TranslationLossyTotal.WithLabelValues("responses", normalizeProtocol(string(tc.format)), "dropped", "Responses encrypted reasoning is incompatible with the destination")
			beforeShadow := testutil.ToFloat64(droppedCounter)
			// Shadow encoding must make the same projection for its own destination.
			shadowBody, err := router.shadowRequestEncoder(ctx, dispatch, engine)(request, &shadowTarget{
				logicalModel: "worker", upstreamModel: "worker", profile: &profile, baseURL: tc.endpoint, format: tc.format,
			})
			require.NoError(t, err)
			assertResponsesReplayDestination(t, shadowBody, tc.hosted)
			if !tc.hosted {
				require.Equal(t, beforeShadow+1, testutil.ToFloat64(droppedCounter))
			}
			// Neither the primary nor shadow dispatch may strip the shared history.
			require.True(t, request.IncludeEncryptedReasoning)
			require.Equal(t, "cipher", request.Messages[0].Content[1].ReasoningDetail.Data.Value)
		})
	}
}

func assertResponsesReplayDestination(t *testing.T, body []byte, hosted bool) {
	t.Helper()
	if hosted {
		require.Contains(t, string(body), `"encrypted_content":"cipher"`)
		require.Contains(t, string(body), `"include":["reasoning.encrypted_content"]`)
		require.Contains(t, string(body), `"content":[]`)
		require.NotContains(t, string(body), "private thought")
		require.NotContains(t, string(body), "plain-only")
	} else {
		require.NotContains(t, string(body), "encrypted_content")
		require.NotContains(t, string(body), "cipher")
		require.Contains(t, string(body), "private thought")
		require.Contains(t, string(body), "plain-only")
	}
}

func TestResponsesReasoningFallbackProjectsPerCandidate(t *testing.T) {
	for _, mode := range []string{"hosted", "compatible", "mixed"} {
		session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-1")
		router, ctx := session.router, session.ctx
		params := router.Config.ModelConfig["model-fallback-1"]
		params.APIFormat = config.APIFormatResponses
		router.Config.ModelConfig["model-fallback-1"] = params
		profile := config.ProviderProfile{Type: "openai", BaseURL: "https://api.openai.com/v1"}
		if mode == "compatible" {
			profile = config.ProviderProfile{Type: "vllm", BaseURL: "http://127.0.0.1:8002/v1"}
		}
		router.Config.ProviderProfiles["replay-target"] = profile
		router.Config.VLLMEndpoints[1].ProviderProfileName = "replay-target"
		if mode == "mixed" {
			router.Config.ProviderProfiles["gateway"] = config.ProviderProfile{Type: "openai", BaseURL: "https://gateway.example.org/v1"}
			gateway := router.Config.VLLMEndpoints[1]
			gateway.Name, gateway.ProviderProfileName = "gateway", "gateway"
			router.Config.VLLMEndpoints = append(router.Config.VLLMEndpoints, gateway)
			params.PreferredEndpoints = append(params.PreferredEndpoints, "gateway")
			router.Config.ModelConfig["model-fallback-1"] = params
		}
		request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(responsesReplayInput))
		require.NoError(t, err)
		ctx.SemanticRequest = &request
		ctx.FallbackRequest, err = cloneSemanticRequestForReplay(&request)
		require.NoError(t, err)
		step, err := session.Fallback().Next(t.Context(), failedWith(http.StatusServiceUnavailable, "unavailable"))
		require.NoError(t, err)
		require.NotNil(t, step.Call)
		assertResponsesReplayDestination(t, step.Call.Request.Body, mode == "hosted")
	}
}

func TestResponsesEncryptedOnlyClientReplay(t *testing.T) {
	for _, streamed := range []bool{false, true} {
		t.Run(map[bool]string{false: "buffered", true: "streamed"}[streamed], func(t *testing.T) {
			router := &OpenAIRouter{}
			ctx := &RequestContext{
				SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: llmprotocol.OpenAIResponsesV1,
				RequestModel: "worker", TraceContext: t.Context(), ResponseObjectState: &ResponseObjectState{GeneratedResponseID: "resp_public"},
			}
			var clientBody []byte
			if streamed {
				fixture, err := os.ReadFile("../protocolcodec/testdata/responses-reasoning-parts.sse")
				require.NoError(t, err)
				var input bytes.Buffer
				sequence := 0
				for _, line := range strings.Split(string(fixture), "\n") {
					if !strings.HasPrefix(line, "data: ") {
						continue
					}
					var frame map[string]any
					require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &frame))
					typ := frame["type"].(string)
					if strings.HasPrefix(typ, "response.reasoning_") || strings.HasPrefix(typ, "response.content_part.") && frame["output_index"] == float64(0) {
						continue
					}
					if item, ok := frame["item"].(map[string]any); ok && item["type"] == "reasoning" {
						item["summary"], item["content"] = []any{}, []any{}
					}
					if response, ok := frame["response"].(map[string]any); ok && typ == "response.completed" {
						item := response["output"].([]any)[0].(map[string]any)
						item["summary"], item["content"] = []any{}, []any{}
					}
					frame["sequence_number"] = sequence
					sequence++
					encoded, encodeErr := json.Marshal(frame)
					require.NoError(t, encodeErr)
					input.WriteString("event: " + typ + "\ndata: " + string(encoded) + "\n\n")
				}
				result := router.handleSemanticStreamingResponseBody(input.Bytes(), true, ctx)
				require.False(t, ctx.StreamingAborted, "%+v", ctx.ProtocolDiagnostics)
				clientBody = result.GetResponseBody().GetResponse().GetBodyMutation().GetBody()
				var terminal struct {
					Response struct {
						Output []json.RawMessage `json:"output"`
					} `json:"response"`
				}
				for _, line := range strings.Split(string(clientBody), "\n") {
					if strings.HasPrefix(line, "data: ") && strings.Contains(line, `"type":"response.completed"`) {
						require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &terminal))
					}
				}
				require.NotEmpty(t, terminal.Response.Output, "%s", clientBody)
				clientBody, err = json.Marshal(map[string]any{"output": terminal.Response.Output})
				require.NoError(t, err)
			} else {
				result := router.handleNonStreamingResponseBody([]byte(`{"id":"resp_upstream","object":"response","created_at":0,"model":"worker","status":"completed","output":[{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"cipher_A"}],"error":null,"incomplete_details":null}`), ctx, 0, false)
				clientBody = result.GetResponseBody().GetResponse().GetBodyMutation().GetBody()
			}
			require.Contains(t, string(clientBody), `"encrypted_content":"cipher_A"`)
			require.NotContains(t, string(clientBody), "reasoning_text")
			var reply struct {
				Output []json.RawMessage `json:"output"`
			}
			require.NoError(t, json.Unmarshal(clientBody, &reply))
			reply.Output = append(reply.Output, json.RawMessage(`{"type":"message","role":"user","content":[{"type":"input_text","text":"next"}]}`))
			replay, err := json.Marshal(map[string]any{"model": "worker", "input": reply.Output, "include": []string{"reasoning.encrypted_content"}})
			require.NoError(t, err)
			request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIResponsesV1, replay)
			require.NoError(t, err)
			router = routingTestRouter("worker")
			params := router.Config.ModelConfig["worker"]
			params.APIFormat = config.APIFormatResponses
			router.Config.ModelConfig["worker"] = params
			router.Config.ProviderProfiles["provider"] = config.ProviderProfile{Type: "openai", BaseURL: "https://api.openai.com/v1"}
			next := routingTestContext(llmprotocol.OpenAIResponsesV1, &request)
			next.ProtocolEnvelope = envelope
			_, err = router.prepareProviderDispatch(&request, "worker", "", false, next)
			require.NoError(t, err)
			body, err := router.encodeDispatchRequest(next)
			require.NoError(t, err)
			require.Contains(t, string(body), `"encrypted_content":"cipher_A"`)
			require.Contains(t, string(body), `"content":[]`)
		})
	}
}

func TestStoredResponsesReasoningKeepsCiphertext(t *testing.T) {
	history, err := materializeStoredResponseHistory(protocolcodec.NewBuiltinEngine(), []*responseapi.StoredResponse{{
		Output: []responseapi.OutputItem{{Type: "reasoning", ID: "rs_1", EncryptedContent: "cipher"}},
	}})
	require.NoError(t, err)
	require.Equal(t, "cipher", history.Messages[0].Content[0].ReasoningDetail.Data.Value)
}

type responsesRecoveryCaller struct{ request []byte }

func (caller *responsesRecoveryCaller) Call(_ context.Context, request *graph.HopRequest) (*graph.HopResponse, error) {
	caller.request = append([]byte(nil), request.Request.Body...)
	return &graph.HopResponse{Status: http.StatusOK, Body: []byte(`{"id":"followup","model":"worker","choices":[{"index":0,"message":{"role":"assistant","content":"final answer"},"finish_reason":"stop"}]}`)}, nil
}

func TestResponsesContinuityDroppedInternally(t *testing.T) {
	engine := protocolcodec.NewBuiltinEngine()
	request, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(responsesReplayInput))
	require.NoError(t, err)
	router := routingTestRouter("worker")
	router.CompressionRecovery = &recoveryStoreStub{}
	caller := &responsesRecoveryCaller{}
	router.hopCaller = caller
	decision := &config.Decision{Name: "panel", ModelRefs: []config.ModelRef{{Model: "worker"}}, Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmReMoM}}
	looperCtx := &RequestContext{SourceFormat: llmprotocol.OpenAIResponsesV1, TraceContext: t.Context()}
	built, rejected := router.buildLooperRequest(&request, decision, looperCtx)
	require.Nil(t, rejected)
	require.NotNil(t, built)
	body, err := json.Marshal(built.OriginalRequest)
	require.NoError(t, err)
	require.NotContains(t, string(body), "cipher")
	require.NotContains(t, string(body), "include")
	require.Equal(t, 1, droppedFieldCount(looperCtx.ProtocolDiagnostics, "input[].encrypted_content"))
	decision.Plugins = []config.DecisionPlugin{{Type: config.DecisionPluginContext, Configuration: compressPayload(map[string]interface{}{
		"recovery": map[string]interface{}{"enabled": true, "store": "redis"},
	})}}
	router.Config.Decisions = []config.Decision{*decision}
	ctx := &RequestContext{RequestID: "request-1", RequestModel: "worker", Headers: map[string]string{"x-authz-user-id": "user-1"}, SourceFormat: llmprotocol.OpenAIResponsesV1, SemanticRequest: &request, VSRSelectedDecision: decision, VSRSelectedDecisionName: decision.Name, ContextCompressionRecoveryKeys: []string{"key-1"}}
	scope := router.contextCompressionScope(ctx)
	router.CompressionRecovery.(*recoveryStoreStub).entries = map[string]contextcompression.RecoveryEntry{
		scope + ":key-1": {Key: "key-1", Scope: scope, Content: "recovered context"},
	}
	reply := []byte(`{"id":"resp_1","object":"response","created_at":0,"model":"worker","status":"completed","output":[{"type":"function_call","id":"fc_1","call_id":"retrieve-1","name":"vsr_context_retrieve","arguments":"{\"key\":\"key-1\"}"}],"error":null,"incomplete_details":null}`)
	// The dispatch to a Responses backend that is not OpenAI-hosted already
	// dropped and reported the blob before the backend asked for recovery.
	_, ctx.ProtocolDiagnostics = llmprotocol.ProjectResponsesReasoning(request, llmprotocol.OpenAIResponsesV1, false)
	followup, err := router.handleContextRecoveryFollowup(t.Context(), reply, ctx)
	require.NoError(t, err)
	require.Contains(t, string(followup), "final answer")
	require.Contains(t, string(caller.request), "recovered context")
	require.NotContains(t, string(caller.request), "cipher")
	require.NotContains(t, string(caller.request), "include")
	require.Equal(t, 1, droppedFieldCount(ctx.ProtocolDiagnostics, "input[].encrypted_content"))
	// Two blobs a hosted dispatch kept are two losses for the followup.
	twoBlobs, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(strings.Replace(responsesReplayInput, `{"type":"reasoning","id":"rs_plain"`, `{"type":"reasoning","id":"rs_2","summary":[],"encrypted_content":"cipher2"},{"type":"reasoning","id":"rs_plain"`, 1)))
	require.NoError(t, err)
	ctx.SemanticRequest, ctx.ProtocolDiagnostics = &twoBlobs, nil
	_, err = router.handleContextRecoveryFollowup(t.Context(), reply, ctx)
	require.NoError(t, err)
	require.Equal(t, 2, droppedFieldCount(ctx.ProtocolDiagnostics, "input[].encrypted_content"))
	history, err := memoryHistoryForExtractor(nil, nil, request.Messages)
	require.NoError(t, err)
	body, err = json.Marshal(history)
	require.NoError(t, err)
	require.NotContains(t, string(body), "cipher")
	require.Contains(t, string(body), "next")

	other, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, []byte(`{"model":"worker","messages":[{"role":"assistant","reasoning_details":[{"type":"reasoning.encrypted","data":"other","format":"google-gemini-v1"}]},{"role":"user","content":"next"}]}`))
	require.NoError(t, err)
	built, rejected = router.buildLooperRequest(&other, decision, &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1, TraceContext: t.Context()})
	require.Nil(t, built)
	require.EqualValues(t, 400, rejected.GetImmediateResponse().GetStatus().GetCode())
	ctx.SemanticRequest = &other
	_, err = router.handleContextRecoveryFollowup(t.Context(), reply, ctx)
	require.ErrorContains(t, err, "reasoning continuity")
}

func TestResponsesMixedEndpointPoolDropsBlob(t *testing.T) {
	router := routingTestRouter("worker")
	params := router.Config.ModelConfig["worker"]
	params.APIFormat = config.APIFormatResponses
	router.Config.ProviderProfiles["provider"] = config.ProviderProfile{Type: "openai", BaseURL: "https://api.openai.com/v1"}
	router.Config.ProviderProfiles["gateway"] = config.ProviderProfile{Type: "openai", BaseURL: "https://gateway.example.org/v1"}
	gateway := router.Config.VLLMEndpoints[0]
	gateway.Name, gateway.ProviderProfileName = "gateway", "gateway"
	router.Config.VLLMEndpoints = append(router.Config.VLLMEndpoints, gateway)
	params.PreferredEndpoints = append(params.PreferredEndpoints, "gateway")
	router.Config.ModelConfig["worker"] = params
	request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIResponsesV1, []byte(responsesReplayInput))
	require.NoError(t, err)
	projected, _ := router.projectResponsesReasoningForBackend(request, "worker", llmprotocol.OpenAIResponsesV1)
	require.False(t, projected.IncludeEncryptedReasoning)
	ctx := routingTestContext(llmprotocol.OpenAIResponsesV1, &request)
	ctx.ProtocolEnvelope = envelope
	dispatch, err := router.prepareProviderDispatch(&request, "worker", "", false, ctx)
	require.NoError(t, err)
	require.NoError(t, router.candidateCapabilityMismatch(config.ModelRef{Model: "worker"}, &request, nil, nil, nil, true))
	body, err := router.encodeDispatchRequest(ctx)
	require.NoError(t, err)
	assertResponsesReplayDestination(t, body, false)
	profile := router.Config.ProviderProfiles["provider"]
	shadowBody, err := router.shadowRequestEncoder(ctx, dispatch, protocolcodec.NewBuiltinEngine())(request, &shadowTarget{
		logicalModel: "worker", upstreamModel: "worker", profile: &profile, baseURL: profile.BaseURL, format: llmprotocol.OpenAIResponsesV1,
	})
	require.NoError(t, err)
	assertResponsesReplayDestination(t, shadowBody, true)
}

func TestResponsesQualificationRefusesOtherContinuityAndSignedThinking(t *testing.T) {
	router := routingTestRouter("worker")
	params := router.Config.ModelConfig["worker"]
	params.APIFormat = config.APIFormatResponses
	router.Config.ModelConfig["worker"] = params
	router.Config.ProviderProfiles["provider"] = config.ProviderProfile{Type: "openai", BaseURL: "https://api.openai.com/v1"}
	for _, content := range []llmprotocol.Content{
		{Kind: llmprotocol.ContentReasoning, Text: "signed", Signature: "signature"},
		{Kind: llmprotocol.ContentReasoning, ReasoningDetail: &llmprotocol.ReasoningDetail{Type: llmprotocol.ReasoningDetailEncrypted, Data: llmprotocol.OptionalString{Present: true, Value: "other"}}},
	} {
		request := llmprotocol.Request{Generation: 1, Model: "worker", Messages: []llmprotocol.Message{{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{content}}}}
		capability := "reasoning_signature"
		if content.ReasoningDetail != nil {
			capability = "reasoning_continuity"
		}
		require.ErrorContains(t, router.candidateCapabilityMismatch(config.ModelRef{Model: "worker"}, &request, nil, nil, nil, true), capability)
		_, err := router.prepareProviderDispatch(&request, "worker", "", false, routingTestContext(llmprotocol.OpenAIChatV1, &request))
		require.ErrorContains(t, err, capability)
	}
}

func droppedFieldCount(diagnostics llmprotocol.Diagnostics, field string) int {
	count := 0
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == field && diagnostic.Action == llmprotocol.DiagnosticDropped {
			count++
		}
	}
	return count
}
