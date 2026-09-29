package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
)

func TestLooperIngressPreservesClientResponseContract(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		writeFusionReplayCompletion(w, request.Model, []map[string]interface{}{{"index": 0, "message": map[string]interface{}{"role": "assistant", "content": "A useful answer"}, "finish_reason": "stop"}}, map[string]int64{"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3})
	}))
	defer server.Close()
	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
		for _, stream := range []bool{false, true} {
			for _, trace := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/stream_%t/trace_%t", format, stream, trace), func(t *testing.T) {
					router, ctx, _ := quorumFallbackRouter(server.URL)
					ctx.SourceFormat = format
					ctx.TraceContext = context.Background()
					raw := looperIngressBody(t, format, stream)
					var objectStore *responsestore.MemoryStore
					if format == llmprotocol.OpenAIResponsesV1 {
						var err error
						objectStore, err = responsestore.NewMemoryStore(responsestore.StoreConfig{Enabled: true})
						require.NoError(t, err)
						t.Cleanup(func() { require.NoError(t, objectStore.Close()) })
						require.NoError(t, objectStore.StoreResponse(ctx.TraceContext, &responseapi.StoredResponse{ID: "resp_previous", Object: "response", Status: "completed"}))
						router.ResponseAPIFilter = NewResponseAPIFilter(objectStore)
						var input map[string]interface{}
						require.NoError(t, json.Unmarshal(raw, &input))
						input["previous_response_id"] = "resp_previous"
						input["store"] = true
						raw, err = json.Marshal(input)
						require.NoError(t, err)
					}
					request, rejected := router.prepareProtocolRequest(raw, ctx)
					require.Nil(t, rejected)
					require.Equal(t, stream, request.Stream)
					require.Equal(t, stream, ctx.ExpectStreamingResponse)
					_, err := router.extractRequestSignalSnapshot(ctx)
					require.NoError(t, err)
					decision := &config.Decision{Name: "ingress-fusion", Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmFusion, Fusion: &config.FusionAlgorithmConfig{Model: "judge", AnalysisModels: []string{"panel-a", "panel-b"}, AnalysisMode: config.FusionAnalysisModeNone}}}
					decision.Algorithm.Fusion.IncludeAnalysis = &trace
					decision.Algorithm.Fusion.IncludeIntermediateResponses = &trace
					response, err := router.handleLooperExecution(context.Background(), request, decision, ctx)
					require.NoError(t, err)
					require.EqualValues(t, 200, response.GetImmediateResponse().GetStatus().GetCode())
					body := string(response.GetImmediateResponse().GetBody())
					require.Contains(t, body, "A useful answer")
					contentType := "application/json"
					if stream {
						contentType = "text/event-stream"
					}
					require.Equal(t, contentType, immediateHeaderValue(response, "content-type"))
					if stream {
						terminal := map[llmprotocol.WireFormat]string{llmprotocol.OpenAIChatV1: "data: [DONE]", llmprotocol.OpenAIResponsesV1: "event: response.completed", llmprotocol.AnthropicMessagesV1: "event: message_stop"}
						require.Contains(t, body, terminal[format])
					}
					if format == llmprotocol.OpenAIResponsesV1 {
						require.NotEmpty(t, ctx.ResponseObjectState.GeneratedResponseID)
						require.Equal(t, ctx.ResponseObjectState.GeneratedResponseID, ctx.SemanticResponse.ID)
						require.Contains(t, body, `"id":"`+ctx.ResponseObjectState.GeneratedResponseID+`"`)
						require.Contains(t, body, `"previous_response_id":"resp_previous"`)
						if stream {
							objects := 0
							for _, line := range strings.Split(body, "\n") {
								if !strings.HasPrefix(line, "data: ") {
									continue
								}
								var event struct {
									Response *struct {
										ID         string `json:"id"`
										PreviousID string `json:"previous_response_id"`
									} `json:"response"`
								}
								require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event))
								if event.Response != nil {
									objects++
									require.Equal(t, ctx.ResponseObjectState.GeneratedResponseID, event.Response.ID)
									require.Equal(t, "resp_previous", event.Response.PreviousID)
								}
							}
							require.GreaterOrEqual(t, objects, 3)
						}
						router.persistImmediateResponseObject(response, ctx)
						stored, getErr := objectStore.GetResponse(ctx.TraceContext, ctx.ResponseObjectState.GeneratedResponseID)
						require.NoError(t, getErr)
						require.Equal(t, ctx.SemanticResponse.ID, stored.ID)
						require.Equal(t, "resp_previous", stored.PreviousResponseID)
						require.Equal(t, "A useful answer", stored.OutputText)
						assertStoredResponseMatchesClientOutput(t, router.ResponseAPIFilter, stored.ID, []byte(body), stream)
					}
					if format != llmprotocol.OpenAIChatV1 {
						require.NotContains(t, body, `"fusion"`)
						warning := immediateHeaderValue(response, headers.VSRProtocolWarnings)
						if trace {
							require.Contains(t, warning, "dropped;router_extension_unsupported_protocol;fusion")
						} else {
							require.NotContains(t, warning, "router_extension")
						}
					}
				})
			}
		}
	}
}

func TestRatingsResponsesIngressPreservesCapabilityGate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		writeFusionReplayCompletion(w, request.Model, []map[string]interface{}{{"index": 0, "message": map[string]interface{}{"role": "assistant", "content": request.Model + " answer"}, "finish_reason": "stop"}}, map[string]int64{"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3})
	}))
	defer server.Close()
	for _, candidates := range []int{1, 2} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("candidates_%d/stream_%t", candidates, stream), func(t *testing.T) {
				router, ctx, _ := quorumFallbackRouter(server.URL)
				ctx.SourceFormat = llmprotocol.OpenAIResponsesV1
				ctx.TraceContext = context.Background()
				request, rejected := router.prepareProtocolRequest(looperIngressBody(t, ctx.SourceFormat, stream), ctx)
				require.Nil(t, rejected)
				refs := []config.ModelRef{{Model: "panel-a"}, {Model: "panel-b"}}
				decision := &config.Decision{Name: "ratings-ingress", ModelRefs: refs[:candidates], Algorithm: &config.AlgorithmConfig{Type: "ratings", Ratings: &config.RatingsAlgorithmConfig{OnError: "fail"}}}
				response, err := router.handleLooperExecution(context.Background(), request, decision, ctx)
				require.NoError(t, err)
				if candidates == 2 {
					require.EqualValues(t, 502, response.GetImmediateResponse().GetStatus().GetCode())
					require.Nil(t, ctx.SemanticResponse)
					runner, runnerErr := router.createLooper(decision, ctx)
					require.NoError(t, runnerErr)
					internal, internalError := router.buildLooperRequest(request, decision, ctx)
					require.Nil(t, internalError)
					result, executeErr := runner.Execute(context.Background(), internal)
					require.NoError(t, executeErr)
					_, _, _, encodeErr := router.prepareLooperResponse(result, ctx)
					var typed *llmprotocol.ProtocolError
					require.ErrorAs(t, encodeErr, &typed)
					require.Equal(t, "unsupported_capability", typed.Code)
					t.Logf("two-candidate Responses retains typed rejection: %s", typed.Code)
					return
				}
				require.EqualValues(t, 200, response.GetImmediateResponse().GetStatus().GetCode())
				body := string(response.GetImmediateResponse().GetBody())
				require.Contains(t, body, "panel-a answer")
				require.NotContains(t, body, "panel-b answer")
				require.Empty(t, ctx.SemanticResponse.Alternatives)
				require.EqualValues(t, 3, *ctx.SemanticResponse.Usage.Total.Value)
				require.Equal(t, ctx.ResponseObjectState.GeneratedResponseID, ctx.SemanticResponse.ID)
				require.Equal(t, stream, strings.Contains(immediateHeaderValue(response, "content-type"), "text/event-stream"))
				if stream {
					require.Contains(t, body, "event: response.completed")
				}
			})
		}
	}
}

func looperIngressBody(t *testing.T, format llmprotocol.WireFormat, stream bool) []byte {
	t.Helper()
	body := map[string]interface{}{"model": "router-entrypoint", "stream": stream}
	if format == llmprotocol.OpenAIResponsesV1 {
		body["input"] = "Compare two ordinary options"
		body["store"] = false
	} else {
		body["messages"] = []map[string]string{{"role": "user", "content": "Compare two ordinary options"}}
		if format == llmprotocol.AnthropicMessagesV1 {
			body["max_tokens"] = 32
		}
	}
	raw, err := json.Marshal(body)
	require.NoError(t, err)
	return raw
}

func TestBufferedLooperClientStreamEnforcesFinalLimits(t *testing.T) {
	response := tracedLooperFixture(t, false, false)
	router := &OpenAIRouter{}
	newContext := func() *RequestContext {
		return &RequestContext{SourceFormat: llmprotocol.OpenAIResponsesV1, ExpectStreamingResponse: true}
	}
	_, _, body, err := router.prepareLooperResponse(response, newContext())
	require.NoError(t, err)
	for _, frames := range []bool{false, true} {
		limit := len(body)
		if frames {
			limit = 0
			for _, frame := range bytes.Split(body, []byte("\n\n")) {
				if len(frame) > 0 && len(frame)+2 > limit {
					limit = len(frame) + 2
				}
			}
		}
		for _, oneOver := range []bool{false, true} {
			t.Run(fmt.Sprintf("frames_%t/one_over_%t", frames, oneOver), func(t *testing.T) {
				policy := llmprotocol.DefaultPolicy()
				budget := limit
				if oneOver {
					budget--
				}
				if frames {
					policy.Limits.SSEFrameBytes = budget
				} else {
					policy.Limits.BodyBytes = budget
				}
				engine, engineErr := protocolcodec.NewEngine(protocolcodec.NewBuiltinRegistry(), policy)
				require.NoError(t, engineErr)
				out, _, _, renderErr := router.prepareLooperResponseWithEngine(response, newContext(), engine)
				if oneOver {
					require.Error(t, renderErr)
					require.Nil(t, out)
				} else {
					require.NoError(t, renderErr)
					require.Equal(t, body, out.GetImmediateResponse().GetBody())
				}
			})
		}
	}
}
