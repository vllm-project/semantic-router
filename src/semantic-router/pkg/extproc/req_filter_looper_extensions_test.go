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

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func tracedLooperFixture(t *testing.T, streaming, toolCalls bool) *looper.Response {
	return tracedLooperFamilyFixture(t, "fusion", streaming, toolCalls)
}

func tracedLooperFamilyFixture(t *testing.T, family string, streaming, toolCalls bool) *looper.Response {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		message := map[string]interface{}{"role": "assistant", "content": strings.Repeat("Useful evidence. ", 128)}
		finish := "stop"
		if toolCalls && request.Model == "judge" {
			message["content"] = nil
			message["tool_calls"] = []map[string]interface{}{{"id": "call1", "type": "function", "function": map[string]string{"name": "weather", "arguments": `{"city":"Paris"}`}}}
			finish = "tool_calls"
		}
		writeFusionReplayCompletion(w, request.Model, []map[string]interface{}{{"index": 0, "message": message, "finish_reason": finish}}, map[string]int64{"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3})
	}))
	t.Cleanup(server.Close)
	algorithm := &config.AlgorithmConfig{Type: config.DecisionAlgorithmFusion, Fusion: &config.FusionAlgorithmConfig{Model: "judge", AnalysisModels: []string{"panel-a", "panel-b"}, AnalysisMode: config.FusionAnalysisModeNone}}
	params := openai.ChatCompletionNewParams{Model: "fixture", Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("Compare two ordinary options")}}
	if toolCalls {
		params.Tools = []openai.ChatCompletionToolParam{{Function: openai.FunctionDefinitionParam{Name: "weather", Parameters: openai.FunctionParameters{"type": "object", "properties": map[string]interface{}{"city": map[string]string{"type": "string"}}}}}}
	}
	var runner looper.Looper
	cfg := &config.LooperConfig{Endpoint: server.URL}
	switch family {
	case "fusion":
		runner = looper.NewFusionLooper(cfg)
	case "flow":
		managed := looper.NewWorkflowsLooper(cfg)
		t.Cleanup(func() { _ = managed.Close() })
		runner = managed
		on := true
		algorithm = &config.AlgorithmConfig{Type: config.DecisionAlgorithmWorkflows, Workflows: &config.WorkflowsAlgorithmConfig{
			Mode: config.WorkflowModeStatic, Roles: []config.WorkflowRoleConfig{{Name: "worker", Models: []string{"panel-a"}, Prompt: "Compare the options"}}, Final: config.WorkflowFinalConfig{Model: "judge"}, IncludeIntermediateResponses: &on,
		}}
	case "reasoning_mom_responses":
		runner = looper.NewReMoMLooper(cfg)
		algorithm = &config.AlgorithmConfig{Type: config.DecisionAlgorithmReMoM, ReMoM: &config.ReMoMAlgorithmConfig{BreadthSchedule: []int{1}, ModelDistribution: "round_robin", SynthesisModel: "judge", IncludeIntermediateResponses: true}}
	}
	response, err := runner.Execute(context.Background(), &looper.Request{
		OriginalRequest: &params, Algorithm: algorithm, IsStreaming: streaming, DecisionName: "fixture", ModelRefs: []config.ModelRef{{Model: "panel-a"}, {Model: "judge"}},
		ModelParams: map[string]config.ModelParams{"panel-a": {Capabilities: []string{"chat"}}, "panel-b": {Capabilities: []string{"chat"}}, "judge": {Capabilities: []string{"chat"}}},
	})
	require.NoError(t, err)
	require.Len(t, response.RouterExtensions(), 1)
	return response
}

func TestLooperTraceExtensionsCrossResponseBoundary(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, tools := range []bool{false, true} {
			t.Run(fmt.Sprintf("stream_%t/tools_%t", stream, tools), func(t *testing.T) {
				response := tracedLooperFixture(t, stream, tools)
				for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
					t.Run(string(target), func(t *testing.T) {
						ctx := &RequestContext{SourceFormat: target}
						out, semantic, body, err := (&OpenAIRouter{}).prepareLooperResponse(response, ctx)
						require.NoError(t, err)
						require.EqualValues(t, 200, out.GetImmediateResponse().GetStatus().GetCode())
						require.NotNil(t, semantic)
						if target == llmprotocol.OpenAIChatV1 {
							require.Contains(t, string(body), `"fusion"`)
							require.Empty(t, ctx.ProtocolDiagnostics)
						} else {
							require.NotContains(t, string(body), `"fusion"`)
							warning := ""
							for _, header := range out.GetImmediateResponse().GetHeaders().GetSetHeaders() {
								if header.GetHeader().GetKey() == headers.VSRProtocolWarnings {
									warning = string(header.GetHeader().GetRawValue())
								}
							}
							require.Contains(t, warning, "dropped;router_extension_unsupported_protocol;fusion")
							require.LessOrEqual(t, len(warning), lossinessHeaderSizeLimit)
						}
						if tools {
							require.Contains(t, string(body), "weather")
						}
					})
				}
			})
		}
	}
}

func TestLooperExtensionsEnforceFinalEncodedLimits(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream_%t", stream), func(t *testing.T) {
			response := tracedLooperFixture(t, stream, false)
			router := &OpenAIRouter{}
			_, _, full, err := router.prepareLooperResponse(response, &RequestContext{})
			require.NoError(t, err)
			for _, frameLimit := range []bool{false, true} {
				if frameLimit && !stream {
					continue
				}
				limit := len(full)
				if frameLimit {
					limit = 0
					for _, frame := range bytes.Split(full, []byte("\n\n")) {
						if len(frame) > 0 && len(frame)+2 > limit {
							limit = len(frame) + 2
						}
					}
				}
				for _, extraByte := range []bool{false, true} {
					t.Run(fmt.Sprintf("frame_%t/one_over_%t", frameLimit, extraByte), func(t *testing.T) {
						policy := llmprotocol.DefaultPolicy()
						budget := limit
						if extraByte {
							budget--
						}
						if frameLimit {
							policy.Limits.SSEFrameBytes = budget
						} else {
							policy.Limits.BodyBytes = budget
						}
						engine, engineErr := protocolcodec.NewEngine(protocolcodec.NewBuiltinRegistry(), policy)
						require.NoError(t, engineErr)
						ctx := &RequestContext{}
						out, _, body, encodeErr := router.prepareLooperResponseWithEngine(response, ctx, engine)
						require.NoError(t, encodeErr)
						require.NoError(t, engine.ValidateEncodedResponse(body, stream))
						require.Contains(t, string(body), "Useful evidence")
						if !extraByte {
							require.Equal(t, full, body)
							require.Empty(t, ctx.ProtocolDiagnostics)
						} else {
							require.NotContains(t, string(body), `"fusion"`)
							require.Len(t, ctx.ProtocolDiagnostics, 1)
							require.Equal(t, "router_extension_size_limit", ctx.ProtocolDiagnostics[0].Reason)
							found := false
							for _, header := range out.GetImmediateResponse().GetHeaders().GetSetHeaders() {
								if header.GetHeader().GetKey() == headers.VSRProtocolWarnings {
									found = true
									require.Contains(t, string(header.GetHeader().GetRawValue()), "router_extension_size_limit;fusion")
								}
							}
							require.True(t, found, "immediate response must carry the omission diagnostic")
						}
					})
				}
			}
			plain := &looper.Response{Body: response.ProtocolBody(), Model: response.Model, ContentType: response.ContentType}
			policy := llmprotocol.DefaultPolicy()
			policy.Limits.BodyBytes = 10
			engine, err := protocolcodec.NewEngine(protocolcodec.NewBuiltinRegistry(), policy)
			require.NoError(t, err)
			_, _, _, err = router.prepareLooperResponseWithEngine(plain, &RequestContext{}, engine)
			require.Error(t, err, "required answer cannot be omitted to fit")
		})
	}
}

func TestLooperUntrustedBodyUsesStrictProtocolInput(t *testing.T) {
	for _, field := range []string{"fusion", "flow", "reasoning_mom_responses"} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream_%t", field, stream), func(t *testing.T) {
				var body []byte
				if stream {
					body = []byte("data: " + `{"id":"c1","object":"chat.completion.chunk","model":"fixture","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"` + field + `":{"summary":"ordinary fixture"}}` + "\n\ndata: [DONE]\n\n")
				} else {
					body = []byte(`{"id":"c1","object":"chat.completion","model":"fixture","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"` + field + `":{"summary":"ordinary fixture"}}`)
				}
				response := &looper.Response{Body: body, Model: "fixture", ContentType: "application/json"}
				if stream {
					response.ContentType = "text/event-stream"
				}
				_, _, _, err := (&OpenAIRouter{}).prepareLooperResponse(response, &RequestContext{})
				require.Error(t, err)
				require.Equal(t, body, response.ProtocolBody())
				require.Empty(t, response.RouterExtensions())
			})
		}
	}
}

func TestLooperWorkflowAndReMoMTraceTransport(t *testing.T) {
	for _, family := range []string{"flow", "reasoning_mom_responses"} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream_%t", family, stream), func(t *testing.T) {
				response := tracedLooperFamilyFixture(t, family, stream, false)
				for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
					t.Run(string(target), func(t *testing.T) {
						ctx := &RequestContext{SourceFormat: target, VSRSelectedDecision: &config.Decision{Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmWorkflows}}}
						out, _, body, err := (&OpenAIRouter{}).prepareLooperResponse(response, ctx)
						require.NoError(t, err)
						require.EqualValues(t, 200, out.GetImmediateResponse().GetStatus().GetCode())
						if target == llmprotocol.OpenAIChatV1 {
							require.Contains(t, string(body), `"`+family+`"`)
						} else {
							require.NotContains(t, string(body), `"`+family+`"`)
							require.Contains(t, ctx.ProtocolDiagnostics, llmprotocol.Diagnostic{Source: llmprotocol.OpenAIChatV1, Target: target, Field: family, Action: llmprotocol.DiagnosticDropped, Reason: "router_extension_unsupported_protocol"})
						}
					})
				}
			})
		}
	}
}

func TestLooperNativeStreamRetainsStrictInputAndFinalLimits(t *testing.T) {
	const snapshot = `{"id":"c1","object":"chat.completion","created":1,"model":"fixture","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}]}`
	const stream = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"fixture\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
	response := &looper.Response{Body: []byte(stream), BufferedBody: []byte(snapshot), Model: "fixture", ContentType: "text/event-stream"}
	router := &OpenAIRouter{}
	for _, over := range []bool{false, true} {
		policy := llmprotocol.DefaultPolicy()
		policy.Limits.BodyBytes = len(stream)
		if over {
			policy.Limits.BodyBytes--
		}
		engine, err := protocolcodec.NewEngine(protocolcodec.NewBuiltinRegistry(), policy)
		require.NoError(t, err)
		_, _, _, err = router.prepareLooperResponseWithEngine(response, &RequestContext{}, engine)
		if over {
			require.Error(t, err)
		} else {
			require.NoError(t, err)
		}
	}
	for _, field := range []string{"fusion", "flow", "reasoning_mom_responses"} {
		response.Body = []byte(strings.Replace(stream, `"choices":`, `"`+field+`":{"summary":"ordinary fixture"},"choices":`, 1))
		_, _, _, err := router.prepareLooperResponse(response, &RequestContext{})
		require.Error(t, err)
	}
}

func TestLooperOmissionWarningRemainsVisibleAndBounded(t *testing.T) {
	response := tracedLooperFixture(t, false, false)
	ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIResponsesV1}
	for i := 0; i < 100; i++ {
		ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, llmprotocol.Diagnostic{Action: llmprotocol.DiagnosticDropped, Field: fmt.Sprintf("ordinary_field_%d", i), Reason: "ordinary_provider_field_has_no_neutral_representation"})
	}
	out, _, _, err := (&OpenAIRouter{}).prepareLooperResponse(response, ctx)
	require.NoError(t, err)
	warning := ""
	for _, header := range out.GetImmediateResponse().GetHeaders().GetSetHeaders() {
		if header.GetHeader().GetKey() == headers.VSRProtocolWarnings {
			warning = string(header.GetHeader().GetRawValue())
		}
	}
	require.LessOrEqual(t, len(warning), lossinessHeaderSizeLimit)
	require.True(t, strings.HasPrefix(warning, "dropped;router_extension_unsupported_protocol;fusion"))
	require.Contains(t, warning, "diagnostics_truncated")
}

func TestLooperWorkflowTraceVisibility(t *testing.T) {
	for _, tc := range []struct {
		name                  string
		include, failed, want bool
	}{
		{"enabled", true, false, true}, {"disabled", false, false, false}, {"failure_evidence", false, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			decision := workflowTestDecisionWithIntermediateResponses(tc.include)
			value := json.RawMessage(`{"steps":[]}`)
			if tc.failed {
				value = json.RawMessage(`{"failed_models":[{"model":"worker","error":"unavailable"}]}`)
			}
			require.Equal(t, tc.want, looperShouldRestoreWorkflowTrace(&RequestContext{VSRSelectedDecision: &decision}, value))
		})
	}
}
