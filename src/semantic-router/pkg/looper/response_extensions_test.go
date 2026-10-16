package looper

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestRouterExtensionEmitterContract(t *testing.T) {
	final := &ModelResponse{Model: "fixture", Content: "A useful answer"}
	tools := &ModelResponse{Model: "fixture", HasToolCalls: true, Raw: []byte(`{"id":"c1","object":"chat.completion","created":1,"model":"fixture","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)}
	fusionCfg := fusionExecutionConfig{IncludeIntermediateResponses: true}
	fusionTrace := &FusionTrace{JudgeModel: "fixture", Responses: []FusionPanelResponse{{Model: "panel", Content: "Useful evidence"}}}
	workflowCfg := workflowsExecutionConfig{IncludeIntermediateResponses: true}
	trace := &workflowTrace{Mode: "fixture"}
	rounds := []RoundResponse{{}}
	usage := TokenUsage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2}
	cases := []struct {
		name, field string
		run         func() (*Response, error)
	}{
		{"fusion_text_json", "fusion", func() (*Response, error) {
			return (&FusionLooper{}).formatFusionJSONResponse(final, nil, 1, fusionCfg, fusionTrace, usage)
		}},
		{"fusion_tool_json", "fusion", func() (*Response, error) {
			return (&FusionLooper{}).formatFusionJSONResponse(tools, nil, 1, fusionCfg, fusionTrace, usage)
		}},
		{"fusion_text_sse", "fusion", func() (*Response, error) {
			return (&FusionLooper{}).formatFusionStreamingResponse(final, nil, 1, fusionCfg, fusionTrace, usage)
		}},
		{"fusion_tool_sse", "fusion", func() (*Response, error) {
			return (&FusionLooper{}).formatFusionStreamingResponse(tools, nil, 1, fusionCfg, fusionTrace, usage)
		}},
		{"workflow_text_json", "flow", func() (*Response, error) { return formatWorkflowJSONResponse(final, nil, 1, trace, usage, workflowCfg) }},
		{"workflow_tool_json", "flow", func() (*Response, error) { return formatWorkflowJSONResponse(tools, nil, 1, trace, usage, workflowCfg) }},
		{"workflow_text_sse", "flow", func() (*Response, error) {
			return formatWorkflowStreamingResponse(final, nil, 1, trace, usage, workflowCfg)
		}},
		{"workflow_tool_sse", "flow", func() (*Response, error) {
			return formatWorkflowStreamingResponse(tools, nil, 1, trace, usage, workflowCfg)
		}},
		{"remom_text_json", "reasoning_mom_responses", func() (*Response, error) {
			return (&ReMoMLooper{}).formatReMoMJSONResponse(IntermediateResp{Model: "fixture", Content: "A useful answer"}, rounds, nil, 1, usage, &config.ReMoMAlgorithmConfig{IncludeIntermediateResponses: true})
		}},
		{"remom_text_sse", "reasoning_mom_responses", func() (*Response, error) {
			return (&ReMoMLooper{}).formatReMoMStreamingResponse(IntermediateResp{Model: "fixture", Content: "A useful answer"}, rounds, nil, 1, usage, &config.ReMoMAlgorithmConfig{IncludeIntermediateResponses: true})
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			response, err := tc.run()
			require.NoError(t, err)
			protocolBody := response.ProtocolBody()
			require.NotContains(t, string(protocolBody), `"`+tc.field+`"`)
			require.Contains(t, string(response.Body), `"`+tc.field+`"`, "direct Go caller retains compatibility")
			extensions := response.RouterExtensions()
			require.Len(t, extensions, 1)
			require.Equal(t, tc.field, extensions[0].Name())
			require.True(t, json.Valid(extensions[0].JSON()))
			engine := protocolcodec.NewBuiltinEngine()
			for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
				if response.ContentType == "text/event-stream" {
					stream, streamErr := engine.NewStream(llmprotocol.OpenAIChatV1, target, llmprotocol.StreamContext{})
					require.NoError(t, streamErr)
					frames, _, _, pushErr := stream.Push(protocolBody)
					require.NoError(t, pushErr)
					terminal, _, _, finalErr := stream.Finalize(nil)
					require.NoError(t, finalErr)
					require.NotEmpty(t, append(frames, terminal...))
				} else {
					_, err = engine.TranslateResponse(llmprotocol.OpenAIChatV1, target, protocolBody, nil)
					require.NoError(t, err)
				}
			}
			saved := bytes.Clone(protocolBody)
			savedExtension := extensions[0].JSON()
			response.Body[0] = '!'
			protocolBody[0] = '?'
			extensions[0].value[0] = '?'
			copyOfExtension := response.RouterExtensions()[0].JSON()
			copyOfExtension[0] = '!'
			require.Equal(t, saved, response.ProtocolBody(), "compatibility bytes cannot alter trusted snapshot")
			require.Equal(t, savedExtension, response.RouterExtensions()[0].JSON(), "returned evidence cannot mutate provenance")
		})
	}
}

func TestUntrustedResponseHasNoRouterExtensionChannel(t *testing.T) {
	for _, field := range []string{"fusion", "flow", "reasoning_mom_responses"} {
		t.Run(field, func(t *testing.T) {
			// Ordinary fixed unknown-field input, evaluated only against the corrected
			// boundary: choosing a name never creates emitter-owned provenance.
			body := []byte(`{"id":"c1","object":"chat.completion","model":"fixture","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"` + field + `":{"summary":"fixture"}}`)
			response := &Response{Body: body}
			require.Empty(t, response.RouterExtensions())
			require.Equal(t, body, response.ProtocolBody())
			_, err := protocolcodec.NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, response.ProtocolBody(), nil)
			require.Error(t, err)
		})
	}
}
