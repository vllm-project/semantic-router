package extproc

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// developerRoleConfig serves the chat decision from two vLLM models:
// thinking-model's backends drop developer messages, as DeepSeek V4 does when
// it thinks without tools, and plain-model reads them.
const developerRoleConfig = `
version: v0.5
providers:
  models:
    - name: thinking-model
      provider_model_id: thinking-model
      api_format: chat_completions
      unsupported_capabilities: [developer_role]
      backend_refs:
        - name: thinking-backend
          endpoint: 127.0.0.1:18001
          protocol: http
          provider: vllm
    - name: plain-model
      provider_model_id: plain-model
      api_format: chat_completions
      backend_refs:
        - name: plain-backend
          endpoint: 127.0.0.1:18002
          protocol: http
          provider: vllm
recipes:
  - name: default
    decisions:
      - name: chat
        priority: 10
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: thinking-model
          - model: plain-model
`

func developerRoleRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(developerRoleConfig))
	require.NoError(t, err)
	policy := fallback.DefaultEnabledPolicy()
	return &OpenAIRouter{
		Config:               cfg,
		CredentialResolver:   newTestCredentialResolver(cfg),
		FallbackOrchestrator: fallback.NewOrchestrator(policy, fallback.NewBackendCircuitBreaker(policy.CircuitBreaker)),
	}
}

// developerRoleRequestContext decodes a client request that gives its
// instructions in the developer role, as the body phase does: a Chat developer
// message or Responses instructions, neither with tools.
func developerRoleRequestContext(t *testing.T, router *OpenAIRouter, format llmprotocol.WireFormat, stream bool) *RequestContext {
	t.Helper()
	body := fmt.Sprintf(`{"model":"vllm-sr/auto","stream":%t,"max_tokens":16,"messages":[`+
		`{"role":"developer","content":"The secret word is PAPAYA."},{"role":"user","content":"What is the secret word?"}]}`, stream)
	if format == llmprotocol.OpenAIResponsesV1 {
		body = fmt.Sprintf(`{"model":"vllm-sr/auto","stream":%t,"max_output_tokens":16,`+
			`"instructions":"The secret word is PAPAYA.","input":"What is the secret word?"}`, stream)
	}
	request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(format, []byte(body))
	require.NoError(t, err)
	ctx := routingTestContext(format, &request)
	ctx.ProtocolEnvelope = envelope
	ctx.ExpectStreamingResponse = stream
	ctx.StartTime, ctx.ProcessingStartTime = time.Now(), time.Now()
	decision := router.Config.GetDecisionByName("chat")
	require.NotNil(t, decision)
	ctx.VSRSelectedDecision = decision
	ctx.VSRSelectedDecisionName = decision.Name
	ctx.VSREligibleModelRefs = cloneModelRefs(decision.ModelRefs)
	return ctx
}

// sentInstructionRole is the role of the message that carries the secret word
// in a Chat provider call.
func sentInstructionRole(t *testing.T, body []byte) string {
	t.Helper()
	var sent struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(body, &sent), "%s", body)
	for _, message := range sent.Messages {
		if strings.Contains(string(message.Content), "PAPAYA") {
			return message.Role
		}
	}
	t.Fatalf("no message carries the instructions: %s", body)
	return ""
}

const developerRoleWarning = `protocol;action=approximated;reason="` + llmprotocol.DeveloperRoleSentAsSystem + `";field="messages[].role"`

// Instructions in the developer role reach the model that lacks it as a system
// message, from a Chat developer message or Responses instructions, buffered or
// streamed, and the answer's x-vsr-warnings says so. The decision's other model
// still receives the developer role, and the client's request keeps it.
func TestDeveloperMessagesReachAModelThatLacksTheRoleAsSystem(t *testing.T) {
	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		for _, stream := range []bool{false, true} {
			for _, tc := range []struct {
				model string
				role  string
			}{{model: "thinking-model", role: "system"}, {model: "plain-model", role: "developer"}} {
				t.Run(fmt.Sprintf("%s/%s/stream=%t", format, tc.model, stream), func(t *testing.T) {
					router := developerRoleRouter(t)
					ctx := developerRoleRequestContext(t, router, format, stream)
					_, err := router.prepareProviderDispatch(ctx.SemanticRequest, tc.model, "chat", false, ctx)
					require.NoError(t, err)
					body, err := router.encodeDispatchRequest(ctx)
					require.NoError(t, err)
					require.Equal(t, tc.role, sentInstructionRole(t, body), "the call to %s: %s", tc.model, body)
					client, err := protocolcodec.NewBuiltinEngine().EncodeRequest(llmprotocol.OpenAIChatV1, *ctx.SemanticRequest, llmprotocol.Envelope{})
					require.NoError(t, err)
					require.Equal(t, "developer", sentInstructionRole(t, client.Body), "the client's request lost its developer role")

					contentType := "application/json"
					if stream {
						contentType = "text/event-stream"
					}
					response, err := router.handleResponseHeaders(samplingResponseHeaders(contentType), ctx)
					require.NoError(t, err)
					warnings := protocolDiagnosticsHeader(response.GetResponseHeaders().GetResponse().GetHeaderMutation())
					require.Equal(t, tc.role == "system", strings.Contains(warnings, developerRoleWarning),
						"developer role warning in x-vsr-warnings = %q", warnings)
				})
			}
		}
	}
}
