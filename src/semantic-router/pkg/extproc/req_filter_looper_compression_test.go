package extproc

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

func TestHandleLooperInternalRequestWithPluginsCompressesWorkingBody(t *testing.T) {
	decision := *contextCompressionTestDecision(false)
	decision.Name = "fusion_compressed"
	decision.ModelRefs = []config.ModelRef{{Model: "panel-a"}}
	router := &OpenAIRouter{
		Cache: &spyCache{},
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{decision},
			},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"panel-a": {PreferredEndpoints: []string{"panel-backend"}},
				},
				VLLMEndpoints: []config.VLLMEndpoint{{
					Name:    "panel-backend",
					Address: "127.0.0.1",
					Port:    8000,
					Type:    "vllm",
					Weight:  1,
				}},
			},
		},
	}
	router.CredentialResolver = newTestCredentialResolver(router.Config)
	largeTool := strings.Repeat("irrelevant inventory ", 300) +
		"authentication validator failed " +
		strings.Repeat("irrelevant billing ", 300)
	request := semanticCompressionRequest(largeTool)
	request.Model = "panel-a"
	ctx := &RequestContext{
		LooperRequest:       true,
		Hop:                 &routing.Hop{Decision: "fusion_compressed"},
		VSRSelectedDecision: &router.Config.Decisions[0],
		SourceFormat:        llmprotocol.OpenAIChatV1,
		SemanticRequest:     request,
		Headers:             map[string]string{},
	}

	response, err := router.handleLooperInternalRequestWithPlugins("panel-a", ctx)

	require.NoError(t, err)
	outbound := response.GetRequestBody().Response.GetBodyMutation().GetBody()
	assert.NotContains(t, string(outbound), strings.Repeat("irrelevant inventory ", 30))
	assert.Contains(t, string(outbound), "authentication validator")
	assert.True(t, ctx.ContextCompressionApplied)
}

func TestHandleLooperInternalRequestWithPluginsSkipsContextRecoveryFollowup(t *testing.T) {
	decision := *contextCompressionTestDecision(false)
	decision.Name = "fusion_compressed"
	decision.ModelRefs = []config.ModelRef{{Model: "panel-a"}}
	router := &OpenAIRouter{
		Cache: &spyCache{},
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{decision},
			},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"panel-a": {PreferredEndpoints: []string{"panel-backend"}},
				},
				VLLMEndpoints: []config.VLLMEndpoint{{
					Name:    "panel-backend",
					Address: "127.0.0.1",
					Port:    8000,
					Type:    "vllm",
					Weight:  1,
				}},
			},
		},
	}
	router.CredentialResolver = newTestCredentialResolver(router.Config)
	recovered := strings.Repeat("irrelevant inventory ", 300) +
		"authentication validator failed " +
		strings.Repeat("irrelevant billing ", 300)
	request := contextRecoveryFollowupSemanticRequest("panel-a", recovered)
	ctx := &RequestContext{
		LooperRequest:       true,
		Hop:                 &routing.Hop{Decision: "fusion_compressed"},
		VSRSelectedDecision: &router.Config.Decisions[0],
		SourceFormat:        llmprotocol.OpenAIChatV1,
		SemanticRequest:     request,
		Headers:             map[string]string{},
	}

	response, err := router.handleLooperInternalRequestWithPlugins("panel-a", ctx)

	require.NoError(t, err)
	require.Nil(t, response.GetImmediateResponse())
	outbound := response.GetRequestBody().Response.GetBodyMutation().GetBody()
	assert.Contains(t, string(outbound), strings.Repeat("irrelevant inventory ", 30))
	assert.Contains(t, string(outbound), "authentication validator")
	assert.False(t, ctx.ContextCompressionApplied)
}

func contextRecoveryFollowupSemanticRequest(model, recovered string) *llmprotocol.Request {
	request := testNeutralRequest(model, "fix authentication validator")
	request.Tools = []llmprotocol.Tool{
		makeTool("lookup"),
		makeTool(contextcompression.RetrieveToolName),
	}
	request.Messages = append(request.Messages,
		llmprotocol.Message{
			Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentToolCall,
				ToolCall: &llmprotocol.ToolCall{
					ID:        "retrieve-1",
					Name:      contextcompression.RetrieveToolName,
					Arguments: `{"key":"key-1"}`,
				},
			}},
		},
		llmprotocol.Message{
			Role: llmprotocol.RoleTool,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentToolResult,
				ToolResult: &llmprotocol.ToolResult{
					CallID:  "retrieve-1",
					Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: recovered}},
				},
			}},
		},
	)
	return request
}
