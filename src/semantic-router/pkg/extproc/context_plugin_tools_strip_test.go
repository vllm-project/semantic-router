package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// A decision whose tools policy strips tool history still sends the model
// what its context sources kept. The strip removes tool exchanges, so the
// documents and memories go in the system prompt instead, on every backend
// wire format.
func TestContextReachesModelWhenToolsPolicyStripsHistory(t *testing.T) {
	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
		t.Run(string(format), func(t *testing.T) {
			fixture, storeID := contextRouter(t)
			router, model := routingTestRouterForFormat(format)
			router.RuntimeRegistry, router.MemoryStore = fixture.RuntimeRegistry, fixture.MemoryStore
			router.Config.GlobalPlugins, router.Config.Memory = fixture.Config.GlobalPlugins, fixture.Config.Memory

			ctx, request := contextRequestContext(config.ContextPluginConfig{Enabled: true, Sources: []config.ContextSource{
				ragSource(storeID, recallOnly(1)), {Type: config.ContextSourceMemory},
			}})
			maxTokens := int64(256)
			request.Sampling.MaxOutputTokens = &maxTokens
			tools, err := config.NewStructuredPayload(config.ToolsPluginConfig{
				Enabled: true, Policy: &config.ToolsPolicyConfig{Mode: config.ToolsPolicyModeNone, StripToolHistory: true},
			})
			require.NoError(t, err)
			decision := ctx.VSRSelectedDecision
			decision.Plugins = append(decision.Plugins, config.DecisionPlugin{Type: config.DecisionPluginTools, Configuration: tools})
			decision.ModelRefs = []config.ModelRef{{Model: model}}
			ctx.SourceFormat, ctx.RequestID = llmprotocol.OpenAIChatV1, "strip-request"
			ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})

			_, early, err := router.prepareRequestForModelRouting(request, contextQuery, ctx)
			require.NoError(t, err)
			require.Nil(t, early)
			response, err := router.handleModelRouting(request, config.DefaultEntrypointModel, decision.Name, entropy.ReasoningDecision{}, model, ctx)
			require.NoError(t, err)

			body := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
			system := upstreamSystemText(t, format, body)
			require.Contains(t, system, "refunds take 14 business days", "the RAG documents are not in the system message: %s", body)
			require.Contains(t, system, "The user prefers short answers.", "the memories are not in the system message: %s", body)
			require.NotContains(t, string(body), ragContextToolName)
			require.NotContains(t, string(body), memoryContextToolName)
		})
	}
}

// upstreamSystemText is the system text of an encoded upstream request: the
// system role messages of Chat messages or Responses input, or the Messages
// system field.
func upstreamSystemText(t *testing.T, format llmprotocol.WireFormat, body []byte) string {
	t.Helper()
	type message struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	var wire struct {
		Messages []message       `json:"messages"`
		Input    []message       `json:"input"`
		System   json.RawMessage `json:"system"`
	}
	require.NoError(t, json.Unmarshal(body, &wire))
	if format == llmprotocol.AnthropicMessagesV1 {
		return string(wire.System)
	}
	var system []string
	for _, item := range append(wire.Messages, wire.Input...) {
		if item.Role == "system" {
			system = append(system, string(item.Content))
		}
	}
	return strings.Join(system, "\n")
}
