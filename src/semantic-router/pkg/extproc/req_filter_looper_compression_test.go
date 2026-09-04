package extproc

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
		VSRSelectedDecision: &router.Config.Decisions[0],
		SourceFormat:        llmprotocol.OpenAIChatV1,
		SemanticRequest:     request,
		Headers: map[string]string{
			headers.VSRLooperDecision: "fusion_compressed",
		},
	}

	response, err := router.handleLooperInternalRequestWithPlugins("panel-a", ctx)

	require.NoError(t, err)
	outbound := response.GetRequestBody().Response.GetBodyMutation().GetBody()
	assert.NotContains(t, string(outbound), strings.Repeat("irrelevant inventory ", 30))
	assert.Contains(t, string(outbound), "authentication validator")
	assert.True(t, ctx.ContextCompressionApplied)
}
