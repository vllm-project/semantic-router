package extproc

import (
	"encoding/json"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

// Exercise the existing stateless request path with the real catalog and a
// small embedding fixture. No sticky manager, store, or model is constructed.
func TestToolSelectionCatalogOrderAndExactSchemaReachRequest(t *testing.T) {
	for _, names := range [][]string{{"beta", "alpha"}, {"alpha", "beta"}} {
		db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
			Enabled: true, Provider: &stubToolSelectionEmbeddingProvider{},
		})
		for _, name := range names {
			tool := openai.ChatCompletionToolParam{Function: openai.FunctionDefinitionParam{
				Name: name, Parameters: openai.FunctionParameters{
					"type": "object", "properties": map[string]any{
						"n": map[string]any{"minimum": json.Number("9007199254740993")},
					},
				},
			}}
			require.NoError(t, db.AddTool(tool, name, "", nil))
		}
		router := makeToolsRouter(t, nil)
		router.ToolsDatabase = db
		request := testNeutralRequest("model", "query")
		request.ToolChoice.Mode = llmprotocol.ToolChoiceAuto
		response := &ext_proc.ProcessingResponse{}
		requestContext := &RequestContext{VSRSelectedDecision: &config.Decision{
			Name: "catalog-order", Plugins: []config.DecisionPlugin{
				mustToolSelectionDecisionPlugin(t, &config.ToolSelectionPluginConfig{
					Enabled: true, Mode: config.ToolSelectionModeAdd, TopK: 1,
					Sticky: &config.StickyToolSelectionConfig{Enabled: false},
				}),
			},
		}}
		require.NoError(t, router.handleToolSelection(request, "query", nil, &response, requestContext))
		require.Len(t, request.Tools, 1)
		require.Equal(t, "alpha", request.Tools[0].Name)
		require.JSONEq(t, `{"type":"object","properties":{"n":{"minimum":9007199254740993}}}`, string(request.Tools[0].InputSchema))
		// JSONEq uses float64 internally; compare the exact numeral separately.
		require.Contains(t, string(request.Tools[0].InputSchema), "9007199254740993")
	}
}
