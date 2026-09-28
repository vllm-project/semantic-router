package extproc

import (
	"encoding/json"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

func TestExternalGatewayDispatchConvertsResponsesWithoutOwningBackend(t *testing.T) {
	const model = "openai/gpt-oss-20b"
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			model: {APIFormat: config.APIFormatOpenAI},
		}},
		IntelligentRouting: config.IntelligentRouting{
			ModelSelection: config.ModelSelectionConfig{Enabled: false},
		},
	}}
	request := testNeutralRequest(model, "continue the conversation")
	ctx := routingTestContext(llmprotocol.OpenAIResponsesV1, request)
	ctx.VSRSelectedDecision = &config.Decision{Name: "must-not-run"}
	ctx.VSRSelectedDecisionName = "must-not-run"
	ctx.VSRSelectionMethod = "knn"

	response, err := router.handleModelRouting(request, model, "", entropy.ReasoningDecision{}, "", ctx)
	if err != nil {
		t.Fatalf("handleModelRouting returned error: %v", err)
	}
	assertExternalGatewayDispatchMutation(t, response, model)
	if ctx.TargetFormat != llmprotocol.OpenAIChatV1 {
		t.Fatalf("target format = %q, want %q", ctx.TargetFormat, llmprotocol.OpenAIChatV1)
	}
	if ctx.VSRSelectedDecision != nil || ctx.VSRSelectedDecisionName != "" || ctx.VSRSelectionMethod != "" {
		t.Fatal("external gateway dispatch retained Semantic Router selection state")
	}
}

func assertExternalGatewayDispatchMutation(t *testing.T, response *ext_proc.ProcessingResponse, model string) {
	t.Helper()
	common := response.GetRequestBody().GetResponse()
	if common.GetClearRouteCache() {
		t.Fatal("external gateway dispatch must preserve the gateway-selected route")
	}
	headersByName := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())
	if got := headersByName[":path"]; got != "/v1/chat/completions" {
		t.Fatalf("converted request path = %q, want /v1/chat/completions", got)
	}
	for _, name := range []string{headers.SelectedModel, "authorization", "x-api-key"} {
		if value, ok := headersByName[name]; ok {
			t.Fatalf("external gateway-owned header %q was mutated to %q", name, value)
		}
	}

	var body struct {
		Model    string `json:"model"`
		Messages []struct {
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(common.GetBodyMutation().GetBody(), &body); err != nil {
		t.Fatalf("decode converted request: %v", err)
	}
	if body.Model != model || len(body.Messages) != 1 || body.Messages[0].Content != "continue the conversation" {
		t.Fatalf("unexpected converted request: %+v", body)
	}
}

func TestExternalGatewayDispatchRequiresExplicitExtProcOnlyMetadataModel(t *testing.T) {
	const model = "external-model"
	tests := []struct {
		name string
		cfg  *config.RouterConfig
		want bool
	}{
		{
			name: "metadata-only model with selection disabled",
			cfg: &config.RouterConfig{
				BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{model: {APIFormat: config.APIFormatOpenAI}}},
				IntelligentRouting: config.IntelligentRouting{
					ModelSelection: config.ModelSelectionConfig{Enabled: false},
				},
			},
			want: true,
		},
		{
			name: "selection enabled",
			cfg: &config.RouterConfig{
				BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{model: {APIFormat: config.APIFormatOpenAI}}},
				IntelligentRouting: config.IntelligentRouting{
					ModelSelection: config.ModelSelectionConfig{Enabled: true},
				},
			},
		},
		{
			name: "router listener configured",
			cfg: &config.RouterConfig{
				BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{model: {APIFormat: config.APIFormatOpenAI}}},
				APIServer:     config.APIServer{Listeners: []config.Listener{{Name: "http", Address: "0.0.0.0", Port: 8899}}},
			},
		},
		{name: "model is not declared", cfg: &config.RouterConfig{}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			router := &OpenAIRouter{Config: test.cfg}
			if got := router.usesExternalGatewayDispatch(model); got != test.want {
				t.Fatalf("usesExternalGatewayDispatch() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestExternalGatewayResponsesRouterStartsWithoutModelSelectors(t *testing.T) {
	const model = "openai/gpt-oss-20b"
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			model: {APIFormat: config.APIFormatOpenAI},
		}},
		IntelligentRouting: config.IntelligentRouting{
			ModelSelection: config.ModelSelectionConfig{Enabled: false},
		},
		ResponseAPI: config.ResponseAPIConfig{
			Enabled:      true,
			StoreBackend: "memory",
			TTLSeconds:   86400,
			MaxResponses: 1000,
		},
	}

	components, err := buildRouterComponents(cfg)
	if err != nil {
		t.Fatalf("build state-only router components: %v", err)
	}
	if components.responseAPIFilter == nil {
		t.Fatal("Responses state filter was not initialized")
	}
	if components.modelSelector != nil || len(components.recipeModelSelectors) != 0 || components.lookupTable != nil {
		t.Fatal("state-only router initialized model-selection resources")
	}
	if err := components.buildRouter().Close(); err != nil {
		t.Fatalf("close state-only router: %v", err)
	}
}
