package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

const twoToolChatBody = `{
	"model": "MoM",
	"messages": [{"role": "user", "content": "What's the weather?"}],
	"tool_choice": "auto",
	"tools": [
		{"type": "function", "function": {"name": "lookup_weather"}},
		{"type": "function", "function": {"name": "noise_tool"}}
	]
}`

func filteredToolsDecision(t *testing.T, allow []string) *config.Decision {
	t.Helper()
	semanticOff := false
	payload, err := config.NewStructuredPayload(config.ToolsPluginConfig{
		Enabled:           true,
		Mode:              config.ToolsPluginModeFiltered,
		AllowTools:        allow,
		SemanticSelection: &semanticOff,
	})
	require.NoError(t, err)
	return &config.Decision{
		Name: "tools_route",
		Plugins: []config.DecisionPlugin{
			{Type: config.DecisionPluginTools, Configuration: payload},
		},
	}
}

func anthropicDispatchRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"claude-sonnet-4.6": {APIFormat: config.APIFormatAnthropic},
				},
			},
		},
		CredentialResolver: authz.NewCredentialResolver(
			authz.NewHeaderInjectionProvider(map[string]string{
				string(authz.ProviderAnthropic): "x-user-anthropic-key",
			}),
		),
	}
	router.CredentialResolver.SetFailOpen(true)
	return router
}

func jsonStringSlice(body []byte, path string) []string {
	var names []string
	gjson.GetBytes(body, path).ForEach(func(_, v gjson.Result) bool {
		if v.String() != "" {
			names = append(names, v.String())
		}
		return true
	})
	return names
}

func TestHandleModelRouting_AnthropicAppliesToolSelectionOnce(t *testing.T) {
	router := anthropicDispatchRouter(t)
	req, err := parseOpenAIRequest([]byte(twoToolChatBody))
	require.NoError(t, err)
	ctx := &RequestContext{
		Headers:             map[string]string{},
		OriginalRequestBody: []byte(twoToolChatBody),
		VSRSelectedDecision: filteredToolsDecision(t, []string{"lookup_weather"}),
	}

	resp, err := router.handleModelRouting(req, "claude-sonnet-4.6", "tools_route", entropy.ReasoningDecision{}, "", ctx)
	require.NoError(t, err)
	body := resp.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	assert.Equal(t, []string{"lookup_weather"}, jsonStringSlice(body, "tools.#.name"))
}

func TestHandleModelRouting_AnthropicAutoRoutingAppliesToolSelection(t *testing.T) {
	router := anthropicDispatchRouter(t)
	req, err := parseOpenAIRequest([]byte(twoToolChatBody))
	require.NoError(t, err)
	ctx := &RequestContext{
		Headers:             map[string]string{},
		OriginalRequestBody: []byte(twoToolChatBody),
		VSRSelectedDecision: filteredToolsDecision(t, []string{"lookup_weather"}),
	}

	resp, err := router.handleModelRouting(req, "MoM", "tools_route", entropy.ReasoningDecision{}, "claude-sonnet-4.6", ctx)
	require.NoError(t, err)
	body := resp.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	assert.Equal(t, []string{"lookup_weather"}, jsonStringSlice(body, "tools.#.name"))
}

func TestHandleModelRouting_AnthropicStreamingAppliesToolSelection(t *testing.T) {
	router := anthropicDispatchRouter(t)
	req, err := parseOpenAIRequest([]byte(twoToolChatBody))
	require.NoError(t, err)
	ctx := &RequestContext{
		Headers:                 map[string]string{},
		OriginalRequestBody:     []byte(twoToolChatBody),
		ExpectStreamingResponse: true,
		VSRSelectedDecision:     filteredToolsDecision(t, []string{"lookup_weather"}),
	}

	resp, err := router.handleModelRouting(req, "claude-sonnet-4.6", "tools_route", entropy.ReasoningDecision{}, "", ctx)
	require.NoError(t, err)
	body := resp.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	assert.Equal(t, []string{"lookup_weather"}, jsonStringSlice(body, "tools.#.name"))
	assert.True(t, gjson.GetBytes(body, "stream").Bool())
}

func passthroughToolsDecision(t *testing.T) *config.Decision {
	t.Helper()
	semanticOff := false
	payload, err := config.NewStructuredPayload(config.ToolsPluginConfig{
		Enabled:           true,
		Mode:              config.ToolsPluginModePassthrough,
		SemanticSelection: &semanticOff,
	})
	require.NoError(t, err)
	return &config.Decision{
		Name: "tools_route",
		Plugins: []config.DecisionPlugin{
			{Type: config.DecisionPluginTools, Configuration: payload},
		},
	}
}

func TestHandleModelRouting_AnthropicExplicitToolModesKeepRequestTools(t *testing.T) {
	router := anthropicDispatchRouter(t)
	tests := []struct {
		name       string
		toolChoice string
	}{
		{name: "none", toolChoice: `"none"`},
		{name: "required", toolChoice: `"required"`},
		{name: "named", toolChoice: `{"type":"function","function":{"name":"noise_tool"}}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := []byte(`{
				"model": "claude-sonnet-4.6",
				"messages": [{"role": "user", "content": "What's the weather?"}],
				"tool_choice": ` + tt.toolChoice + `,
				"tools": [
					{"type": "function", "function": {"name": "lookup_weather"}},
					{"type": "function", "function": {"name": "noise_tool"}}
				]
			}`)
			req, err := parseOpenAIRequest(raw)
			require.NoError(t, err)
			ctx := &RequestContext{
				Headers:             map[string]string{},
				OriginalRequestBody: raw,
				VSRSelectedDecision: passthroughToolsDecision(t),
			}
			resp, err := router.handleModelRouting(req, "claude-sonnet-4.6", "tools_route", entropy.ReasoningDecision{}, "", ctx)
			require.NoError(t, err)
			got := jsonStringSlice(resp.GetRequestBody().GetResponse().GetBodyMutation().GetBody(), "tools.#.name")
			assert.Equal(t, []string{"lookup_weather", "noise_tool"}, got)
		})
	}
}

func TestHandleModelRouting_OpenAISpecifiedKeepsSelectedTools(t *testing.T) {
	semanticOff := false
	payload, err := config.NewStructuredPayload(config.ToolsPluginConfig{
		Enabled:           true,
		Mode:              config.ToolsPluginModeFiltered,
		AllowTools:        []string{"lookup_weather"},
		SemanticSelection: &semanticOff,
	})
	require.NoError(t, err)
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"known-model": {PreferredEndpoints: []string{"local_vllm"}},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{Name: "local_vllm", Address: "127.0.0.1", Port: 8000, Type: "vllm", Weight: 1},
			},
		},
	}
	router := &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: buildDefaultCredentialResolver(cfg, true),
	}
	req, err := parseOpenAIRequest([]byte(twoToolChatBody))
	require.NoError(t, err)
	ctx := &RequestContext{
		Headers:             map[string]string{},
		OriginalRequestBody: []byte(twoToolChatBody),
		VSRSelectedDecision: &config.Decision{
			Name: "tools_route",
			Plugins: []config.DecisionPlugin{
				{Type: config.DecisionPluginTools, Configuration: payload},
			},
		},
	}

	resp, err := router.handleModelRouting(req, "known-model", "tools_route", entropy.ReasoningDecision{}, "", ctx)
	require.NoError(t, err)
	body := resp.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	assert.Equal(t, []string{"lookup_weather"}, jsonStringSlice(body, "tools.#.function.name"))
}
