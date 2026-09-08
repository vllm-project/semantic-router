package extproc

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

type catalogReasoningWireCase struct {
	name          string
	catalog       string
	provider      string
	apiFormat     string
	enabled       bool
	mode          string
	effort        string
	wantTransport modelcatalog.ReasoningTransport
	wantControls  map[string]interface{}
}

// TestBuiltInCatalogReasoningWireContracts crosses the complete maintained
// path: catalog model -> provider-model binding -> materialized profile ->
// neutral protocol codec -> final provider request. Fixture-only transport
// tests remain useful, but this table prevents the authored catalog and the
// runtime adapter from drifting independently.
func TestBuiltInCatalogReasoningWireContracts(t *testing.T) {
	tests := []catalogReasoningWireCase{
		{
			name: "OpenAI Astra Chat xhigh effort", catalog: "openai/gpt-6-astra", provider: "openai",
			enabled: true, effort: "xhigh", wantTransport: modelcatalog.ReasoningTransportTopLevelEffort,
			wantControls: map[string]interface{}{"reasoning_effort": "xhigh"},
		},
		{
			name: "OpenAI Astra Responses max effort", catalog: "openai/gpt-6-astra", provider: "openai",
			apiFormat: config.APIFormatResponses, enabled: true, effort: "max",
			wantTransport: modelcatalog.ReasoningTransportTopLevelEffort,
			wantControls:  map[string]interface{}{"reasoning": map[string]interface{}{"effort": "max"}},
		},
		{
			name: "OpenAI Chat effort", catalog: "openai/gpt-5.6-sol", provider: "openai",
			enabled: true, effort: "high", wantTransport: modelcatalog.ReasoningTransportTopLevelEffort,
			wantControls: map[string]interface{}{"reasoning_effort": "high"},
		},
		{
			name: "OpenAI Responses disabled", catalog: "openai/gpt-5.6-sol", provider: "openai",
			apiFormat: config.APIFormatResponses, enabled: false, mode: config.ReasoningModeDisabled,
			wantTransport: modelcatalog.ReasoningTransportTopLevelEffort,
			wantControls:  map[string]interface{}{"reasoning": map[string]interface{}{"effort": "none"}},
		},
		{
			name: "DeepSeek Chat thinking and effort", catalog: "deepseek/deepseek-v4-pro", provider: "deepseek",
			enabled: true, effort: "max", wantTransport: modelcatalog.ReasoningTransportDeepSeekThinking,
			wantControls: map[string]interface{}{
				"thinking": map[string]interface{}{"type": "enabled"}, "reasoning_effort": "max",
			},
		},
		{
			name: "DeepSeek Responses effort", catalog: "deepseek/deepseek-v4-pro", provider: "deepseek",
			apiFormat: config.APIFormatResponses, enabled: true, effort: "max",
			wantTransport: modelcatalog.ReasoningTransportDeepSeekThinking,
			wantControls:  map[string]interface{}{"reasoning": map[string]interface{}{"effort": "max"}},
		},
		{
			name: "Anthropic adaptive thinking and effort", catalog: "anthropic/claude-fable-5", provider: "anthropic",
			enabled: true, mode: config.ReasoningModeAdaptive, effort: "xhigh",
			wantTransport: modelcatalog.ReasoningTransportOutputConfig,
			wantControls: map[string]interface{}{
				"thinking":      map[string]interface{}{"type": "adaptive"},
				"output_config": map[string]interface{}{"effort": "xhigh"},
			},
		},
		{
			name: "Anthropic thinking disabled", catalog: "anthropic/claude-opus-4.8", provider: "anthropic",
			enabled: false, mode: config.ReasoningModeDisabled,
			wantTransport: modelcatalog.ReasoningTransportOutputConfig,
			wantControls:  map[string]interface{}{"thinking": map[string]interface{}{"type": "disabled"}},
		},
		{
			name: "Z.ai GLM switch disabled", catalog: "zai/glm-5.1", provider: "zai",
			enabled: false, mode: config.ReasoningModeDisabled,
			wantTransport: modelcatalog.ReasoningTransportThinkingObject,
			wantControls:  map[string]interface{}{"thinking": map[string]interface{}{"type": "disabled"}},
		},
		{
			name: "Z.ai GLM switch and effort", catalog: "zai/glm-5.2", provider: "zai",
			enabled: true, mode: config.ReasoningModeEnabled, effort: "high",
			wantTransport: modelcatalog.ReasoningTransportThinkingEffort,
			wantControls: map[string]interface{}{
				"thinking": map[string]interface{}{"type": "enabled"}, "reasoning_effort": "high",
			},
		},
	}
	runCatalogReasoningWireCases(t, tests)
}

func TestBuiltInCatalogLocalRuntimeReasoningWireContracts(t *testing.T) {
	tests := []catalogReasoningWireCase{
		{
			name: "vLLM Qwen Flash Next top-level effort and template switch", catalog: "qwen/qwen3.8-flash-next", provider: "vllm",
			enabled: true, mode: config.ReasoningModeEnabled, effort: "xhigh",
			wantTransport: modelcatalog.ReasoningTransportEffortTemplateSwitch,
			wantControls: map[string]interface{}{
				"reasoning_effort":     "xhigh",
				"chat_template_kwargs": map[string]interface{}{"enable_thinking": true},
			},
		},
		{
			name: "vLLM Qwen Flash Next template switch disabled", catalog: "qwen/qwen3.8-flash-next", provider: "vllm",
			enabled: false, mode: config.ReasoningModeDisabled,
			wantTransport: modelcatalog.ReasoningTransportEffortTemplateSwitch,
			wantControls: map[string]interface{}{"chat_template_kwargs": map[string]interface{}{
				"enable_thinking": false,
			}},
		},
		{
			name: "vLLM Qwen Flash Next Responses uses protocol-native reasoning object", catalog: "qwen/qwen3.8-flash-next", provider: "vllm",
			apiFormat: config.APIFormatResponses, enabled: true, mode: config.ReasoningModeEnabled, effort: "medium",
			wantTransport: modelcatalog.ReasoningTransportEffortTemplateSwitch,
			wantControls:  map[string]interface{}{"reasoning": map[string]interface{}{"effort": "medium"}},
		},
		{
			name: "SGLang Qwen top-level effort and template switch", catalog: "qwen/qwen3.8-27b", provider: "sglang",
			enabled: true, mode: config.ReasoningModeEnabled, effort: "low",
			wantTransport: modelcatalog.ReasoningTransportEffortTemplateSwitch,
			wantControls: map[string]interface{}{
				"reasoning_effort":     "low",
				"chat_template_kwargs": map[string]interface{}{"enable_thinking": true},
			},
		},
		{
			name: "DashScope Qwen top-level effort and switch", catalog: "qwen/qwen3.8-max", provider: "dashscope",
			enabled: true, mode: config.ReasoningModeEnabled, effort: "medium",
			wantTransport: modelcatalog.ReasoningTransportEffortBooleanSwitch,
			wantControls:  map[string]interface{}{"reasoning_effort": "medium", "enable_thinking": true},
		},
		{
			name: "DashScope Qwen switch disabled", catalog: "qwen/qwen3.8-max", provider: "dashscope",
			enabled: false, mode: config.ReasoningModeDisabled,
			wantTransport: modelcatalog.ReasoningTransportEffortBooleanSwitch,
			wantControls:  map[string]interface{}{"enable_thinking": false},
		},
		{
			name: "DashScope Qwen Responses uses protocol-native reasoning object", catalog: "qwen/qwen3.8-max", provider: "dashscope",
			apiFormat: config.APIFormatResponses, enabled: true, mode: config.ReasoningModeEnabled, effort: "xhigh",
			wantTransport: modelcatalog.ReasoningTransportEffortBooleanSwitch,
			wantControls:  map[string]interface{}{"reasoning": map[string]interface{}{"effort": "xhigh"}},
		},
		{
			name: "vLLM Hunyuan native disabled sentinel", catalog: "tencent/hy3", provider: "vllm",
			apiFormat: config.APIFormatResponses, enabled: false, mode: config.ReasoningModeDisabled,
			wantTransport: modelcatalog.ReasoningTransportChatTemplate,
			wantControls: map[string]interface{}{"chat_template_kwargs": map[string]interface{}{
				"reasoning_effort": "no_think",
			}},
		},
	}
	runCatalogReasoningWireCases(t, tests)
}

func TestBuiltInCatalogSpecializedReasoningWireContracts(t *testing.T) {
	tests := []catalogReasoningWireCase{
		{
			name: "MiniMax adaptive mode", catalog: "minimax/minimax-m3", provider: "minimax",
			enabled: true, mode: config.ReasoningModeAdaptive,
			wantTransport: modelcatalog.ReasoningTransportThinkingObject,
			wantControls:  map[string]interface{}{"thinking": map[string]interface{}{"type": "adaptive"}},
		},
		{
			name: "MiniMax enabled mode", catalog: "minimax/minimax-m3", provider: "minimax",
			enabled: true, mode: config.ReasoningModeEnabled,
			wantTransport: modelcatalog.ReasoningTransportThinkingObject,
			wantControls:  map[string]interface{}{"thinking": map[string]interface{}{"type": "enabled"}},
		},
		{
			name: "Nemotron Super low effort flag", catalog: "nvidia/nemotron-3-super", provider: "vllm",
			enabled: true, mode: config.ReasoningModeEnabled, effort: "low",
			wantTransport: modelcatalog.ReasoningTransportChatTemplate,
			wantControls: map[string]interface{}{"chat_template_kwargs": map[string]interface{}{
				"enable_thinking": true, "low_effort": true,
			}},
		},
		{
			name: "Nemotron Super full effort omits low flag", catalog: "nvidia/nemotron-3-super", provider: "vllm",
			enabled: true, mode: config.ReasoningModeEnabled, effort: "high",
			wantTransport: modelcatalog.ReasoningTransportChatTemplate,
			wantControls: map[string]interface{}{"chat_template_kwargs": map[string]interface{}{
				"enable_thinking": true,
			}},
		},
		{
			name: "Nemotron Ultra medium effort flag", catalog: "nvidia/nemotron-3-ultra", provider: "sglang",
			enabled: true, mode: config.ReasoningModeEnabled, effort: "medium",
			wantTransport: modelcatalog.ReasoningTransportChatTemplate,
			wantControls: map[string]interface{}{"chat_template_kwargs": map[string]interface{}{
				"enable_thinking": true, "medium_effort": true,
			}},
		},
		{
			name: "Nemotron Cascade switch disabled", catalog: "nvidia/nemotron-cascade-2-30b-a3b", provider: "vllm",
			enabled: false, mode: config.ReasoningModeDisabled,
			wantTransport: modelcatalog.ReasoningTransportChatTemplate,
			wantControls: map[string]interface{}{"chat_template_kwargs": map[string]interface{}{
				"enable_thinking": false,
			}},
		},
		{
			name: "CompactifAI Nemotron switch", catalog: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", provider: "compactifai",
			enabled: true, mode: config.ReasoningModeEnabled,
			wantTransport: modelcatalog.ReasoningTransportChatTemplate,
			wantControls: map[string]interface{}{"chat_template_kwargs": map[string]interface{}{
				"enable_thinking": true,
			}},
		},
		{
			name: "Novita top-level switch", catalog: "xiaomi/mimo-v2-flash", provider: "novita",
			enabled: false, mode: config.ReasoningModeDisabled,
			wantTransport: modelcatalog.ReasoningTransportTopLevelBoolean,
			wantControls:  map[string]interface{}{"enable_thinking": false},
		},
		{
			name: "OpenRouter normalized reasoning object", catalog: "qwen/qwen3.8-27b", provider: "openrouter",
			enabled: true, mode: config.ReasoningModeEnabled, effort: "medium",
			wantTransport: modelcatalog.ReasoningTransportReasoningObject,
			wantControls:  map[string]interface{}{"reasoning": map[string]interface{}{"effort": "medium"}},
		},
	}
	runCatalogReasoningWireCases(t, tests)
}

func runCatalogReasoningWireCases(t *testing.T, tests []catalogReasoningWireCase) {
	t.Helper()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request, profile := renderCatalogReasoningWire(t, test.catalog, test.provider, test.apiFormat, test.enabled, test.mode, test.effort)
			transport, err := profile.ResolveReasoningTransport()
			require.NoError(t, err)
			assert.Equal(t, test.wantTransport, transport)
			assert.Equal(t, test.wantControls, providerReasoningControls(request))
		})
	}
}

func renderCatalogReasoningWire(
	t *testing.T,
	catalogID string,
	providerID string,
	apiFormat string,
	enabled bool,
	mode string,
	effort string,
) (map[string]interface{}, config.ProviderProfile) {
	t.Helper()
	apiFormatLine := ""
	if apiFormat != "" {
		apiFormatLine = fmt.Sprintf("      api_format: %s\n", apiFormat)
	}
	cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(`
version: v0.3
providers:
  models:
    - name: routed
      catalog: %s
%s      backend_refs:
        - name: primary
          provider: %s
          endpoint: 127.0.0.1:8000
          protocol: http
routing: {}
`, catalogID, apiFormatLine, providerID)))
	require.NoError(t, err)

	decision := config.Decision{Name: "route", ModelRefs: []config.ModelRef{{
		Model: "routed",
		ModelReasoningControl: config.ModelReasoningControl{
			UseReasoning: boolPtr(enabled), ReasoningMode: mode, ReasoningEffort: effort,
		},
	}}}
	router := &OpenAIRouter{Config: cfg}
	target, err := wireFormatForModel(cfg.GetModelAPIFormat("routed"))
	require.NoError(t, err)
	request := llmprotocol.Request{
		Generation: 1,
		Model:      "routed",
		Messages: []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hello"}},
		}},
	}
	if target != llmprotocol.OpenAIChatV1 {
		router.applySemanticReasoningMode(&request, "routed", target, enabled, &decision)
	}
	encoded, err := protocolcodec.NewBuiltinEngine().EncodeRequest(target, request, llmprotocol.Envelope{})
	require.NoError(t, err)

	params := cfg.ModelConfig["routed"]
	require.Len(t, params.PreferredEndpoints, 1)
	profile, ok := cfg.ProviderProfiles[params.PreferredEndpoints[0]]
	require.True(t, ok)
	adapted, err := router.adaptProviderRequest(
		encoded.Body,
		&providerDispatch{
			logicalModel: "routed", targetFormat: target, decisionName: decision.Name,
			useReasoning: enabled, profile: &profile,
		},
		&RequestContext{VSRSelectedDecision: &decision},
	)
	require.NoError(t, err)
	return unmarshalReasoningRequest(t, adapted), profile
}

func providerReasoningControls(request map[string]interface{}) map[string]interface{} {
	controls := map[string]interface{}{}
	for _, field := range []string{
		"reasoning", "reasoning_effort", "thinking", "thinking_mode",
		"enable_thinking", "chat_template_kwargs", "output_config",
	} {
		if value, ok := request[field]; ok {
			controls[field] = value
		}
	}
	return controls
}
