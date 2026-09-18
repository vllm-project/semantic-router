package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestResponsesLocalReasoningUsesOneWireControl(t *testing.T) {
	router := newReasoningRouter(
		config.ReasoningConfig{ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
			"local-effort": {
				Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
				Levels: []string{"low", "high"}, Default: "high",
				Modes: []string{config.ReasoningModeEnabled}, DefaultMode: config.ReasoningModeEnabled,
			},
		}},
		[]config.Decision{reasoningDecision("route", "", 0, "local-model", boolPtr(true), "high")},
		map[string]config.ModelParams{"local-model": {ReasoningFamily: "local-effort"}},
	)
	decision := router.Config.GetDecisionByName("route")
	body := []byte(`{
		"model":"local-model",
		"input":"hello",
		"reasoning":{"effort":"high","summary":"auto"},
		"output_config":{"effort":"low","format":{"type":"json_schema"}}
	}`)

	adapted, err := router.adaptProviderRequest(
		body,
		&providerDispatch{
			logicalModel: "local-model", targetFormat: llmprotocol.OpenAIResponsesV1,
			decisionName: "route", useReasoning: true,
			profile: &config.ProviderProfile{ReasoningTransport: modelcatalog.ReasoningTransportChatTemplate},
		},
		&RequestContext{VSRSelectedDecision: decision},
	)
	require.NoError(t, err)
	request := unmarshalReasoningRequest(t, adapted)

	assertChatTemplateReasoningField(t, request, "reasoning_effort", "high")
	_, hasReasoning := request["reasoning"]
	assert.False(t, hasReasoning)
	_, hasTopLevelEffort := request["reasoning_effort"]
	assert.False(t, hasTopLevelEffort)
	_, hasThinking := request["thinking"]
	assert.False(t, hasThinking)
	outputConfig, ok := request["output_config"].(map[string]interface{})
	require.True(t, ok)
	assert.NotNil(t, outputConfig["format"])
	_, hasOutputEffort := outputConfig["effort"]
	assert.False(t, hasOutputEffort)
}

func TestCustomModelWithoutReasoningFamilyPreservesRequest(t *testing.T) {
	router := newReasoningRouter(
		config.ReasoningConfig{},
		[]config.Decision{reasoningDecision("route", "", 0, "custom-model", boolPtr(true), "vendor-ultra")},
		map[string]config.ModelParams{"custom-model": {}},
	)
	body := []byte(`{"model":"custom-model","messages":[],"reasoning_effort":"vendor-ultra","thinking":{"type":"custom"}}`)

	adapted, err := router.setReasoningModeToRequestBodyForModelAndProvider(
		body,
		"custom-model",
		true,
		router.Config.GetDecisionByName("route"),
		&config.ProviderProfile{Type: "custom"},
	)
	require.NoError(t, err)
	assert.Equal(t, body, adapted)
}

func TestChatTemplateEffortFlagsReplaceOnlyManagedControls(t *testing.T) {
	family := config.ReasoningFamilyConfig{
		Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
		ActivationParameter: "enable_thinking", EffortFlags: map[string]string{
			"low": "low_effort", "medium": "medium_effort",
		},
		Levels: []string{"low", "medium", "high"}, Default: "high",
		Modes: []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled}, DefaultMode: config.ReasoningModeEnabled,
	}
	router := newReasoningRouter(
		config.ReasoningConfig{ReasoningFamilies: map[string]config.ReasoningFamilyConfig{"nemotron": family}},
		[]config.Decision{reasoningDecision("route", "", 0, "nemotron", boolPtr(true), "medium")},
		map[string]config.ModelParams{"nemotron": {ReasoningFamily: "nemotron"}},
	)
	body := []byte(`{
		"model":"nemotron",
		"messages":[],
		"reasoning":{"effort":"low"},
		"reasoning_effort":"low",
		"thinking_token_budget":17408,
		"chat_template_kwargs":{
			"enable_thinking":false,
			"low_effort":true,
			"medium_effort":false,
			"reasoning_budget":16384,
			"force_nonempty_content":true
		}
	}`)

	adapted, err := router.setReasoningModeToRequestBodyForModelAndProvider(
		body, "nemotron", true, router.Config.GetDecisionByName("route"),
		&config.ProviderProfile{Type: "vllm"},
	)
	require.NoError(t, err)
	request := unmarshalReasoningRequest(t, adapted)
	assert.Equal(t, float64(17408), request["thinking_token_budget"])
	kwargs := request["chat_template_kwargs"].(map[string]interface{})
	assert.Equal(t, true, kwargs["enable_thinking"])
	assert.Equal(t, true, kwargs["medium_effort"])
	assert.Equal(t, float64(16384), kwargs["reasoning_budget"])
	assert.Equal(t, true, kwargs["force_nonempty_content"])
	_, hasLow := kwargs["low_effort"]
	assert.False(t, hasLow)
	_, hasStringEffort := kwargs["reasoning_effort"]
	assert.False(t, hasStringEffort)

	disabled, err := router.setReasoningModeToRequestBodyForModelAndProvider(
		adapted, "nemotron", false, router.Config.GetDecisionByName("route"),
		&config.ProviderProfile{Type: "vllm"},
	)
	require.NoError(t, err)
	request = unmarshalReasoningRequest(t, disabled)
	kwargs = request["chat_template_kwargs"].(map[string]interface{})
	assert.Equal(t, false, kwargs["enable_thinking"])
	assert.Equal(t, float64(16384), kwargs["reasoning_budget"])
	assert.Equal(t, true, kwargs["force_nonempty_content"])
	_, hasMedium := kwargs["medium_effort"]
	assert.False(t, hasMedium)
}

func TestProviderReasoningTransportRemovesCompetingWireControls(t *testing.T) {
	router := newReasoningRouter(
		config.ReasoningConfig{ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
			"mode": {
				Type: config.ReasoningFamilyTypeReasoningMode, Parameter: "thinking_mode",
				Modes:       []string{config.ReasoningModeAdaptive, config.ReasoningModeDisabled},
				DefaultMode: config.ReasoningModeAdaptive,
			},
		}},
		[]config.Decision{{
			Name: "route",
			ModelRefs: []config.ModelRef{{
				Model: "mode-model",
				ModelReasoningControl: config.ModelReasoningControl{
					UseReasoning: boolPtr(true), ReasoningMode: config.ReasoningModeAdaptive,
				},
			}},
		}},
		map[string]config.ModelParams{"mode-model": {ReasoningFamily: "mode"}},
	)
	body, err := json.Marshal(map[string]interface{}{
		"model": "mode-model", "messages": []map[string]string{{"role": "user", "content": "hello"}},
		"reasoning":            map[string]interface{}{"effort": "high"},
		"reasoning_effort":     "high",
		"thinking_mode":        "enabled",
		"chat_template_kwargs": map[string]interface{}{"thinking_mode": "enabled"},
		"output_config":        map[string]interface{}{"effort": "high", "format": map[string]interface{}{"type": "json_schema"}},
	})
	require.NoError(t, err)

	adapted, err := router.setReasoningModeToRequestBodyForModelAndProvider(
		body,
		"mode-model",
		true,
		router.Config.GetDecisionByName("route"),
		&config.ProviderProfile{ReasoningTransport: modelcatalog.ReasoningTransportThinkingObject},
	)
	require.NoError(t, err)
	request := unmarshalReasoningRequest(t, adapted)
	assertThinkingObjectReasoningRequest(t, request, config.ReasoningModeAdaptive)
	_, hasReasoning := request["reasoning"]
	assert.False(t, hasReasoning)
	_, hasMode := request["thinking_mode"]
	assert.False(t, hasMode)
	outputConfig, ok := request["output_config"].(map[string]interface{})
	require.True(t, ok)
	assert.NotNil(t, outputConfig["format"])
	_, hasEffort := outputConfig["effort"]
	assert.False(t, hasEffort)
}

func TestLocalProviderReasoningWireContracts(t *testing.T) {
	t.Run("local switch uses chat template kwargs", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeChatTemplateKwargs, Parameter: "enable_thinking",
				Modes:       []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled},
				DefaultMode: config.ReasoningModeEnabled,
			},
			target:  llmprotocol.OpenAIChatV1,
			profile: config.ProviderProfile{Type: "vllm"},
			enabled: true,
		})
		assertChatTemplateReasoningField(t, request, "enable_thinking", true)
		assertNoProviderReasoningFields(t, request)
	})

	t.Run("local effort and switch stay together in chat template kwargs", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
				ActivationParameter: "enable_thinking", Levels: []string{"low", "high"}, Default: "high",
				Modes:       []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled},
				DefaultMode: config.ReasoningModeEnabled,
			},
			effort: "low", target: llmprotocol.OpenAIChatV1,
			profile: config.ProviderProfile{Type: "vllm"}, enabled: true,
		})
		assertChatTemplateReasoningField(t, request, "enable_thinking", true)
		assertChatTemplateReasoningField(t, request, "reasoning_effort", "low")
		assertNoProviderReasoningFields(t, request)
	})

	t.Run("local effort flag selects a non-default template profile", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
				ActivationParameter: "enable_thinking", EffortFlags: map[string]string{"low": "low_effort"},
				Levels: []string{"low", "high"}, Default: "high",
				Modes: []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled}, DefaultMode: config.ReasoningModeEnabled,
			},
			effort: "low", target: llmprotocol.OpenAIChatV1,
			profile: config.ProviderProfile{Type: "vllm"}, enabled: true,
		})
		assertChatTemplateReasoningField(t, request, "enable_thinking", true)
		assertChatTemplateReasoningField(t, request, "low_effort", true)
		kwargs := request["chat_template_kwargs"].(map[string]interface{})
		_, hasStringEffort := kwargs["reasoning_effort"]
		assert.False(t, hasStringEffort)
	})

	t.Run("local default effort is encoded by flag omission", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
				ActivationParameter: "enable_thinking", EffortFlags: map[string]string{"medium": "medium_effort"},
				Levels: []string{"medium", "high"}, Default: "high",
				Modes: []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled}, DefaultMode: config.ReasoningModeEnabled,
			},
			effort: "high", target: llmprotocol.OpenAIResponsesV1,
			profile: config.ProviderProfile{Type: "sglang"}, enabled: true,
		})
		assertChatTemplateReasoningField(t, request, "enable_thinking", true)
		kwargs := request["chat_template_kwargs"].(map[string]interface{})
		_, hasMediumFlag := kwargs["medium_effort"]
		assert.False(t, hasMediumFlag)
		_, hasStringEffort := kwargs["reasoning_effort"]
		assert.False(t, hasStringEffort)
		_, hasReasoningObject := request["reasoning"]
		assert.False(t, hasReasoningObject)
	})

	t.Run("local adaptive mode stays in chat template kwargs", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeReasoningMode, Parameter: "thinking_mode",
				Modes:       []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled, config.ReasoningModeAdaptive},
				DefaultMode: config.ReasoningModeAdaptive,
			},
			mode: config.ReasoningModeAdaptive, target: llmprotocol.OpenAIChatV1,
			profile: config.ProviderProfile{Type: "vllm"}, enabled: true,
		})
		assertChatTemplateReasoningField(t, request, "thinking_mode", config.ReasoningModeAdaptive)
		assertNoProviderReasoningFields(t, request)
	})

	t.Run("local responses translates a native disabled sentinel after neutral encoding", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
				Levels: []string{"low", "high"}, Default: "high", Disabled: "no_think",
				Modes:       []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled},
				DefaultMode: config.ReasoningModeDisabled,
			},
			target:  llmprotocol.OpenAIResponsesV1,
			profile: config.ProviderProfile{Type: "vllm"}, enabled: false,
		})
		assertChatTemplateReasoningField(t, request, "reasoning_effort", "no_think")
		assertNoProviderReasoningFields(t, request)
	})
}

func TestOpenAIAndDeepSeekReasoningWireContracts(t *testing.T) {
	t.Run("openai chat uses top level effort", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family: effortReasoningFamily([]string{"low", "medium", "high"}, "medium", true),
			effort: "high", target: llmprotocol.OpenAIChatV1,
			profile: config.ProviderProfile{Type: "openai"}, enabled: true,
		})
		assert.Equal(t, "high", request["reasoning_effort"])
		assertNoReasoningObjectsOrTemplate(t, request)
	})

	t.Run("openai responses uses the standard reasoning object", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family: effortReasoningFamily([]string{"low", "medium", "high"}, "medium", true),
			effort: "high", target: llmprotocol.OpenAIResponsesV1,
			profile: config.ProviderProfile{Type: "openai"}, enabled: true,
		})
		assertReasoningObjectEffort(t, request, "high")
		assertNoTopLevelOrTemplateReasoning(t, request)
	})

	t.Run("openai responses disables reasoning with the standard none effort", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family:  effortReasoningFamily([]string{"low", "medium", "high"}, "medium", true),
			target:  llmprotocol.OpenAIResponsesV1,
			profile: config.ProviderProfile{Type: "openai"}, enabled: false,
		})
		assertReasoningObjectEffort(t, request, "none")
		assertNoTopLevelOrTemplateReasoning(t, request)
	})

	t.Run("openai disables reasoning with its none effort sentinel", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family:  effortReasoningFamily([]string{"low", "medium", "high"}, "medium", true),
			target:  llmprotocol.OpenAIChatV1,
			profile: config.ProviderProfile{Type: "openai"}, enabled: false,
		})
		assert.Equal(t, "none", request["reasoning_effort"])
		assertNoReasoningObjectsOrTemplate(t, request)
	})

	t.Run("deepseek chat combines thinking switch and top level effort", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family: effortReasoningFamily([]string{"low", "high", "max"}, "high", true),
			effort: "max", target: llmprotocol.OpenAIChatV1,
			profile: config.ProviderProfile{Type: "deepseek"}, enabled: true,
		})
		assertThinkingObjectReasoningRequestWithEffort(t, request, "enabled", "max")
		_, hasReasoning := request["reasoning"]
		assert.False(t, hasReasoning)
	})

	t.Run("deepseek responses keeps the responses reasoning object", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family: effortReasoningFamily([]string{"low", "high", "max"}, "high", true),
			effort: "max", target: llmprotocol.OpenAIResponsesV1,
			profile: config.ProviderProfile{Type: "deepseek"}, enabled: true,
		})
		assertReasoningObjectEffort(t, request, "max")
		assertNoTopLevelOrTemplateReasoning(t, request)
	})

	t.Run("deepseek responses disables reasoning with the standard none effort", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family:  effortReasoningFamily([]string{"low", "high", "max"}, "high", true),
			target:  llmprotocol.OpenAIResponsesV1,
			profile: config.ProviderProfile{Type: "deepseek"}, enabled: false,
		})
		assertReasoningObjectEffort(t, request, "none")
		assertNoTopLevelOrTemplateReasoning(t, request)
	})

	t.Run("deepseek disables chat thinking without an effort", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family:  effortReasoningFamily([]string{"low", "high", "max"}, "high", true),
			target:  llmprotocol.OpenAIChatV1,
			profile: config.ProviderProfile{Type: "deepseek"}, enabled: false,
		})
		assertThinkingObjectReasoningRequest(t, request, config.ReasoningModeDisabled)
		_, hasReasoning := request["reasoning"]
		assert.False(t, hasReasoning)
	})
}

func TestHostedReasoningWireContracts(t *testing.T) {
	t.Run("hosted mode switch uses thinking object", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeReasoningMode, Parameter: "thinking_mode",
				Modes:       []string{config.ReasoningModeDisabled, config.ReasoningModeAdaptive},
				DefaultMode: config.ReasoningModeAdaptive,
			},
			mode: config.ReasoningModeAdaptive, target: llmprotocol.OpenAIChatV1,
			profile: config.ProviderProfile{ReasoningTransport: modelcatalog.ReasoningTransportThinkingObject},
			enabled: true,
		})
		assertThinkingObjectReasoningRequest(t, request, config.ReasoningModeAdaptive)
		assertNoReasoningObjectOrTemplate(t, request)
	})

	t.Run("hosted thinking object can carry a separate effort", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family: effortReasoningFamily([]string{"low", "high", "max"}, "max", true),
			effort: "high", target: llmprotocol.OpenAIChatV1,
			profile: config.ProviderProfile{ReasoningTransport: modelcatalog.ReasoningTransportThinkingEffort},
			enabled: true,
		})
		assertThinkingObjectReasoningRequestWithEffort(t, request, "enabled", "high")
		assertNoReasoningObjectOrTemplate(t, request)
	})

	t.Run("hybrid effort model maps adaptive semantics to provider enabled wire value", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
				ActivationParameter: "enable_thinking", Levels: []string{"high", "max"}, Default: "max",
				Modes:       []string{config.ReasoningModeAdaptive, config.ReasoningModeDisabled},
				DefaultMode: config.ReasoningModeAdaptive,
			},
			mode: config.ReasoningModeAdaptive, effort: "high", target: llmprotocol.OpenAIChatV1,
			profile: config.ProviderProfile{ReasoningTransport: modelcatalog.ReasoningTransportThinkingEffort},
			enabled: true,
		})
		assertThinkingObjectReasoningRequestWithEffort(t, request, "enabled", "high")
		assertNoReasoningObjectOrTemplate(t, request)
	})

	t.Run("top level boolean provider emits no template extension", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeChatTemplateKwargs, Parameter: "enable_thinking",
				Modes:       []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled},
				DefaultMode: config.ReasoningModeEnabled,
			},
			target:  llmprotocol.OpenAIChatV1,
			profile: config.ProviderProfile{ReasoningTransport: modelcatalog.ReasoningTransportTopLevelBoolean},
			enabled: false,
		})
		assert.Equal(t, false, request["enable_thinking"])
		assertNoReasoningObjectsOrTemplate(t, request)
	})
}

func TestGatewayAndAnthropicReasoningWireContracts(t *testing.T) {
	t.Run("gateway uses one normalized reasoning object", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family: effortReasoningFamily([]string{"low", "high"}, "high", true),
			effort: "low", target: llmprotocol.OpenAIChatV1,
			profile: config.ProviderProfile{Type: "openrouter"}, enabled: true,
		})
		assertReasoningObjectEffort(t, request, "low")
		assertNoTopLevelOrTemplateReasoning(t, request)
	})

	t.Run("anthropic messages uses adaptive thinking plus output config effort", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "effort",
				Levels: []string{"low", "medium", "high", "xhigh", "max"}, Default: "high",
				Modes:       []string{config.ReasoningModeAdaptive, config.ReasoningModeDisabled},
				DefaultMode: config.ReasoningModeAdaptive,
			},
			effort: "xhigh", target: llmprotocol.AnthropicMessagesV1,
			profile: config.ProviderProfile{Type: "anthropic"}, enabled: true,
		})
		assertThinkingObjectReasoningRequest(t, request, config.ReasoningModeAdaptive)
		outputConfig, ok := request["output_config"].(map[string]interface{})
		require.True(t, ok)
		assert.Equal(t, "xhigh", outputConfig["effort"])
		_, hasReasoning := request["reasoning"]
		assert.False(t, hasReasoning)
	})

	t.Run("anthropic messages disables thinking without an effort field", func(t *testing.T) {
		request := renderProviderReasoningWire(t, reasoningWireFixture{
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "effort",
				Levels: []string{"low", "medium", "high"}, Default: "high",
				Modes:       []string{config.ReasoningModeAdaptive, config.ReasoningModeDisabled},
				DefaultMode: config.ReasoningModeAdaptive,
			},
			target:  llmprotocol.AnthropicMessagesV1,
			profile: config.ProviderProfile{Type: "anthropic"}, enabled: false,
		})
		assertThinkingObjectReasoningRequest(t, request, config.ReasoningModeDisabled)
		_, hasOutputConfig := request["output_config"]
		assert.False(t, hasOutputConfig)
	})
}

type reasoningWireFixture struct {
	family  config.ReasoningFamilyConfig
	effort  string
	mode    string
	target  llmprotocol.WireFormat
	profile config.ProviderProfile
	enabled bool
}

func renderProviderReasoningWire(t *testing.T, fixture reasoningWireFixture) map[string]interface{} {
	t.Helper()
	const model = "reasoning-model"
	ref := config.ModelRef{
		Model: model,
		ModelReasoningControl: config.ModelReasoningControl{
			UseReasoning: boolPtr(fixture.enabled), ReasoningMode: fixture.mode,
			ReasoningEffort: fixture.effort,
		},
	}
	decision := config.Decision{Name: "route", ModelRefs: []config.ModelRef{ref}}
	router := newReasoningRouter(
		config.ReasoningConfig{ReasoningFamilies: map[string]config.ReasoningFamilyConfig{"family": fixture.family}},
		[]config.Decision{decision},
		map[string]config.ModelParams{model: {ReasoningFamily: "family"}},
	)
	request := llmprotocol.Request{
		Generation: 1,
		Model:      model,
		Messages: []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hello"}},
		}},
	}
	if fixture.target != llmprotocol.OpenAIChatV1 {
		router.applySemanticReasoningMode(
			&request, model, fixture.target, fixture.enabled, &decision,
		)
	}
	encoded, err := protocolcodec.NewBuiltinEngine().EncodeRequest(
		fixture.target, request, llmprotocol.Envelope{},
	)
	require.NoError(t, err)
	adapted, err := router.adaptProviderRequest(
		encoded.Body,
		&providerDispatch{
			logicalModel: model, targetFormat: fixture.target, decisionName: decision.Name,
			useReasoning: fixture.enabled, profile: &fixture.profile,
		},
		&RequestContext{VSRSelectedDecision: &decision},
	)
	require.NoError(t, err)
	return unmarshalReasoningRequest(t, adapted)
}

func effortReasoningFamily(levels []string, defaultEffort string, canDisable bool) config.ReasoningFamilyConfig {
	modes := []string{config.ReasoningModeEnabled}
	disabled := ""
	if canDisable {
		modes = append(modes, config.ReasoningModeDisabled)
		disabled = "none"
	}
	return config.ReasoningFamilyConfig{
		Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
		Levels: levels, Default: defaultEffort, Disabled: disabled,
		Modes: modes, DefaultMode: config.ReasoningModeEnabled,
	}
}

func assertReasoningObjectEffort(t *testing.T, request map[string]interface{}, effort string) {
	t.Helper()
	reasoning, ok := request["reasoning"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, effort, reasoning["effort"])
}

func assertNoProviderReasoningFields(t *testing.T, request map[string]interface{}) {
	t.Helper()
	for _, field := range []string{"reasoning", "reasoning_effort", "thinking", "output_config"} {
		_, exists := request[field]
		assert.False(t, exists, "%s must be absent", field)
	}
}

func assertNoReasoningObjectsOrTemplate(t *testing.T, request map[string]interface{}) {
	t.Helper()
	for _, field := range []string{"reasoning", "thinking", "chat_template_kwargs", "output_config"} {
		_, exists := request[field]
		assert.False(t, exists, "%s must be absent", field)
	}
}

func assertNoTopLevelOrTemplateReasoning(t *testing.T, request map[string]interface{}) {
	t.Helper()
	for _, field := range []string{"reasoning_effort", "thinking", "chat_template_kwargs", "output_config"} {
		_, exists := request[field]
		assert.False(t, exists, "%s must be absent", field)
	}
}

func assertNoReasoningObjectOrTemplate(t *testing.T, request map[string]interface{}) {
	t.Helper()
	for _, field := range []string{"reasoning", "chat_template_kwargs", "output_config"} {
		_, exists := request[field]
		assert.False(t, exists, "%s must be absent", field)
	}
}
