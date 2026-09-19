package extproc

import (
	"time"

	"go.opentelemetry.io/otel/attribute"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
)

func (r *OpenAIRouter) applySemanticReasoningMode(
	request *llmprotocol.Request,
	model string,
	targetFormat llmprotocol.WireFormat,
	enabled bool,
	decision *config.Decision,
) bool {
	if request == nil {
		return false
	}
	family := r.getModelReasoningFamily(model)
	if family == nil {
		return false
	}
	effort, mode := semanticReasoningControls(r, family, decision, model, targetFormat, enabled)
	var budget *int64
	if targetFormat == llmprotocol.AnthropicMessagesV1 && mode == llmprotocol.ReasoningModeEnabled {
		budget = request.ReasoningBudgetTokens
	}
	if request.ReasoningEffort == effort && request.ReasoningMode == mode && request.ReasoningBudgetTokens == budget {
		return false
	}
	request.ReasoningMode = mode
	request.ReasoningEffort = effort
	request.ReasoningBudgetTokens = budget
	logging.Infof("Applied reasoning controls to model %q", model)
	return true
}

func semanticReasoningControls(
	router *OpenAIRouter,
	family *config.ReasoningFamilyConfig,
	decision *config.Decision,
	model string,
	targetFormat llmprotocol.WireFormat,
	enabled bool,
) (string, llmprotocol.ReasoningMode) {
	if targetFormat == llmprotocol.AnthropicMessagesV1 {
		return semanticAnthropicReasoningControls(router, family, decision, model, enabled)
	}
	if targetFormat != llmprotocol.OpenAIResponsesV1 && targetFormat != llmprotocol.OpenAIChatV1 {
		return "", ""
	}
	return semanticOpenAIReasoningControls(router, family, decision, model, enabled)
}

func semanticAnthropicReasoningControls(
	router *OpenAIRouter,
	family *config.ReasoningFamilyConfig,
	decision *config.Decision,
	model string,
	enabled bool,
) (string, llmprotocol.ReasoningMode) {
	if enabled {
		return router.getReasoningEffort(decision, model), llmprotocol.ReasoningMode(router.getReasoningMode(decision, model, true))
	}
	if reasoningFamilySupportsMode(family, string(llmprotocol.ReasoningModeDisabled)) {
		return "", llmprotocol.ReasoningModeDisabled
	}
	return "", ""
}

func semanticOpenAIReasoningControls(
	router *OpenAIRouter,
	family *config.ReasoningFamilyConfig,
	decision *config.Decision,
	model string,
	enabled bool,
) (string, llmprotocol.ReasoningMode) {
	if enabled {
		return semanticEnabledOpenAIReasoningControls(router, family, decision, model)
	}
	return semanticDisabledOpenAIReasoningControls(family)
}

func semanticEnabledOpenAIReasoningControls(
	router *OpenAIRouter,
	family *config.ReasoningFamilyConfig,
	decision *config.Decision,
	model string,
) (string, llmprotocol.ReasoningMode) {
	switch family.Type {
	case config.ReasoningFamilyTypeReasoningEffort,
		config.ReasoningFamilyTypeTopLevelReasoningEffort:
		return router.getReasoningEffort(decision, model), ""
	case config.ReasoningFamilyTypeChatTemplateKwargs:
		// vLLM's OpenAI-compatible Responses surface maps a standard effort
		// to enable_thinking for boolean template controls.
		return "high", ""
	default:
		// Provider adaptation owns non-standard string modes such as
		// MiniMax's adaptive thinking_mode.
		return "", ""
	}
}

func semanticDisabledOpenAIReasoningControls(
	family *config.ReasoningFamilyConfig,
) (string, llmprotocol.ReasoningMode) {
	switch family.Type {
	case config.ReasoningFamilyTypeReasoningEffort,
		config.ReasoningFamilyTypeTopLevelReasoningEffort:
		if family.Disabled != "" ||
			reasoningFamilySupportsMode(family, string(llmprotocol.ReasoningModeDisabled)) {
			// The neutral OpenAI-compatible contract has one portable disabled
			// effort value. Provider-native sentinels such as Hunyuan's
			// "no_think" are projected only after encoding, at the final
			// provider adapter boundary.
			return "none", ""
		}
	case config.ReasoningFamilyTypeChatTemplateKwargs:
		if family.Disabled != "" {
			return "none", ""
		}
	}
	return "", ""
}

func reasoningFamilySupportsMode(family *config.ReasoningFamilyConfig, mode string) bool {
	if family == nil {
		return false
	}
	for _, supported := range family.Modes {
		if supported == mode {
			return true
		}
	}
	return false
}

func (r *OpenAIRouter) addSemanticSystemPromptIfConfigured(
	request *llmprotocol.Request,
	decisionName string,
	model string,
	ctx *RequestContext,
) (bool, error) {
	if request == nil || decisionName == "" || ctx == nil || ctx.VSRSelectedDecision == nil {
		return false, nil
	}
	decision := ctx.VSRSelectedDecision
	promptConfig := decision.GetSystemPromptConfig()
	if promptConfig == nil || promptConfig.SystemPrompt == "" || !decision.IsSystemPromptEnabled() {
		return false, nil
	}
	start := time.Now()
	_, span := tracing.StartPluginSpan(ctx.TraceContext, "system_prompt", decisionName)
	mode := decision.GetSystemPromptMode()
	injected := llmprotocol.SetSystemInstruction(request, promptConfig.SystemPrompt, mode)
	latency := time.Since(start).Milliseconds()
	tracing.SetSpanAttributes(span,
		attribute.Bool("system_prompt.injected", injected),
		attribute.String("system_prompt.mode", mode),
		attribute.String(tracing.AttrDecisionName, decisionName),
	)
	tracing.EndPluginSpan(span, "success", latency, "prompt_injected")
	ctx.VSRInjectedSystemPrompt = true
	logging.Infof("Applied system instruction for decision %q to model %q", decisionName, model)
	return true, nil
}

func (r *OpenAIRouter) applySemanticRequestParams(
	decision *config.Decision,
	request *llmprotocol.Request,
	routingScope config.RecipeName,
) (bool, error) {
	if decision == nil || request == nil || decision.GetRequestParamsConfig() == nil {
		return false, nil
	}
	result, err := pluginruntime.ApplyRequestParams(request, decision.GetRequestParamsConfig())
	decisionKey := config.RoutingDecisionKey(routingScope, decision.Name)
	for _, field := range result.Blocked {
		metrics.RecordBlockedParam(decisionKey, field)
	}
	if result.CappedOutputTokens {
		metrics.RecordMaxTokensCapped(decisionKey)
	}
	if result.CappedCandidateCount {
		metrics.RecordMaxNCapped(decisionKey)
	}
	return result.Changed, err
}
