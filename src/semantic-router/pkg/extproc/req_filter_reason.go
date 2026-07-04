package extproc

import (
	"encoding/json"
	"fmt"
	"net/url"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

type reasoningRequestMutation struct {
	requestMap              map[string]interface{}
	chatTemplateKwargs      map[string]interface{}
	model                   string
	originalReasoningEffort interface{}
	hasOriginalEffort       bool
	appliedEffort           string
	reasoningApplied        bool
}

func (r *OpenAIRouter) setReasoningModeToRequestBody(requestBody []byte, enabled bool, categoryName string) ([]byte, error) {
	return r.setReasoningModeToRequestBodyForProvider(requestBody, enabled, categoryName, nil)
}

// setReasoningModeToRequestBodyForProvider adds provider-compatible reasoning fields to the JSON request body.
func (r *OpenAIRouter) setReasoningModeToRequestBodyForProvider(
	requestBody []byte,
	enabled bool,
	categoryName string,
	profile *config.ProviderProfile,
) ([]byte, error) {
	mutation, err := parseReasoningRequestMutation(requestBody)
	if err != nil {
		return nil, err
	}
	familyConfig := r.getModelReasoningFamily(mutation.model)
	if enabled {
		r.applyEnabledReasoningMutation(mutation, familyConfig, categoryName, profile)
	} else {
		applyDisabledReasoningMutation(mutation, familyConfig, profile)
	}

	logReasoningMutation(mutation, enabled)
	r.recordReasoningMutationMetrics(mutation, enabled, familyConfig)

	modifiedBody, err := json.Marshal(mutation.requestMap)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize modified request: %w", err)
	}

	return modifiedBody, nil
}

func parseReasoningRequestMutation(requestBody []byte) (*reasoningRequestMutation, error) {
	var requestMap map[string]interface{}
	if err := json.Unmarshal(requestBody, &requestMap); err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}

	originalReasoningEffort, hasOriginalEffort := requestMap["reasoning_effort"]
	if !hasOriginalEffort {
		originalReasoningEffort = "low"
	}
	// Normalize the request before applying the selected family syntax. The
	// top-level field is restored only for providers that accept it; vLLM-style
	// backends receive reasoning_effort through chat_template_kwargs instead.
	delete(requestMap, "reasoning_effort")

	return &reasoningRequestMutation{
		requestMap:              requestMap,
		chatTemplateKwargs:      extractChatTemplateKwargs(requestMap),
		model:                   extractReasoningRequestModel(requestMap),
		originalReasoningEffort: originalReasoningEffort,
		hasOriginalEffort:       hasOriginalEffort,
	}, nil
}

func extractReasoningRequestModel(requestMap map[string]interface{}) string {
	modelValue, ok := requestMap["model"]
	if !ok {
		return consts.UnknownLabel
	}
	model, ok := modelValue.(string)
	if !ok {
		return consts.UnknownLabel
	}
	return model
}

func extractChatTemplateKwargs(requestMap map[string]interface{}) map[string]interface{} {
	kwargs, ok := requestMap["chat_template_kwargs"].(map[string]interface{})
	if !ok || kwargs == nil {
		return map[string]interface{}{}
	}
	return kwargs
}

func (r *OpenAIRouter) applyEnabledReasoningMutation(
	mutation *reasoningRequestMutation,
	familyConfig *config.ReasoningFamilyConfig,
	categoryName string,
	profile *config.ProviderProfile,
) {
	if familyConfig == nil {
		return
	}
	switch familyConfig.Type {
	case "chat_template_kwargs":
		mutation.chatTemplateKwargs[familyConfig.Parameter] = true
		mutation.requestMap["chat_template_kwargs"] = mutation.chatTemplateKwargs
		mutation.reasoningApplied = true
	case "reasoning_effort":
		effort := r.getReasoningEffort(categoryName, mutation.model)
		applyReasoningEffortField(mutation, familyConfig.Parameter, effort, profile)
		mutation.appliedEffort = effort
		mutation.reasoningApplied = true
	default:
		return
	}
}

func applyDisabledReasoningMutation(
	mutation *reasoningRequestMutation,
	familyConfig *config.ReasoningFamilyConfig,
	profile *config.ProviderProfile,
) {
	if familyConfig == nil {
		return
	}
	switch familyConfig.Type {
	case "reasoning_effort":
		preserveReasoningEffort(mutation, familyConfig.Parameter, profile)
	case "chat_template_kwargs":
		// Some chat-template models default to thinking enabled, so disabled
		// reasoning still needs an explicit false flag for those families.
		mutation.chatTemplateKwargs[familyConfig.Parameter] = false
		mutation.requestMap["chat_template_kwargs"] = mutation.chatTemplateKwargs
	default:
		return
	}
}

func applyReasoningEffortField(
	mutation *reasoningRequestMutation,
	parameter string,
	effort string,
	profile *config.ProviderProfile,
) {
	if usesTopLevelReasoningEffort(profile) {
		mutation.requestMap[parameter] = effort
		return
	}
	// Local vLLM-compatible reasoning_effort models expect the value under
	// chat_template_kwargs, not as an OpenAI top-level request field.
	mutation.chatTemplateKwargs[parameter] = effort
	mutation.requestMap["chat_template_kwargs"] = mutation.chatTemplateKwargs
}

func preserveReasoningEffort(
	mutation *reasoningRequestMutation,
	parameter string,
	profile *config.ProviderProfile,
) {
	if usesTopLevelReasoningEffort(profile) {
		// When routing to OpenAI with reasoning disabled, keep a user-supplied
		// top-level effort but do not synthesize a new one.
		if mutation.hasOriginalEffort {
			mutation.requestMap[parameter] = mutation.originalReasoningEffort
		}
	} else {
		// Non-OpenAI reasoning_effort families preserve the effective effort in
		// chat_template_kwargs so backends that require the template arg still see it.
		mutation.chatTemplateKwargs[parameter] = mutation.originalReasoningEffort
		mutation.requestMap["chat_template_kwargs"] = mutation.chatTemplateKwargs
	}
	if effort, ok := mutation.originalReasoningEffort.(string); ok {
		mutation.appliedEffort = effort
	}
}

func logReasoningMutation(mutation *reasoningRequestMutation, enabled bool) {
	if enabled && !mutation.reasoningApplied {
		logging.Infof("No reasoning support for model: %s (no reasoning family configured)", mutation.model)
		return
	}
	if mutation.reasoningApplied {
		logging.Infof("Applied reasoning mode (enabled: %v) with effort (%s) to model: %s", enabled, mutation.appliedEffort, mutation.model)
		return
	}
	logging.Infof("Reasoning mode disabled for model: %s", mutation.model)
}

func (r *OpenAIRouter) recordReasoningMutationMetrics(
	mutation *reasoningRequestMutation,
	enabled bool,
	familyConfig *config.ReasoningFamilyConfig,
) {
	if !enabled {
		return
	}
	modelFamily, templateParam := r.reasoningMetricLabels(mutation.model, familyConfig)
	metrics.RecordReasoningTemplateUsage(modelFamily, templateParam)
	if mutation.appliedEffort != "" {
		metrics.RecordReasoningEffortUsage(modelFamily, mutation.appliedEffort)
	}
}

func (r *OpenAIRouter) reasoningMetricLabels(
	model string,
	familyConfig *config.ReasoningFamilyConfig,
) (string, string) {
	if familyConfig == nil {
		return consts.UnknownLabel, "reasoning_effort"
	}
	modelFamily := consts.UnknownLabel
	if r.Config != nil {
		if familyName, exists := r.Config.GetModelReasoningFamilyName(model); exists {
			modelFamily = familyName
		}
	}
	if familyConfig.Type == "chat_template_kwargs" {
		return modelFamily, familyConfig.Parameter
	}
	return modelFamily, "reasoning_effort"
}

func usesTopLevelReasoningEffort(profile *config.ProviderProfile) bool {
	if profile == nil || profile.Type != "openai" || profile.BaseURL == "" {
		return false
	}
	u, err := url.Parse(profile.BaseURL)
	if err != nil {
		return false
	}
	// The official OpenAI API and OpenRouter accept reasoning_effort as a
	// top-level OpenAI-compatible field. Local vLLM-compatible servers keep the
	// value under chat_template_kwargs.
	return supportsTopLevelReasoningEffortHost(u.Hostname())
}

func supportsTopLevelReasoningEffortHost(host string) bool {
	return strings.EqualFold(host, "api.openai.com") ||
		strings.EqualFold(host, "openrouter.ai")
}

// getReasoningEffort returns the reasoning effort level for a given decision and model
func (r *OpenAIRouter) getReasoningEffort(categoryName string, modelName string) string {
	// Handle case where Config is nil (e.g., in tests)
	if r.Config == nil {
		return "medium"
	}

	for _, decision := range r.Config.Decisions {
		if decision.Name != categoryName {
			continue
		}
		if effort := r.reasoningEffortForDecision(decision, modelName); effort != "" {
			return effort
		}
		break
	}

	// Fall back to global default if configured
	if r.Config.DefaultReasoningEffort != "" {
		return r.Config.DefaultReasoningEffort
	}

	// Final fallback to "medium" as a reasonable default
	return "medium"
}

func (r *OpenAIRouter) reasoningEffortForDecision(decision config.Decision, modelName string) string {
	for _, modelRef := range decision.ModelRefs {
		if !r.Config.ModelNameMatches(modelRef.Model, modelName) {
			continue
		}
		return modelRef.ReasoningEffort
	}
	return ""
}

// getModelReasoningFamily finds the reasoning family configuration for a model using the config system
func (r *OpenAIRouter) getModelReasoningFamily(model string) *config.ReasoningFamilyConfig {
	if r.Config == nil {
		return nil
	}
	return r.Config.GetModelReasoningFamily(model)
}

func (r *OpenAIRouter) buildReasoningRequestFieldsForProvider(
	model string,
	useReasoning bool,
	categoryName string,
	profile *config.ProviderProfile,
) (map[string]interface{}, string) {
	familyConfig := r.getModelReasoningFamily(model)
	if familyConfig == nil {
		// No reasoning family configured for this model - don't apply any reasoning syntax
		// Models without reasoning_family don't support reasoning mode
		return nil, ""
	}

	if !useReasoning {
		// When reasoning is disabled, don't add any reasoning fields
		return nil, ""
	}

	// When reasoning is enabled, use the configured family syntax
	switch familyConfig.Type {
	case "chat_template_kwargs":
		kwargs := map[string]interface{}{
			familyConfig.Parameter: useReasoning,
		}
		return map[string]interface{}{"chat_template_kwargs": kwargs}, ""
	case "reasoning_effort":
		effort := r.getReasoningEffort(categoryName, model)
		if usesTopLevelReasoningEffort(profile) {
			return map[string]interface{}{familyConfig.Parameter: effort}, effort
		}
		// Put reasoning_effort inside chat_template_kwargs (vLLM requirement)
		kwargs := map[string]interface{}{
			familyConfig.Parameter: effort,
		}
		return map[string]interface{}{"chat_template_kwargs": kwargs}, effort
	default:
		// Unknown reasoning syntax type - don't apply anything
		return nil, ""
	}
}
