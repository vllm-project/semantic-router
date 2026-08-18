package extproc

import (
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

type reasoningRequestMutation struct {
	requestMap              map[string]json.RawMessage
	chatTemplateKwargs      map[string]json.RawMessage
	chatTemplateKwargsDirty bool
	model                   string
	originalReasoningEffort json.RawMessage
	hasOriginalEffort       bool
	appliedEffort           string
	reasoningApplied        bool
}

func (r *OpenAIRouter) setReasoningModeToRequestBody(
	requestBody []byte,
	enabled bool,
	decision *config.Decision,
) ([]byte, error) {
	return r.setReasoningModeToRequestBodyForProvider(requestBody, enabled, decision, nil)
}

// setReasoningModeToRequestBodyForProvider adds provider-compatible reasoning fields to the JSON request body.
func (r *OpenAIRouter) setReasoningModeToRequestBodyForProvider(
	requestBody []byte,
	enabled bool,
	decision *config.Decision,
	profile *config.ProviderProfile,
) ([]byte, error) {
	mutation, err := parseReasoningRequestMutation(requestBody)
	if err != nil {
		return nil, err
	}
	familyConfig := r.getModelReasoningFamily(mutation.model)
	dialect := resolveOpenAIBackendDialect(profile)
	if enabled {
		r.applyEnabledReasoningMutation(mutation, familyConfig, decision, dialect)
	} else {
		r.applyDisabledReasoningMutation(mutation, familyConfig, dialect)
	}

	logReasoningMutation(mutation, enabled)
	r.recordReasoningMutationMetrics(mutation, enabled, familyConfig)

	if mutation.chatTemplateKwargsDirty {
		kwargs, marshalErr := json.Marshal(mutation.chatTemplateKwargs)
		if marshalErr != nil {
			return nil, fmt.Errorf("failed to serialize chat template kwargs: %w", marshalErr)
		}
		mutation.requestMap["chat_template_kwargs"] = kwargs
	}
	modifiedBody, err := json.Marshal(mutation.requestMap)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize modified request: %w", err)
	}

	return modifiedBody, nil
}

func parseReasoningRequestMutation(requestBody []byte) (*reasoningRequestMutation, error) {
	var requestMap map[string]json.RawMessage
	if err := json.Unmarshal(requestBody, &requestMap); err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}

	originalReasoningEffort, hasOriginalEffort := requestMap["reasoning_effort"]
	if !hasOriginalEffort {
		originalReasoningEffort = reasoningStringValue("low")
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

func extractReasoningRequestModel(requestMap map[string]json.RawMessage) string {
	modelValue, ok := requestMap["model"]
	if !ok {
		return consts.UnknownLabel
	}
	var model string
	if err := json.Unmarshal(modelValue, &model); err != nil {
		return consts.UnknownLabel
	}
	return model
}

func extractChatTemplateKwargs(requestMap map[string]json.RawMessage) map[string]json.RawMessage {
	kwargs := map[string]json.RawMessage{}
	rawKwargs, ok := requestMap["chat_template_kwargs"]
	if !ok || json.Unmarshal(rawKwargs, &kwargs) != nil || kwargs == nil {
		return map[string]json.RawMessage{}
	}
	return kwargs
}

func reasoningStringValue(value string) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err != nil {
		return json.RawMessage(`""`)
	}
	return encoded
}

func setChatTemplateReasoningValue(
	mutation *reasoningRequestMutation,
	parameter string,
	value json.RawMessage,
) {
	mutation.chatTemplateKwargs[parameter] = value
	mutation.chatTemplateKwargsDirty = true
}

func removeChatTemplateReasoningValue(
	mutation *reasoningRequestMutation,
	parameter string,
) {
	if _, exists := mutation.chatTemplateKwargs[parameter]; !exists {
		return
	}
	delete(mutation.chatTemplateKwargs, parameter)
	if len(mutation.chatTemplateKwargs) == 0 {
		delete(mutation.requestMap, "chat_template_kwargs")
		mutation.chatTemplateKwargsDirty = false
		return
	}
	mutation.chatTemplateKwargsDirty = true
}

func (r *OpenAIRouter) applyEnabledReasoningMutation(
	mutation *reasoningRequestMutation,
	familyConfig *config.ReasoningFamilyConfig,
	decision *config.Decision,
	dialect openAIBackendDialect,
) {
	if familyConfig == nil {
		return
	}
	if usesDeepSeekOfficialReasoning(familyConfig, dialect) {
		effort := r.getReasoningEffort(decision, mutation.model)
		applyDeepSeekOfficialReasoningMutation(mutation, true, effort)
		return
	}
	switch familyConfig.Type {
	case config.ReasoningFamilyTypeChatTemplateKwargs:
		setChatTemplateReasoningValue(mutation, familyConfig.Parameter, json.RawMessage("true"))
		mutation.reasoningApplied = true
	case config.ReasoningFamilyTypeReasoningEffort:
		effort := r.getReasoningEffort(decision, mutation.model)
		applyReasoningEffortField(mutation, familyConfig.Parameter, effort, dialect)
		mutation.appliedEffort = effort
		mutation.reasoningApplied = true
	case config.ReasoningFamilyTypeTopLevelReasoningEffort:
		effort := r.getReasoningEffort(decision, mutation.model)
		applyTopLevelReasoningEffortField(mutation, familyConfig.Parameter, effort)
		mutation.appliedEffort = effort
		mutation.reasoningApplied = true
	default:
		return
	}
}

func (r *OpenAIRouter) applyDisabledReasoningMutation(
	mutation *reasoningRequestMutation,
	familyConfig *config.ReasoningFamilyConfig,
	dialect openAIBackendDialect,
) {
	if familyConfig == nil {
		return
	}
	if usesDeepSeekOfficialReasoning(familyConfig, dialect) {
		applyDeepSeekOfficialReasoningMutation(mutation, false, "")
		return
	}
	switch familyConfig.Type {
	case config.ReasoningFamilyTypeReasoningEffort:
		preserveReasoningEffort(mutation, familyConfig.Parameter, dialect)
	case config.ReasoningFamilyTypeTopLevelReasoningEffort:
		preserveTopLevelReasoningEffort(mutation, familyConfig.Parameter)
	case config.ReasoningFamilyTypeChatTemplateKwargs:
		// Some chat-template models default to thinking enabled, so disabled
		// reasoning still needs an explicit false flag for those families.
		setChatTemplateReasoningValue(mutation, familyConfig.Parameter, json.RawMessage("false"))
	default:
		return
	}
}

func applyTopLevelReasoningEffortField(
	mutation *reasoningRequestMutation,
	parameter string,
	effort string,
) {
	removeChatTemplateReasoningValue(mutation, parameter)
	mutation.requestMap[parameter] = reasoningStringValue(effort)
}

func preserveTopLevelReasoningEffort(
	mutation *reasoningRequestMutation,
	parameter string,
) {
	removeChatTemplateReasoningValue(mutation, parameter)
	if mutation.hasOriginalEffort {
		mutation.requestMap[parameter] = mutation.originalReasoningEffort
	}
	var effort string
	if json.Unmarshal(mutation.originalReasoningEffort, &effort) == nil {
		mutation.appliedEffort = effort
	}
}

func applyReasoningEffortField(
	mutation *reasoningRequestMutation,
	parameter string,
	effort string,
	dialect openAIBackendDialect,
) {
	if dialect.usesTopLevelReasoningEffort() {
		mutation.requestMap[parameter] = reasoningStringValue(effort)
		return
	}
	// Local vLLM-compatible reasoning_effort models expect the value under
	// chat_template_kwargs, not as an OpenAI top-level request field.
	setChatTemplateReasoningValue(mutation, parameter, reasoningStringValue(effort))
}

func preserveReasoningEffort(
	mutation *reasoningRequestMutation,
	parameter string,
	dialect openAIBackendDialect,
) {
	if dialect.usesTopLevelReasoningEffort() {
		// When routing to OpenAI with reasoning disabled, keep a user-supplied
		// top-level effort but do not synthesize a new one.
		if mutation.hasOriginalEffort {
			mutation.requestMap[parameter] = mutation.originalReasoningEffort
		}
	} else {
		// Non-OpenAI reasoning_effort families preserve the effective effort in
		// chat_template_kwargs so backends that require the template arg still see it.
		setChatTemplateReasoningValue(mutation, parameter, mutation.originalReasoningEffort)
	}
	var effort string
	if json.Unmarshal(mutation.originalReasoningEffort, &effort) == nil {
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
	if familyConfig.Type == config.ReasoningFamilyTypeChatTemplateKwargs {
		return modelFamily, familyConfig.Parameter
	}
	return modelFamily, "reasoning_effort"
}

func usesDeepSeekOfficialReasoning(
	familyConfig *config.ReasoningFamilyConfig,
	dialect openAIBackendDialect,
) bool {
	return familyConfig != nil && dialect.usesDeepSeekOfficialReasoning()
}

func applyDeepSeekOfficialReasoningMutation(mutation *reasoningRequestMutation, enabled bool, effort string) {
	// DeepSeek's official OpenAI-compatible API uses top-level thinking plus
	// reasoning_effort. Drop template kwargs so local-template controls do not
	// leak into the provider request alongside official fields.
	delete(mutation.requestMap, "chat_template_kwargs")
	mutation.chatTemplateKwargs = map[string]json.RawMessage{}
	mutation.chatTemplateKwargsDirty = false

	if enabled {
		mutation.requestMap["thinking"] = json.RawMessage(`{"type":"enabled"}`)
		mutation.requestMap["reasoning_effort"] = reasoningStringValue(effort)
		mutation.appliedEffort = effort
	} else {
		mutation.requestMap["thinking"] = json.RawMessage(`{"type":"disabled"}`)
		delete(mutation.requestMap, "reasoning_effort")
	}
	// Disabled official reasoning is still an applied provider mutation because
	// the request must carry thinking.type=disabled.
	mutation.reasoningApplied = true
}

// getReasoningEffort returns the reasoning effort level for a given decision and model
func (r *OpenAIRouter) getReasoningEffort(
	decision *config.Decision,
	modelName string,
) string {
	// Handle case where Config is nil (e.g., in tests)
	if r.Config == nil {
		return "medium"
	}

	if decision != nil {
		if effort := r.reasoningEffortForDecision(*decision, modelName); effort != "" {
			return effort
		}
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
	decision *config.Decision,
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
	dialect := resolveOpenAIBackendDialect(profile)
	if usesDeepSeekOfficialReasoning(familyConfig, dialect) {
		effort := r.getReasoningEffort(decision, model)
		return map[string]interface{}{
			"thinking":         map[string]interface{}{"type": "enabled"},
			"reasoning_effort": effort,
		}, effort
	}
	switch familyConfig.Type {
	case config.ReasoningFamilyTypeChatTemplateKwargs:
		kwargs := map[string]interface{}{
			familyConfig.Parameter: useReasoning,
		}
		return map[string]interface{}{"chat_template_kwargs": kwargs}, ""
	case config.ReasoningFamilyTypeReasoningEffort:
		effort := r.getReasoningEffort(decision, model)
		if dialect.usesTopLevelReasoningEffort() {
			return map[string]interface{}{familyConfig.Parameter: effort}, effort
		}
		// Put reasoning_effort inside chat_template_kwargs (vLLM requirement)
		kwargs := map[string]interface{}{
			familyConfig.Parameter: effort,
		}
		return map[string]interface{}{"chat_template_kwargs": kwargs}, effort
	case config.ReasoningFamilyTypeTopLevelReasoningEffort:
		effort := r.getReasoningEffort(decision, model)
		return map[string]interface{}{familyConfig.Parameter: effort}, effort
	default:
		// Unknown reasoning syntax type - don't apply anything
		return nil, ""
	}
}
