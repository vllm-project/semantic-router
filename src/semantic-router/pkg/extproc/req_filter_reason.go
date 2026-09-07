package extproc

import (
	"encoding/json"
	"fmt"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
	return r.setReasoningModeToRequestBodyForModelAndProvider(
		requestBody, "", enabled, decision, profile,
	)
}

func (r *OpenAIRouter) setReasoningModeToRequestBodyForModelAndProvider(
	requestBody []byte,
	logicalModel string,
	enabled bool,
	decision *config.Decision,
	profile *config.ProviderProfile,
) ([]byte, error) {
	mutation, err := parseReasoningRequestMutation(requestBody)
	if err != nil {
		return nil, err
	}
	if logicalModel != "" {
		mutation.model = logicalModel
	}
	familyConfig := r.getModelReasoningFamily(mutation.model)
	if familyConfig == nil {
		// A model without a reasoning family is operator-defined and has no
		// projection contract. Preserve its request byte-for-byte, including any
		// provider-specific reasoning fields supplied by the client.
		return requestBody, nil
	}
	transport := resolveProviderReasoningTransport(profile)
	if enabled {
		r.applyEnabledReasoningMutation(mutation, familyConfig, decision, transport)
	} else {
		r.applyDisabledReasoningMutation(mutation, familyConfig, transport)
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

	originalReasoningEffort, hasOriginalEffort := extractOriginalReasoningEffort(requestMap)
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

func extractOriginalReasoningEffort(requestMap map[string]json.RawMessage) (json.RawMessage, bool) {
	if effort, ok := requestMap["reasoning_effort"]; ok {
		return effort, true
	}
	var reasoning map[string]json.RawMessage
	if raw, ok := requestMap["reasoning"]; ok && json.Unmarshal(raw, &reasoning) == nil {
		if effort, ok := reasoning["effort"]; ok {
			return effort, true
		}
	}
	var outputConfig map[string]json.RawMessage
	if raw, ok := requestMap["output_config"]; ok && json.Unmarshal(raw, &outputConfig) == nil {
		if effort, ok := outputConfig["effort"]; ok {
			return effort, true
		}
	}
	return nil, false
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
	transport modelcatalog.ReasoningTransport,
) {
	if familyConfig == nil {
		return
	}
	if r.applyEnabledReasoningTransportMutation(mutation, familyConfig, decision, transport) {
		return
	}
	if usesChatTemplateEffortFlags(familyConfig, transport) {
		effort := r.getReasoningEffort(decision, mutation.model)
		applyChatTemplateEffortFlagsMutation(mutation, familyConfig, true, effort)
		return
	}
	switch familyConfig.Type {
	case config.ReasoningFamilyTypeChatTemplateKwargs:
		applyBooleanReasoningField(mutation, familyConfig.Parameter, true, transport)
		mutation.reasoningApplied = true
	case config.ReasoningFamilyTypeReasoningEffort:
		effort := r.getReasoningEffort(decision, mutation.model)
		applyReasoningEffortField(mutation, familyConfig.Parameter, effort, transport)
		applyReasoningActivation(mutation, familyConfig, true, transport)
		mutation.appliedEffort = effort
		mutation.reasoningApplied = true
	case config.ReasoningFamilyTypeReasoningMode:
		mode := r.getReasoningMode(decision, mutation.model, true)
		applyReasoningModeField(mutation, familyConfig.Parameter, mode, transport)
		mutation.appliedEffort = mode
		mutation.reasoningApplied = mode != ""
	case config.ReasoningFamilyTypeTopLevelReasoningEffort:
		effort := r.getReasoningEffort(decision, mutation.model)
		applyTopLevelReasoningEffortField(mutation, familyConfig.Parameter, effort)
		mutation.appliedEffort = effort
		mutation.reasoningApplied = true
	default:
		return
	}
}

func (r *OpenAIRouter) applyEnabledReasoningTransportMutation(
	mutation *reasoningRequestMutation,
	familyConfig *config.ReasoningFamilyConfig,
	decision *config.Decision,
	transport modelcatalog.ReasoningTransport,
) bool {
	switch {
	case usesIndependentEffortSwitch(transport):
		effort := r.getReasoningEffort(decision, mutation.model)
		applyIndependentEffortSwitchMutation(mutation, familyConfig, true, effort, transport)
	case usesDeepSeekOfficialReasoning(familyConfig, transport):
		effort := r.getReasoningEffort(decision, mutation.model)
		applyDeepSeekOfficialReasoningMutation(mutation, true, effort)
	case usesReasoningObjectTransport(transport):
		control := r.enabledReasoningObjectControl(mutation.model, familyConfig, decision)
		applyReasoningObjectMutation(mutation, familyConfig, true, control)
		mutation.appliedEffort = control
	case usesThinkingObjectTransport(transport):
		r.applyEnabledThinkingObjectMutation(mutation, familyConfig, decision, transport)
	case usesOutputConfigEffortTransport(transport):
		effort := r.getReasoningEffort(decision, mutation.model)
		applyOutputConfigEffortMutation(mutation, familyConfig.Parameter, effort)
		mutation.appliedEffort = effort
	default:
		return false
	}
	return true
}

func (r *OpenAIRouter) enabledReasoningObjectControl(
	model string,
	familyConfig *config.ReasoningFamilyConfig,
	decision *config.Decision,
) string {
	switch familyConfig.Type {
	case config.ReasoningFamilyTypeChatTemplateKwargs:
		return ""
	case config.ReasoningFamilyTypeReasoningMode:
		return r.getReasoningMode(decision, model, true)
	default:
		return r.getReasoningEffort(decision, model)
	}
}

func (r *OpenAIRouter) applyEnabledThinkingObjectMutation(
	mutation *reasoningRequestMutation,
	familyConfig *config.ReasoningFamilyConfig,
	decision *config.Decision,
	transport modelcatalog.ReasoningTransport,
) {
	thinkingType := "enabled"
	if familyConfig.Type == config.ReasoningFamilyTypeReasoningMode {
		thinkingType = r.getReasoningMode(decision, mutation.model, true)
	}
	applyThinkingObjectReasoningMutation(mutation, familyConfig.Parameter, thinkingType)
	if !usesThinkingObjectEffortTransport(transport) {
		return
	}
	effort := r.getReasoningEffort(decision, mutation.model)
	applyTopLevelReasoningEffortField(mutation, familyConfig.Parameter, effort)
	mutation.appliedEffort = effort
	mutation.reasoningApplied = true
}

func (r *OpenAIRouter) applyDisabledReasoningMutation(
	mutation *reasoningRequestMutation,
	familyConfig *config.ReasoningFamilyConfig,
	transport modelcatalog.ReasoningTransport,
) {
	if familyConfig == nil {
		return
	}
	if usesIndependentEffortSwitch(transport) {
		applyIndependentEffortSwitchMutation(mutation, familyConfig, false, "", transport)
		return
	}
	if applyDisabledReasoningWireMutation(mutation, familyConfig, transport) {
		return
	}
	applyDisabledReasoningFamilyMutation(mutation, familyConfig, transport)
}

func applyDisabledReasoningWireMutation(
	mutation *reasoningRequestMutation,
	familyConfig *config.ReasoningFamilyConfig,
	transport modelcatalog.ReasoningTransport,
) bool {
	if usesDeepSeekOfficialReasoning(familyConfig, transport) {
		applyDeepSeekOfficialReasoningMutation(mutation, false, "")
		return true
	}
	if usesReasoningObjectTransport(transport) {
		applyReasoningObjectMutation(mutation, familyConfig, false, "")
		mutation.appliedEffort = familyConfig.Disabled
		return true
	}
	if usesThinkingObjectTransport(transport) {
		applyDisabledThinkingObjectMutation(mutation, familyConfig, transport)
		return true
	}
	if usesOutputConfigEffortTransport(transport) {
		applyDisabledOutputConfigEffortMutation(mutation, familyConfig)
		return true
	}
	if usesChatTemplateEffortFlags(familyConfig, transport) {
		applyChatTemplateEffortFlagsMutation(mutation, familyConfig, false, "")
		return true
	}
	if familyConfig.ActivationParameter != "" && transport != modelcatalog.ReasoningTransportTopLevelEffort {
		applyReasoningActivation(mutation, familyConfig, false, transport)
		removeReasoningEffortField(mutation, familyConfig.Parameter, transport)
		mutation.reasoningApplied = true
		return true
	}
	return false
}

func applyDisabledThinkingObjectMutation(
	mutation *reasoningRequestMutation,
	familyConfig *config.ReasoningFamilyConfig,
	transport modelcatalog.ReasoningTransport,
) {
	if !reasoningFamilyCanDisable(familyConfig) {
		if usesThinkingObjectEffortTransport(transport) {
			preserveTopLevelReasoningEffort(mutation, familyConfig.Parameter)
		}
		return
	}
	applyThinkingObjectReasoningMutation(
		mutation,
		familyConfig.Parameter,
		string(llmprotocol.ReasoningModeDisabled),
	)
	if usesThinkingObjectEffortTransport(transport) {
		delete(mutation.requestMap, familyConfig.Parameter)
	}
}

func applyDisabledOutputConfigEffortMutation(
	mutation *reasoningRequestMutation,
	familyConfig *config.ReasoningFamilyConfig,
) {
	if familyConfig.Disabled == "" {
		preserveOutputConfigEffort(mutation, familyConfig.Parameter)
		return
	}
	applyOutputConfigEffortMutation(mutation, familyConfig.Parameter, familyConfig.Disabled)
	mutation.appliedEffort = familyConfig.Disabled
}

func applyDisabledReasoningFamilyMutation(
	mutation *reasoningRequestMutation,
	familyConfig *config.ReasoningFamilyConfig,
	transport modelcatalog.ReasoningTransport,
) {
	switch familyConfig.Type {
	case config.ReasoningFamilyTypeReasoningEffort:
		if familyConfig.Disabled != "" {
			applyReasoningEffortField(mutation, familyConfig.Parameter, familyConfig.Disabled, transport)
			mutation.appliedEffort = familyConfig.Disabled
			mutation.reasoningApplied = true
		} else if familyConfig.ActivationParameter != "" && transport == modelcatalog.ReasoningTransportTopLevelEffort {
			// A plain effort-only API cannot represent an independent local
			// switch. Validated catalog bindings reject this combination; keep
			// an explicitly authored value as the safe fallback for defensive
			// callers instead of inventing an unsupported "none" effort.
			preserveReasoningEffort(mutation, familyConfig.Parameter, transport)
		} else if reasoningFamilySupportsMode(familyConfig, string(llmprotocol.ReasoningModeDisabled)) {
			applyReasoningEffortField(mutation, familyConfig.Parameter, "none", transport)
			mutation.appliedEffort = "none"
			mutation.reasoningApplied = true
		} else {
			preserveReasoningEffort(mutation, familyConfig.Parameter, transport)
		}
	case config.ReasoningFamilyTypeReasoningMode:
		if reasoningFamilySupportsMode(familyConfig, string(llmprotocol.ReasoningModeDisabled)) {
			applyReasoningModeField(mutation, familyConfig.Parameter, string(llmprotocol.ReasoningModeDisabled), transport)
			mutation.appliedEffort = string(llmprotocol.ReasoningModeDisabled)
			mutation.reasoningApplied = true
		} else {
			removeReasoningEffortField(mutation, familyConfig.Parameter, transport)
		}
	case config.ReasoningFamilyTypeTopLevelReasoningEffort:
		if familyConfig.Disabled != "" {
			applyTopLevelReasoningEffortField(mutation, familyConfig.Parameter, familyConfig.Disabled)
			mutation.appliedEffort = familyConfig.Disabled
			mutation.reasoningApplied = true
		} else {
			preserveTopLevelReasoningEffort(mutation, familyConfig.Parameter)
		}
	case config.ReasoningFamilyTypeChatTemplateKwargs:
		// Some chat-template models default to thinking enabled, so disabled
		// reasoning still needs an explicit false flag for those families.
		applyBooleanReasoningField(mutation, familyConfig.Parameter, false, transport)
		mutation.appliedEffort = familyConfig.Disabled
		mutation.reasoningApplied = true
	default:
		return
	}
}

func applyBooleanReasoningField(
	mutation *reasoningRequestMutation,
	parameter string,
	enabled bool,
	transport modelcatalog.ReasoningTransport,
) {
	value := json.RawMessage("false")
	if enabled {
		value = json.RawMessage("true")
	}
	if transport == modelcatalog.ReasoningTransportTopLevelBoolean ||
		transport == modelcatalog.ReasoningTransportEffortBooleanSwitch {
		prepareTopLevelReasoningMutation(mutation, parameter)
		mutation.requestMap[parameter] = value
		return
	}
	prepareChatTemplateReasoningMutation(mutation, parameter)
	setChatTemplateReasoningValue(mutation, parameter, value)
}

// applyIndependentEffortSwitchMutation projects models with two independent
// controls without conflating their locations. Self-hosted Qwen puts
// reasoning_effort at the request top level and enable_thinking in
// chat_template_kwargs; Qwen's first-party Chat API puts both at the top level.
// Unrelated local template kwargs are retained only for the self-hosted shape.
func applyIndependentEffortSwitchMutation(
	mutation *reasoningRequestMutation,
	familyConfig *config.ReasoningFamilyConfig,
	enabled bool,
	effort string,
	transport modelcatalog.ReasoningTransport,
) {
	removeOutputConfigEffort(mutation, "effort")
	if familyConfig.Parameter != "effort" {
		removeOutputConfigEffort(mutation, familyConfig.Parameter)
	}
	delete(mutation.requestMap, "reasoning")
	delete(mutation.requestMap, "thinking")
	delete(mutation.requestMap, familyConfig.Parameter)
	delete(mutation.requestMap, familyConfig.ActivationParameter)

	if transport == modelcatalog.ReasoningTransportEffortBooleanSwitch {
		resetChatTemplateReasoningFields(mutation)
	} else {
		removeChatTemplateReasoningValue(mutation, familyConfig.Parameter)
		removeChatTemplateReasoningValue(mutation, familyConfig.ActivationParameter)
	}

	if enabled && effort != "" {
		mutation.requestMap[familyConfig.Parameter] = reasoningStringValue(effort)
	} else {
		delete(mutation.requestMap, familyConfig.Parameter)
	}
	activation := json.RawMessage("false")
	if enabled {
		activation = json.RawMessage("true")
	}
	if transport == modelcatalog.ReasoningTransportEffortBooleanSwitch {
		mutation.requestMap[familyConfig.ActivationParameter] = activation
	} else {
		setChatTemplateReasoningValue(mutation, familyConfig.ActivationParameter, activation)
	}
	mutation.appliedEffort = effort
	mutation.reasoningApplied = true
}

func applyTopLevelReasoningEffortField(
	mutation *reasoningRequestMutation,
	parameter string,
	effort string,
) {
	removeChatTemplateReasoningValue(mutation, parameter)
	if effort == "" {
		delete(mutation.requestMap, parameter)
		return
	}
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
	if mutation.hasOriginalEffort && json.Unmarshal(mutation.originalReasoningEffort, &effort) == nil {
		mutation.appliedEffort = effort
	}
}

func applyReasoningEffortField(
	mutation *reasoningRequestMutation,
	parameter string,
	effort string,
	transport modelcatalog.ReasoningTransport,
) {
	if effort == "" {
		removeReasoningEffortField(mutation, parameter, transport)
		return
	}
	if usesTopLevelReasoningEffort(transport) {
		prepareTopLevelReasoningMutation(mutation, parameter)
		mutation.requestMap[parameter] = reasoningStringValue(effort)
		return
	}
	// Local vLLM-compatible reasoning_effort models expect the value under
	// chat_template_kwargs, not as an OpenAI top-level request field.
	prepareChatTemplateReasoningMutation(mutation, parameter)
	setChatTemplateReasoningValue(mutation, parameter, reasoningStringValue(effort))
}

func applyReasoningModeField(
	mutation *reasoningRequestMutation,
	parameter string,
	mode string,
	transport modelcatalog.ReasoningTransport,
) {
	if mode == "" {
		removeReasoningEffortField(mutation, parameter, transport)
		return
	}
	if usesTopLevelReasoningEffort(transport) {
		prepareTopLevelReasoningMutation(mutation, parameter)
		applyTopLevelReasoningEffortField(mutation, parameter, mode)
		return
	}
	prepareChatTemplateReasoningMutation(mutation, parameter)
	setChatTemplateReasoningValue(mutation, parameter, reasoningStringValue(mode))
}

func removeReasoningEffortField(
	mutation *reasoningRequestMutation,
	parameter string,
	transport modelcatalog.ReasoningTransport,
) {
	removeOutputConfigEffort(mutation, parameter)
	if usesTopLevelReasoningEffort(transport) {
		delete(mutation.requestMap, parameter)
		return
	}
	removeChatTemplateReasoningValue(mutation, parameter)
}

func applyReasoningActivation(
	mutation *reasoningRequestMutation,
	familyConfig *config.ReasoningFamilyConfig,
	enabled bool,
	transport modelcatalog.ReasoningTransport,
) {
	if familyConfig.ActivationParameter == "" {
		return
	}
	if transport == modelcatalog.ReasoningTransportTopLevelEffort {
		// Top-level effort APIs use the effort value as the complete control.
		// Sending the model's local activation flag alongside it creates two
		// overlapping controls and is rejected by providers such as DashScope.
		removeChatTemplateReasoningValue(mutation, familyConfig.ActivationParameter)
		delete(mutation.requestMap, familyConfig.ActivationParameter)
		return
	}
	applyBooleanReasoningField(mutation, familyConfig.ActivationParameter, enabled, transport)
}

func preserveReasoningEffort(
	mutation *reasoningRequestMutation,
	parameter string,
	transport modelcatalog.ReasoningTransport,
) {
	if !mutation.hasOriginalEffort {
		if !usesTopLevelReasoningEffort(transport) {
			removeChatTemplateReasoningValue(mutation, parameter)
		}
		return
	}
	if usesTopLevelReasoningEffort(transport) {
		// When routing to OpenAI with reasoning disabled, keep a user-supplied
		// top-level effort but do not synthesize a new one.
		mutation.requestMap[parameter] = mutation.originalReasoningEffort
	} else {
		// Preserve only an explicitly authored value. Always-reasoning families
		// must not receive a synthesized "low" level that their contract may not
		// support.
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
	transport modelcatalog.ReasoningTransport,
) bool {
	return familyConfig != nil && isDeepSeekThinkingTransport(transport)
}

// prepareChatTemplateReasoningMutation removes every provider-level reasoning
// control before projecting the selected model's local chat-template syntax.
// In particular, an OpenAI Responses request may already contain a standard
// reasoning object after neutral encoding; forwarding that object together
// with chat_template_kwargs would give the backend two conflicting controls.
func prepareChatTemplateReasoningMutation(mutation *reasoningRequestMutation, parameter string) {
	delete(mutation.requestMap, "reasoning")
	delete(mutation.requestMap, "reasoning_effort")
	delete(mutation.requestMap, "thinking")
	delete(mutation.requestMap, parameter)
	removeOutputConfigEffort(mutation, "effort")
	if parameter != "effort" {
		removeOutputConfigEffort(mutation, parameter)
	}
}

// prepareTopLevelReasoningMutation is the inverse projection: it removes local
// template and object controls before writing a provider's top-level field.
func prepareTopLevelReasoningMutation(mutation *reasoningRequestMutation, parameter string) {
	resetChatTemplateReasoningFields(mutation)
	delete(mutation.requestMap, "reasoning")
	delete(mutation.requestMap, "thinking")
	delete(mutation.requestMap, "reasoning_effort")
	delete(mutation.requestMap, parameter)
	removeOutputConfigEffort(mutation, "effort")
	if parameter != "effort" {
		removeOutputConfigEffort(mutation, parameter)
	}
}

func resetChatTemplateReasoningFields(mutation *reasoningRequestMutation) {
	delete(mutation.requestMap, "chat_template_kwargs")
	mutation.chatTemplateKwargs = map[string]json.RawMessage{}
	mutation.chatTemplateKwargsDirty = false
}
