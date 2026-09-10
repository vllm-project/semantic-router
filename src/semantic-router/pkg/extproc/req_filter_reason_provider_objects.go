package extproc

import (
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func applyReasoningObjectMutation(
	mutation *reasoningRequestMutation,
	familyConfig *config.ReasoningFamilyConfig,
	enabled bool,
	effort string,
) {
	reasoning := prepareReasoningObjectMutation(mutation, familyConfig.Parameter)
	if !enabled && !reasoningFamilyCanDisable(familyConfig) {
		storeReasoningObjectMutation(mutation, reasoning, false)
		return
	}
	applyReasoningObjectControl(reasoning, familyConfig, enabled, effort)
	storeReasoningObjectMutation(mutation, reasoning, true)
}

func prepareReasoningObjectMutation(
	mutation *reasoningRequestMutation,
	parameter string,
) map[string]json.RawMessage {
	// OpenAI-compatible gateways such as OpenRouter normalize provider-specific
	// controls into a top-level reasoning object. Keep unrelated presentation
	// options such as exclude, but remove mutually exclusive budget controls.
	resetChatTemplateReasoningFields(mutation)
	removeOutputConfigEffort(mutation, "effort")
	removeOutputConfigEffort(mutation, parameter)
	delete(mutation.requestMap, parameter)
	delete(mutation.requestMap, "reasoning_effort")
	delete(mutation.requestMap, "thinking")

	reasoning := map[string]json.RawMessage{}
	if raw, ok := mutation.requestMap["reasoning"]; ok {
		_ = json.Unmarshal(raw, &reasoning)
	}
	delete(reasoning, "effort")
	delete(reasoning, "max_tokens")
	delete(reasoning, "enabled")
	return reasoning
}

func applyReasoningObjectControl(
	reasoning map[string]json.RawMessage,
	familyConfig *config.ReasoningFamilyConfig,
	enabled bool,
	effort string,
) {
	if familyConfig.Type == config.ReasoningFamilyTypeReasoningMode {
		applyReasoningObjectModeControl(reasoning, familyConfig.Disabled, enabled, effort)
		return
	}
	if !enabled {
		reasoning["enabled"] = json.RawMessage("false")
	} else if effort != "" {
		reasoning["effort"] = reasoningStringValue(effort)
	} else {
		reasoning["enabled"] = json.RawMessage("true")
	}
}

func applyReasoningObjectModeControl(
	reasoning map[string]json.RawMessage,
	disabledValue string,
	enabled bool,
	effort string,
) {
	switch {
	case !enabled || effort == disabledValue:
		reasoning["enabled"] = json.RawMessage("false")
	case effort == string(llmprotocol.ReasoningModeEnabled):
		reasoning["enabled"] = json.RawMessage("true")
	case effort == string(llmprotocol.ReasoningModeAdaptive):
		// The gateway has no adaptive enum. Omitting the control preserves
		// the model's native adaptive default.
	}
}

func storeReasoningObjectMutation(
	mutation *reasoningRequestMutation,
	reasoning map[string]json.RawMessage,
	applied bool,
) {
	if len(reasoning) == 0 {
		delete(mutation.requestMap, "reasoning")
		mutation.reasoningApplied = applied
		return
	}
	encoded, err := json.Marshal(reasoning)
	if err != nil {
		return
	}
	mutation.requestMap["reasoning"] = encoded
	mutation.reasoningApplied = applied
}

func applyThinkingObjectReasoningMutation(
	mutation *reasoningRequestMutation,
	parameter string,
	thinkingType string,
) {
	resetChatTemplateReasoningFields(mutation)
	removeOutputConfigEffort(mutation, "effort")
	if parameter != "effort" {
		removeOutputConfigEffort(mutation, parameter)
	}
	delete(mutation.requestMap, "reasoning")
	delete(mutation.requestMap, "reasoning_effort")
	if parameter != "thinking" {
		delete(mutation.requestMap, parameter)
	}
	if thinkingType == "" {
		delete(mutation.requestMap, "thinking")
		mutation.reasoningApplied = false
		return
	}
	mutation.requestMap["thinking"] = json.RawMessage(`{"type":"` + thinkingType + `"}`)
	mutation.reasoningApplied = true
}

func reasoningFamilyCanDisable(family *config.ReasoningFamilyConfig) bool {
	return family != nil && (family.ActivationParameter != "" || family.Disabled != "" ||
		reasoningFamilySupportsMode(family, string(llmprotocol.ReasoningModeDisabled)))
}

func applyOutputConfigEffortMutation(
	mutation *reasoningRequestMutation,
	parameter string,
	effort string,
) {
	resetChatTemplateReasoningFields(mutation)
	delete(mutation.requestMap, "reasoning_effort")
	delete(mutation.requestMap, "reasoning")
	delete(mutation.requestMap, "thinking")
	outputConfig := outputConfigFields(mutation.requestMap)
	if effort == "" {
		delete(outputConfig, parameter)
		if len(outputConfig) == 0 {
			delete(mutation.requestMap, "output_config")
		}
		return
	}
	outputConfig[parameter] = reasoningStringValue(effort)
	encoded, err := json.Marshal(outputConfig)
	if err != nil {
		return
	}
	mutation.requestMap["output_config"] = encoded
	mutation.reasoningApplied = true
}

func preserveOutputConfigEffort(mutation *reasoningRequestMutation, parameter string) {
	if mutation.hasOriginalEffort {
		outputConfig := outputConfigFields(mutation.requestMap)
		outputConfig[parameter] = mutation.originalReasoningEffort
		encoded, err := json.Marshal(outputConfig)
		if err == nil {
			mutation.requestMap["output_config"] = encoded
		}
	}
	var effort string
	if mutation.hasOriginalEffort && json.Unmarshal(mutation.originalReasoningEffort, &effort) == nil {
		mutation.appliedEffort = effort
	}
}

func removeOutputConfigEffort(mutation *reasoningRequestMutation, parameter string) {
	outputConfig := outputConfigFields(mutation.requestMap)
	if _, ok := outputConfig[parameter]; !ok {
		return
	}
	delete(outputConfig, parameter)
	if len(outputConfig) == 0 {
		delete(mutation.requestMap, "output_config")
		return
	}
	encoded, err := json.Marshal(outputConfig)
	if err == nil {
		mutation.requestMap["output_config"] = encoded
	}
}

func outputConfigFields(requestMap map[string]json.RawMessage) map[string]json.RawMessage {
	result := map[string]json.RawMessage{}
	if raw, ok := requestMap["output_config"]; ok {
		_ = json.Unmarshal(raw, &result)
	}
	return result
}

func applyDeepSeekOfficialReasoningMutation(mutation *reasoningRequestMutation, enabled bool, effort string) {
	resetChatTemplateReasoningFields(mutation)
	removeOutputConfigEffort(mutation, "effort")
	delete(mutation.requestMap, "reasoning")
	if enabled {
		mutation.requestMap["thinking"] = json.RawMessage(`{"type":"enabled"}`)
		if effort != "" {
			mutation.requestMap["reasoning_effort"] = reasoningStringValue(effort)
		} else {
			delete(mutation.requestMap, "reasoning_effort")
		}
		mutation.appliedEffort = effort
	} else {
		mutation.requestMap["thinking"] = json.RawMessage(`{"type":"disabled"}`)
		delete(mutation.requestMap, "reasoning_effort")
	}
	mutation.reasoningApplied = true
}
