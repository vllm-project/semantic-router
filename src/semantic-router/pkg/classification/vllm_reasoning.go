package classification

import (
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// reasoningRequestControl is resolved once when an LLM classifier is built.
type reasoningRequestControl struct {
	reasoningEffort    *string
	chatTemplateKwargs map[string]json.RawMessage
}

type reasoningFamilyProjector struct {
	enable  func(*reasoningRequestControl, config.ReasoningFamilyConfig, string) error
	disable func(*reasoningRequestControl, config.ReasoningFamilyConfig) error
}

var reasoningFamilyProjectors = map[string]reasoningFamilyProjector{
	config.ReasoningFamilyTypeChatTemplateKwargs: {
		enable:  enableChatTemplateReasoning,
		disable: disableChatTemplateReasoning,
	},
	config.ReasoningFamilyTypeReasoningMode: {
		enable:  enableReasoningMode,
		disable: disableReasoningMode,
	},
	config.ReasoningFamilyTypeReasoningEffort: {
		enable:  enableReasoningEffort,
		disable: disableReasoningEffort,
	},
	config.ReasoningFamilyTypeTopLevelReasoningEffort: {
		enable:  enableTopLevelReasoningEffort,
		disable: disableTopLevelReasoningEffort,
	},
}

func resolveExternalModelReasoningControl(
	families map[string]config.ReasoningFamilyConfig,
	external *config.ExternalModelConfig,
) (*reasoningRequestControl, error) {
	if external == nil || external.Reasoning == nil {
		return nil, nil
	}
	reasoning := external.Reasoning
	family, exists := families[reasoning.Family]
	if !exists {
		return nil, fmt.Errorf("reasoning family %q is not configured", reasoning.Family)
	}
	if reasoning.UseReasoning == nil {
		return nil, fmt.Errorf("reasoning use_reasoning is required")
	}
	projector, exists := reasoningFamilyProjectors[family.Type]
	if !exists {
		return nil, fmt.Errorf("unsupported reasoning family type %q", family.Type)
	}

	control := &reasoningRequestControl{}
	if *reasoning.UseReasoning {
		return control, projector.enable(control, family, reasoning.ReasoningEffort)
	}
	return control, projector.disable(control, family)
}

func enableChatTemplateReasoning(
	control *reasoningRequestControl,
	family config.ReasoningFamilyConfig,
	_ string,
) error {
	return control.setChatTemplate(family.Parameter, true)
}

func disableChatTemplateReasoning(
	control *reasoningRequestControl,
	family config.ReasoningFamilyConfig,
) error {
	return control.setChatTemplate(family.Parameter, false)
}

func enableReasoningMode(
	control *reasoningRequestControl,
	family config.ReasoningFamilyConfig,
	_ string,
) error {
	mode := enabledReasoningMode(family)
	if mode == "" {
		return nil
	}
	return control.setChatTemplate(family.Parameter, mode)
}

func disableReasoningMode(
	control *reasoningRequestControl,
	family config.ReasoningFamilyConfig,
) error {
	return control.setChatTemplate(family.Parameter, config.ReasoningModeDisabled)
}

func enableReasoningEffort(
	control *reasoningRequestControl,
	family config.ReasoningFamilyConfig,
	effort string,
) error {
	if effort == "" {
		effort = family.Default
	}
	if len(family.EffortFlags) > 0 {
		if err := control.setChatTemplate(family.ActivationParameter, true); err != nil {
			return err
		}
		if parameter := family.EffortFlags[effort]; parameter != "" {
			return control.setChatTemplate(parameter, true)
		}
		return nil
	}
	if effort != "" {
		if err := control.setChatTemplate(family.Parameter, effort); err != nil {
			return err
		}
	}
	if family.ActivationParameter != "" {
		return control.setChatTemplate(family.ActivationParameter, true)
	}
	return nil
}

func disableReasoningEffort(
	control *reasoningRequestControl,
	family config.ReasoningFamilyConfig,
) error {
	if len(family.EffortFlags) > 0 || family.ActivationParameter != "" {
		return control.setChatTemplate(family.ActivationParameter, false)
	}
	disabled := family.Disabled
	if disabled == "" {
		disabled = "none"
	}
	return control.setChatTemplate(family.Parameter, disabled)
}

func enableTopLevelReasoningEffort(
	control *reasoningRequestControl,
	family config.ReasoningFamilyConfig,
	effort string,
) error {
	if family.Parameter != "reasoning_effort" {
		return fmt.Errorf("top-level reasoning family parameter must be reasoning_effort")
	}
	if effort == "" {
		effort = family.Default
	}
	if effort == "" {
		return nil
	}
	return control.setReasoningEffort(effort)
}

func disableTopLevelReasoningEffort(
	control *reasoningRequestControl,
	family config.ReasoningFamilyConfig,
) error {
	if family.Disabled == "" {
		return fmt.Errorf("top-level reasoning family has no disabled value")
	}
	if family.Parameter != "reasoning_effort" {
		return fmt.Errorf("top-level reasoning family parameter must be reasoning_effort")
	}
	return control.setReasoningEffort(family.Disabled)
}

func enabledReasoningMode(family config.ReasoningFamilyConfig) string {
	if family.DefaultMode != "" && family.DefaultMode != config.ReasoningModeDisabled {
		return family.DefaultMode
	}
	for _, candidate := range []string{config.ReasoningModeAdaptive, config.ReasoningModeEnabled} {
		for _, supported := range family.Modes {
			if supported == candidate {
				return candidate
			}
		}
	}
	return ""
}

func (c *reasoningRequestControl) setReasoningEffort(value string) error {
	if c.reasoningEffort != nil {
		return fmt.Errorf("reasoning_effort is configured more than once")
	}
	c.reasoningEffort = &value
	return nil
}

func (c *reasoningRequestControl) setChatTemplate(key string, value interface{}) error {
	if key == "" {
		return fmt.Errorf("reasoning chat-template field cannot be empty")
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("marshal reasoning chat-template field %q: %w", key, err)
	}
	if c.chatTemplateKwargs == nil {
		c.chatTemplateKwargs = make(map[string]json.RawMessage)
	}
	if _, exists := c.chatTemplateKwargs[key]; exists {
		return fmt.Errorf("reasoning chat-template field %q is configured more than once", key)
	}
	c.chatTemplateKwargs[key] = encoded
	return nil
}
