package config

const (
	ReasoningFamilyTypeChatTemplateKwargs = "chat_template_kwargs"
	ReasoningFamilyTypeReasoningEffort    = "reasoning_effort"
	ReasoningFamilyTypeReasoningMode      = "reasoning_mode"
	// ReasoningFamilyTypeTopLevelReasoningEffort is retained for authored
	// legacy config. Built-in families use reasoning_effort; provider profiles
	// own the outbound wire placement.
	ReasoningFamilyTypeTopLevelReasoningEffort = "top_level_reasoning_effort"
)
