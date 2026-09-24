package extproc

import (
	"encoding/json"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// usesChatTemplateEffortFlags identifies effort ladders whose local model
// template represents each non-default level as an independent boolean flag.
// Provider-native transports still receive their normal string effort shape.
func usesChatTemplateEffortFlags(
	family *config.ReasoningFamilyConfig,
	transport modelcatalog.ReasoningTransport,
) bool {
	return family != nil &&
		family.Type == config.ReasoningFamilyTypeReasoningEffort &&
		len(family.EffortFlags) > 0 &&
		transport == modelcatalog.ReasoningTransportChatTemplate
}

func applyChatTemplateEffortFlagsMutation(
	mutation *reasoningRequestMutation,
	family *config.ReasoningFamilyConfig,
	enabled bool,
	effort string,
) {
	prepareChatTemplateReasoningMutation(mutation, family.Parameter)
	removeChatTemplateReasoningValue(mutation, family.Parameter)
	removeChatTemplateReasoningValue(mutation, family.ActivationParameter)
	for _, parameter := range family.EffortFlags {
		removeChatTemplateReasoningValue(mutation, parameter)
	}

	activation := json.RawMessage("false")
	if enabled {
		activation = json.RawMessage("true")
	}
	setChatTemplateReasoningValue(mutation, family.ActivationParameter, activation)
	if enabled {
		if parameter := family.EffortFlags[effort]; parameter != "" {
			setChatTemplateReasoningValue(mutation, parameter, json.RawMessage("true"))
		}
	}

	mutation.appliedEffort = effort
	mutation.reasoningApplied = true
}
