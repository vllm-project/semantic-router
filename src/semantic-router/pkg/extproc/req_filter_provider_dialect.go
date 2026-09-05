package extproc

import (
	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// resolveProviderReasoningTransport reads provider request semantics from the
// compiled catalog. A nil or invalid internal profile safely retains the local
// template-kwargs behavior; validated user config cannot produce an invalid
// materialized profile.
func resolveProviderReasoningTransport(profile *config.ProviderProfile) modelcatalog.ReasoningTransport {
	if profile == nil {
		return modelcatalog.ReasoningTransportChatTemplate
	}
	transport, err := profile.ResolveReasoningTransport()
	if err != nil {
		return modelcatalog.ReasoningTransportChatTemplate
	}
	return transport
}

func usesTopLevelReasoningEffort(transport modelcatalog.ReasoningTransport) bool {
	return transport == modelcatalog.ReasoningTransportTopLevelEffort ||
		transport == modelcatalog.ReasoningTransportDeepSeekThinking
}

func usesThinkingObjectTransport(transport modelcatalog.ReasoningTransport) bool {
	return transport == modelcatalog.ReasoningTransportThinkingObject ||
		transport == modelcatalog.ReasoningTransportDeepSeekThinking
}

func usesReasoningObjectTransport(transport modelcatalog.ReasoningTransport) bool {
	return transport == modelcatalog.ReasoningTransportReasoningObject
}

func isDeepSeekThinkingTransport(transport modelcatalog.ReasoningTransport) bool {
	return transport == modelcatalog.ReasoningTransportDeepSeekThinking
}
