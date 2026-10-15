package pluginruntime

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// SkipReason is a typed, content-free reason a plugin did not run.
type SkipReason string

const (
	// SkipRequestDoesNotRequire: the request needs none of what the plugin acts on.
	SkipRequestDoesNotRequire SkipReason = "request_does_not_require"
	// SkipBackendLacksCapability: the backend declares capabilities and omits a task capability the plugin needs.
	SkipBackendLacksCapability SkipReason = "backend_lacks_capability"
)

// pluginRequirements lists what each gated plugin acts on. A plugin missing
// here is not gated and keeps running on configuration alone.
var pluginRequirements = map[string]llmprotocol.CapabilitySet{
	config.DecisionPluginRAG:    llmprotocol.Capabilities(llmprotocol.CapabilityText),
	config.DecisionPluginMemory: llmprotocol.Capabilities(llmprotocol.CapabilityText),
}

// Applicable reports whether a plugin runs: what it needs must be within the
// request's set, and its task capabilities within the backend's declared set.
// Text is assumed for every model, so only task capabilities gate the backend.
// An undeclared backend is not gated.
func Applicable(pluginType string, request, backend llmprotocol.CapabilitySet, backendDeclared bool) (bool, SkipReason) {
	required, gated := pluginRequirements[pluginType]
	if !gated {
		return true, ""
	}
	if !request.Contains(required) {
		return false, SkipRequestDoesNotRequire
	}
	if backendDeclared && !backend.Contains(required.TaskCapabilities()) {
		return false, SkipBackendLacksCapability
	}
	return true, ""
}
