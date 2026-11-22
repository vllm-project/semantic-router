package extproc

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// serverToolProviders are the catalog providers whose Messages backends take
// Anthropic server tools: Anthropic's own API, and an endpoint that names
// itself Anthropic-compatible, such as a gateway in front of it. A model
// server with a provider of its own, such as vllm, speaks the wire format
// without running them.
var serverToolProviders = []string{"anthropic", "anthropic-compatible"}

// projectServerToolsForBackend keeps Anthropic server tools for a backend that
// runs them. Any other backend would ignore or reject the tool, so a request
// that declares one is refused by name before it is sent, and selection and
// fallback pass over that backend for one that runs it. History the tools
// left in an earlier turn is sent without them instead.
func (r *OpenAIRouter) projectServerToolsForBackend(
	request llmprotocol.Request,
	model string,
	target llmprotocol.WireFormat,
) (llmprotocol.Request, llmprotocol.Diagnostics, error) {
	if r == nil || r.Config == nil || !llmprotocol.UsesServerTools(request) {
		return request, nil, nil
	}
	provider := r.backendProviderType(model)
	if target == llmprotocol.AnthropicMessagesV1 && slices.Contains(serverToolProviders, provider) {
		return request, nil, nil
	}
	for _, tool := range request.Tools {
		if tool.Kind == llmprotocol.ToolKindServer {
			return request, nil, serverToolRefusal(tool, model, provider, target)
		}
	}
	projected, diagnostics := llmprotocol.ProjectServerToolHistory(request, target)
	return projected, diagnostics, nil
}

func (r *OpenAIRouter) backendProviderType(model string) string {
	_, backend, found, err := r.Config.ResolvePrimaryBackendForModel(model)
	if err != nil || !found {
		return ""
	}
	profile, err := r.Config.GetProviderProfileForEndpoint(backend)
	if err != nil || profile == nil {
		return ""
	}
	return profile.Type
}

func serverToolRefusal(tool llmprotocol.Tool, model, provider string, target llmprotocol.WireFormat) error {
	var definition struct {
		Type string `json:"type"`
	}
	_ = json.Unmarshal(tool.Raw, &definition)
	served := "provider " + provider
	if provider == "" {
		served = "a backend with no provider"
	}
	return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_server_tool", fmt.Sprintf(
		"Anthropic tool %q (%s) runs only on an Anthropic backend, and model %q is served by %s over %s",
		tool.Name, definition.Type, model, served, target), nil)
}
