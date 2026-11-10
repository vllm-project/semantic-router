package extproc

import (
	"fmt"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// applyToolsPlugin applies the selected decision's tools plugin to the
// provider-bound request: the trusted-facts gate, the policy, then the
// selection, which runs when the request leaves the choice of tools to the
// model. A selection that refuses the request turns response into the
// client's error.
func (r *OpenAIRouter) applyToolsPlugin(request *llmprotocol.Request, response *ext_proc.ProcessingResponse, ctx *RequestContext) {
	if response.GetImmediateResponse() != nil {
		return
	}
	plugin := resolveDecisionToolsConfig(ctx)
	if plugin != nil && plugin.Enabled && !r.applyTrustedFactsGate(request, ctx, plugin, trustedFactsRequestStage) {
		if applyToolsPolicy(request, plugin.Policy) && plugin.Select != nil && leavesToolChoiceToModel(request) {
			if failure := r.selectTools(request, plugin, ctx); failure != nil {
				response.Response = r.routingFailureResponse(ctx, *failure).Response
				return
			}
		}
	}
	clearSemanticToolChoiceWhenNoTools(request)
	if err := commitToolSelection(request, ctx); err != nil {
		logging.Errorf("Error applying the tools plugin: %v", err)
	}
}

// applyToolsPolicy applies the policy to the request's tools and reports
// whether the request may still carry tools: mode none removes them all.
func applyToolsPolicy(request *llmprotocol.Request, policy *config.ToolsPolicyConfig) bool {
	switch policy.EffectiveMode() {
	case config.ToolsPolicyModeNone:
		if changed, _ := stripSemanticToolPolicy(request, policy.StripToolHistory); changed {
			request.Generation++
		}
		return false
	case config.ToolsPolicyModeFiltered:
		if filtered := filterToolsByPolicy(request.Tools, policy); len(filtered) != len(request.Tools) {
			request.Tools = filtered
			request.Generation++
		}
	}
	return true
}

// leavesToolChoiceToModel reports whether the request lets the model decide
// whether and which tool to call: no tool_choice, or auto.
func leavesToolChoiceToModel(request *llmprotocol.Request) bool {
	return request.ToolChoice.Mode == "" || request.ToolChoice.Mode == llmprotocol.ToolChoiceAuto
}

// filterToolsByPolicy keeps the tools the policy allows, in their order.
func filterToolsByPolicy(tools []llmprotocol.Tool, policy *config.ToolsPolicyConfig) []llmprotocol.Tool {
	filtered := make([]llmprotocol.Tool, 0, len(tools))
	for _, tool := range tools {
		if policy.Allows(tool.Name) {
			filtered = append(filtered, tool)
		}
	}
	return filtered
}

func resolveDecisionToolsConfig(ctx *RequestContext) *config.ToolsPluginConfig {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return nil
	}
	return ctx.VSRSelectedDecision.GetToolsConfig()
}

// commitToolSelection publishes the mutated neutral request to the dispatch
// context. Provider serialization deliberately remains in
// finalizeProviderDispatchResponse, after every semantic plugin has run.
func commitToolSelection(request *llmprotocol.Request, ctx *RequestContext) error {
	if request == nil || ctx == nil {
		return fmt.Errorf("neutral inference request is unavailable")
	}
	ctx.SemanticRequest = request
	return nil
}
