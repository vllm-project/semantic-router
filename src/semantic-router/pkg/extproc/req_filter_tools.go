package extproc

import (
	"fmt"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// applyToolsPlugin applies the tools plugin the request runs to the
// provider-bound request: the trusted-facts gate, the policy, then the
// selection, which runs when the request leaves the choice of tools to the
// model. A selection that refuses the request turns response into the
// client's error.
func (r *OpenAIRouter) applyToolsPlugin(request *llmprotocol.Request, response *ext_proc.ProcessingResponse, ctx *RequestContext) {
	if response.GetImmediateResponse() != nil {
		return
	}
	plugin := resolveDecisionToolsConfig(r.Config, ctx)
	if plugin != nil && plugin.Enabled && !r.applyTrustedFactsGate(request, ctx, plugin, trustedFactsRequestStage) {
		if failure := r.applyToolsPolicyAndSelection(request, plugin, ctx); failure != nil {
			response.Response = r.routingFailureResponse(ctx, *failure).Response
			return
		}
	}
	clearSemanticToolChoiceWhenNoTools(request)
	if err := commitToolSelection(request, ctx); err != nil {
		logging.Errorf("Error applying the tools plugin: %v", err)
	}
}

// applyLooperToolsPlugin applies the decision's tools policy and selection to
// the request every Looper model call is built from, as applyToolsPlugin
// does to a single-model request. The trusted-facts gate stays per call
// (looperToolStageGate); under it a selection adds no catalog tools, since a
// call it narrows keeps only the request's own allowed tools.
func (r *OpenAIRouter) applyLooperToolsPlugin(request *llmprotocol.Request, ctx *RequestContext) *ext_proc.ProcessingResponse {
	plugin := resolveDecisionToolsConfig(r.Config, ctx)
	if plugin == nil || !plugin.Enabled {
		return nil
	}
	if plugin.TrustedFactsEnabled() && plugin.Select != nil && plugin.Select.Mode == config.ToolsSelectModeAdd {
		applyToolsPolicy(request, plugin.Policy)
	} else if failure := r.applyToolsPolicyAndSelection(request, plugin, ctx); failure != nil {
		return r.routingFailureResponse(ctx, *failure)
	}
	clearSemanticToolChoiceWhenNoTools(request)
	return nil
}

// applyToolsPolicyAndSelection applies the policy, then the selection when
// the request leaves the choice of tools to the model. It returns the failure
// to answer the client with when the selection refuses the request.
func (r *OpenAIRouter) applyToolsPolicyAndSelection(request *llmprotocol.Request, plugin *config.ToolsPluginConfig, ctx *RequestContext) *routingFailure {
	if !applyToolsPolicy(request, plugin.Policy) || plugin.Select == nil || !leavesToolChoiceToModel(request) {
		return nil
	}
	return r.selectTools(request, plugin, ctx)
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
			// A tool_choice naming a removed tool goes with it, as under none,
			// and the model chooses among the tools left.
			if request.ToolChoice.Mode == llmprotocol.ToolChoiceNamed && !policy.Allows(request.ToolChoice.Name) {
				request.ToolChoice = llmprotocol.ToolChoice{}
			}
			request.Generation++
		}
		// The hosted image_generation tool lives outside Tools and goes by its
		// type, the word its tool_choice uses too.
		if request.ImageGeneration != nil && !policy.Allows(string(llmprotocol.ToolChoiceImageGeneration)) {
			request.ImageGeneration = nil
			if request.ToolChoice.Mode == llmprotocol.ToolChoiceImageGeneration {
				request.ToolChoice = llmprotocol.ToolChoice{}
			}
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

func resolveDecisionToolsConfig(cfg *config.RouterConfig, ctx *RequestContext) *config.ToolsPluginConfig {
	if ctx == nil {
		return nil
	}
	return config.ToolsPluginOf(requestPlugins(cfg, ctx))
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
