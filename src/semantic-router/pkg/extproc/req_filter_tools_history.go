package extproc

import (
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// applyPreDispatchToolsPolicy mutates only neutral semantic state. Every client
// and backend format observes the same decision policy through its codec.
func (r *OpenAIRouter) applyPreDispatchToolsPolicy(
	ctx *RequestContext,
) (bool, error) {
	toolsCfg := resolveDecisionToolsConfig(ctx)
	request := ctx.SemanticRequest
	if request == nil || toolsCfg == nil || !toolsCfg.Enabled ||
		toolsCfg.EffectiveMode() != config.ToolsPluginModeNone {
		return false, nil
	}
	changed, removed := stripSemanticToolPolicy(request, toolsCfg.StripToolHistory)
	if changed {
		request.Generation++
	}
	if removed > 0 {
		logging.Infof("[ToolsPlugin] Decision %q stripped %d prior tool-history messages or blocks", ctx.VSRSelectedDecision.Name, removed)
	}
	return changed, nil
}

func stripSemanticToolPolicy(request *llmprotocol.Request, stripHistory bool) (bool, int) {
	return llmprotocol.StripTools(request, stripHistory)
}

// cloneMessagesForToolStrip detaches exactly what StripTools overwrites. That
// filter compacts each Content slice in place, so a preview needs its own
// message and content backing arrays; it never mutates a block's own fields,
// and drops tool blocks whole, so the deep memory-snapshot clone is not needed.
func cloneMessagesForToolStrip(messages []llmprotocol.Message) []llmprotocol.Message {
	result := slices.Clone(messages)
	for index := range result {
		result[index].Content = slices.Clone(messages[index].Content)
	}
	return result
}

func clearSemanticToolChoiceWhenNoTools(request *llmprotocol.Request) bool {
	if request == nil || len(request.Tools) > 0 || request.ImageGeneration != nil || request.ToolChoice.Mode == "" {
		return false
	}
	request.ToolChoice = llmprotocol.ToolChoice{}
	request.ParallelToolCalls = nil
	request.Generation++
	return true
}
