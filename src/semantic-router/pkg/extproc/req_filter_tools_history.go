package extproc

import (
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
	if request == nil {
		return false, 0
	}
	changed := len(request.Tools) > 0 || request.ToolChoice.Mode != "" ||
		request.ToolChoice.Name != "" || request.ParallelToolCalls != nil
	request.Tools = nil
	request.ToolChoice = llmprotocol.ToolChoice{}
	request.ParallelToolCalls = nil
	if !stripHistory {
		return changed, 0
	}
	filtered := make([]llmprotocol.Message, 0, len(request.Messages))
	removed := 0
	for _, message := range request.Messages {
		if message.Role == llmprotocol.RoleTool {
			removed++
			changed = true
			continue
		}
		content := message.Content[:0]
		for _, block := range message.Content {
			if block.Kind == llmprotocol.ContentToolCall || block.Kind == llmprotocol.ContentToolResult {
				removed++
				changed = true
				continue
			}
			content = append(content, block)
		}
		if len(content) == 0 {
			if len(message.Content) > 0 {
				removed++
				changed = true
			}
			continue
		}
		message.Content = content
		filtered = append(filtered, message)
	}
	request.Messages = filtered
	return changed, removed
}

func clearSemanticToolChoiceWhenNoTools(request *llmprotocol.Request) bool {
	if request == nil || len(request.Tools) > 0 || request.ToolChoice.Mode == "" {
		return false
	}
	request.ToolChoice = llmprotocol.ToolChoice{}
	request.ParallelToolCalls = nil
	request.Generation++
	return true
}
