package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
)

// pluginSkip is the content-free record of a plugin that did not run.
type pluginSkip struct {
	Plugin string
	Reason pluginruntime.SkipReason
}

// pluginApplies gates a decision plugin on what the request needs and what the
// selected backend declares. A skip is recorded once per plugin and request.
func (r *OpenAIRouter) pluginApplies(ctx *RequestContext, pluginType string) (bool, pluginruntime.SkipReason) {
	if ctx == nil || ctx.SemanticRequest == nil {
		return true, ""
	}
	declared, annotated := r.declaredModelCapabilities(ctx.PluginBackendModel)
	apply, reason := pluginruntime.Applicable(pluginType, llmprotocol.RequiredCapabilities(*ctx.SemanticRequest), declared, annotated)
	if apply {
		return true, ""
	}
	for _, skip := range ctx.PluginSkips {
		if skip.Plugin == pluginType {
			return false, reason
		}
	}
	ctx.PluginSkips = append(ctx.PluginSkips, pluginSkip{Plugin: pluginType, Reason: reason})
	metrics.RecordPluginExecution(pluginType, requestDecisionStateKey(ctx), "skipped_"+string(reason), 0)
	logging.ComponentDebugEvent("extproc", "plugin_skipped", map[string]interface{}{
		"request_id": ctx.RequestID,
		"plugin":     pluginType,
		"reason":     string(reason),
	})
	return false, reason
}
