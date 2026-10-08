package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// personalDataReplayAllowed uses the resolved request policy and verified PII
// evidence. Missing or incomplete detection never implies personal-data-free.
func (r *OpenAIRouter) personalDataReplayAllowed(ctx *RequestContext) bool {
	if ctx == nil {
		return false
	}
	if ctx.RouterReplayPluginConfig.CapturesPersonalData() {
		return true
	}
	// Tool schemas are captured but are not read by the text PII classifier.
	if ctx.SemanticRequest != nil && len(ctx.SemanticRequest.Tools) > 0 {
		return false
	}
	return ctx.PIIContentVerified && !ctx.PIIDetected
}

// omitReplayContent keeps a record's routing evidence, including which PII
// types matched, and drops everything the request or response said.
func omitReplayContent(record *routerreplay.RoutingRecord) {
	record.RequestBody, record.RequestBodyTruncated = "", false
	record.ResponseBody, record.ResponseBodyTruncated = "", false
	record.Prompt, record.PromptTruncated = "", false
	record.ToolDefinitions, record.ToolDefinitionsTruncated = "", false
	record.ToolTrace = nil
}

func (r *OpenAIRouter) effectiveReplayConfigForRequest(_ *RequestContext, decision *config.Decision) *config.RouterReplayPluginConfig {
	return r.Config.EffectiveRouterReplayConfig(decision)
}

// Startup has recipe-qualified decision references but no request context.
// Resolve each profile before considering a shared or isolated store.
func replayConfigForDecisionRef(cfg *config.RouterConfig, ref config.RoutingDecisionRef) *config.RouterReplayPluginConfig {
	return cfg.EffectiveRouterReplayConfig(ref.Decision)
}
