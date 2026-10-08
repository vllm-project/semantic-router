package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// personalDataReplayAllowed concerns only the exact prompt, not the whole
// body, tool arguments, history or a response generated later.
func (r *OpenAIRouter) personalDataReplayAllowed(ctx *RequestContext) bool {
	if ctx == nil {
		return false
	}
	if ctx.RouterReplayPluginConfig.CapturesPersonalData() {
		return true
	}
	prompt, _ := extractSemanticPromptAndTools(ctx.SemanticRequest)
	if prompt == "" {
		return false
	}
	for _, evidence := range ctx.PIIEvidence {
		if evidence.CoversClean("request", prompt) {
			return true
		}
	}
	return false
}

// No existing response-stage task scans personal data. A request-stage clean
// result never grants access to generated text; do not add inference for Replay.
func replayResponseContentAllowed(ctx *RequestContext) bool {
	return ctx != nil && ctx.RouterReplayPluginConfig.CapturesPersonalData()
}

func applyReplayPrivacyEvidence(ctx *RequestContext, record *routerreplay.RoutingRecord, promptAllowed bool) {
	if ctx.RouterReplayPluginConfig.CapturesPersonalData() {
		return
	}
	// Raw bodies contain arbitrary metadata and structured fields outside the
	// text task input. Tools and response excerpts have no matching evidence.
	record.RequestBody, record.RequestBodyTruncated = "", false
	record.ResponseBody, record.ResponseBodyTruncated = "", false
	record.ToolDefinitions, record.ToolDefinitionsTruncated = "", false
	record.ToolTrace = nil
	if !promptAllowed {
		record.Prompt, record.PromptTruncated = "", false
	}
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
