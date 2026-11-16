package extproc

import (
	"context"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// replayPromptAllowed reports whether the record may keep the request's exact
// prompt when it is written: always under raw capture, never under none, and
// under verified only where the routing evidence already read it whole and
// found it clean. It concerns only the prompt, not the whole body, tool
// arguments, history or a response generated later.
func (r *OpenAIRouter) replayPromptAllowed(ctx *RequestContext) bool {
	if ctx == nil {
		return false
	}
	policy := ctx.RouterReplayPluginConfig
	if policy.CapturesRawContent() {
		return true
	}
	if !policy.VerifiesPrompt() {
		return false
	}
	prompt := replayPrompt(ctx, ctx.SemanticRequest)
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

// Under verified capture, a record whose prompt no routing evidence covers is
// written without it; the prompt is read after routing and attached once found
// clean, so a request never waits for Replay's privacy check. none and raw
// capture read no prompt for personal data. At most replayPromptChecks run at
// once, and a record whose check cannot start keeps its prompt withheld. Nor
// does a check start that could not read the prompt whole in its time: it
// would end incomplete_read and withhold the prompt anyway, after taking the
// decision model from the requests that follow.
const (
	replayPromptChecks       = 2
	replayPromptCheckTimeout = 30 * time.Second
)

var replayPromptCheckSlots = make(chan struct{}, replayPromptChecks)

// replayPromptEvidence reads a prompt with a recipe's PII rules (tests replace it).
var replayPromptEvidence = (*classification.Classifier).ReplayPromptEvidence

// replayPromptReadable reports whether that read could finish in time (tests replace it).
var replayPromptReadable = (*classification.Classifier).ReplayPromptReadable

// attachPromptOnceVerified reads the request's exact prompt in the background
// and attaches stored, the prompt the record withheld, when that read is
// complete and clean.
func (r *OpenAIRouter) attachPromptOnceVerified(ctx *RequestContext, recorder *routerreplay.Recorder, stored string) {
	if !ctx.RouterReplayPluginConfig.VerifiesPrompt() {
		return
	}
	classifier := r.classifierForRequest(ctx)
	prompt := replayPrompt(ctx, ctx.SemanticRequest)
	if classifier == nil || classifier.Config == nil || !classifier.Config.ReplayNeedsPIIEvidence() || prompt == "" {
		return
	}
	if !replayPromptReadable(classifier, prompt, replayPromptCheckTimeout, guardReadShort(ctx)) {
		logging.ComponentDebugEvent("extproc", "router_replay_prompt_check_skipped", map[string]interface{}{
			"request_id": ctx.RequestID, "reason": "unreadable_in_time",
		})
		return
	}
	select {
	case replayPromptCheckSlots <- struct{}{}:
	default:
		logging.ComponentDebugEvent("extproc", "router_replay_prompt_check_skipped", map[string]interface{}{
			"request_id": ctx.RequestID, "reason": "checks_busy",
		})
		return
	}
	id, requestID := ctx.RouterReplayID, ctx.RequestID
	go func() {
		defer func() { <-replayPromptCheckSlots }()
		checkCtx, cancel := context.WithTimeout(context.Background(), replayPromptCheckTimeout)
		defer cancel()
		evidence := replayPromptEvidence(classifier, checkCtx, prompt)
		if !evidence.CoversClean("request", prompt) {
			reason := "personal_data"
			if evidence.Coverage != "complete" {
				reason = "incomplete_read"
			}
			logging.ComponentEvent("extproc", "router_replay_prompt_withheld", map[string]interface{}{
				"request_id": requestID, "reason": reason,
			})
			return
		}
		if err := recorder.AttachPrompt(id, stored); err != nil {
			logging.ComponentErrorEvent("extproc", "router_replay_prompt_attach_failed", map[string]interface{}{
				"request_id": requestID, "error": err.Error(),
			})
		}
	}()
}

// guardReadShort reports whether the request's guard scans ran out of time on
// its input: a guard rule read it only in part, and no guard rule matched,
// which would have ended the scans early instead.
func guardReadShort(ctx *RequestContext) bool {
	if len(ctx.VSRMatchedJailbreak)+len(ctx.VSRMatchedSafety)+len(ctx.VSRMatchedPII) > 0 {
		return false
	}
	for _, coverage := range ctx.VSRScanCoverage {
		if strings.HasPrefix(coverage, "partial") {
			return true
		}
	}
	return false
}

// Only raw capture keeps response text. No existing response-stage task scans
// personal data, and a request-stage clean result never grants access to
// generated text; do not add inference for Replay.
func replayResponseContentAllowed(ctx *RequestContext) bool {
	return ctx != nil && ctx.RouterReplayPluginConfig.CapturesRawContent()
}

// applyReplayPrivacyEvidence leaves a raw capture's record as it is. Verified
// and none keep the prompt alone, where allowed, and none of the rest.
func applyReplayPrivacyEvidence(ctx *RequestContext, record *routerreplay.RoutingRecord, promptAllowed bool) {
	if ctx.RouterReplayPluginConfig.CapturesRawContent() {
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

// effectiveReplayConfigForRequest is the replay policy of the request's
// decision, or, for a request that matched none, of its recipe's defaults.
func (r *OpenAIRouter) effectiveReplayConfigForRequest(ctx *RequestContext, decision *config.Decision) *config.RouterReplayPluginConfig {
	if decision != nil {
		return config.RouterReplayPolicyOf(decision.Plugins)
	}
	return config.RouterReplayPolicyOf(requestPlugins(r.Config, ctx))
}

// Startup has recipe-qualified decision references but no request context.
// Resolve each profile before considering a shared or isolated store.
func replayConfigForDecisionRef(cfg *config.RouterConfig, ref config.RoutingDecisionRef) *config.RouterReplayPluginConfig {
	return cfg.EffectiveRouterReplayConfig(ref.Decision)
}
