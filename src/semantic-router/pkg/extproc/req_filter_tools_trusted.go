package extproc

import (
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// trustedFactsRequestStage is the Looper role attributed to request-body tool
// selection. The request path proposes tool candidates before relevance or
// learned ranking; final-stage binding is a separate seam and stays denied
// unless the decision's stage_roles authorize it. Looper calls are gated per
// hop at the stage each call names (see looperHopStage).
const trustedFactsRequestStage = llmprotocol.TrustedStageCandidate

// trustedFactsNow is the clock for availability freshness; tests override it.
var trustedFactsNow = time.Now

// resolveTrustedFactsGate builds the distinct capability, authorization,
// availability, and stage-role facts for the llmprotocol eligibility gate. It
// reports false when the decision does not opt in to trusted facts, in which
// case the caller must leave the request untouched.
func resolveTrustedFactsGate(request *llmprotocol.Request, toolsCfg *config.ToolsPluginConfig, toolsAvailable bool, stage llmprotocol.TrustedStage) (llmprotocol.TrustedFacts, bool) {
	if request == nil || toolsCfg == nil || !toolsCfg.TrustedFactsEnabled() {
		return llmprotocol.TrustedFacts{}, false
	}
	trusted := toolsCfg.TrustedFacts
	return llmprotocol.TrustedFacts{
		Enforcement:   llmprotocol.TrustedEnforcement(trusted.EffectiveEnforcement()),
		Capable:       llmprotocol.RequiredCapabilities(*request).Supports(llmprotocol.CapabilityTools),
		Authorized:    trustedFactsAuthorized(trusted.TrustSources),
		Available:     toolsAvailable,
		Stage:         stage,
		AllowedStages: trustedFactsAllowedStages(trusted.StageRoles),
	}, true
}

// trustedFactsAuthorized reports whether the declared trust sources authorize
// tool use. Only operator-owned recipe policy authorizes by declaration.
// Gateway attestation must be verified per request and no verification signal
// is plumbed yet, so a gateway-attested declaration alone cannot authorize
// (deterministic partial-deployment behavior: deny). Runtime availability
// evidence can only narrow and never authorizes.
func trustedFactsAuthorized(sources []string) bool {
	for _, s := range sources {
		if llmprotocol.TrustedSource(s).Authorizes() {
			return true
		}
	}
	return false
}

// trustedFactsAllowedStages maps configured role names onto the gate
// vocabulary. Config validation guarantees membership; unknown values are
// dropped so they can never widen the allow-list.
func trustedFactsAllowedStages(roles []string) []llmprotocol.TrustedStage {
	allowed := make([]llmprotocol.TrustedStage, 0, len(roles))
	for _, r := range roles {
		switch llmprotocol.TrustedStage(r) {
		case llmprotocol.TrustedStageCandidate, llmprotocol.TrustedStageVerifier, llmprotocol.TrustedStageAdvisor, llmprotocol.TrustedStageFinal:
			allowed = append(allowed, llmprotocol.TrustedStage(r))
		}
	}
	return allowed
}

// applyTrustedFactsGate evaluates the trusted-facts eligibility gate for the
// requesting stage before relevance, learned ranking, or Looper execution and
// enforces the outcome on the request tool set. It returns true when the
// caller must stop: deny strips all tools and narrow keeps only explicitly
// policy-filtered tools with no retrieval expansion, so neither outcome can
// widen privileges. Allow and observe leave the request untouched.
func (r *OpenAIRouter) applyTrustedFactsGate(request *llmprotocol.Request, ctx *RequestContext, toolsCfg *config.ToolsPluginConfig, stage llmprotocol.TrustedStage) bool {
	if !toolsCfg.TrustedFactsEnabled() {
		return false
	}
	facts, ok := resolveTrustedFactsGate(request, toolsCfg, r.trustedFactsToolsAvailable(toolsCfg.TrustedFacts.FreshnessSeconds), stage)
	if !ok || facts.Enforcement == llmprotocol.TrustedDisabled {
		return false
	}
	outcome := llmprotocol.EvaluateTrustedFacts(facts)
	r.recordTrustedFactsOutcome(ctx, toolsCfg, facts, outcome)
	switch outcome {
	case llmprotocol.TrustedDeny, llmprotocol.TrustedNarrow:
		restrictToolsForTrustedOutcome(request, ctx, toolsCfg, outcome)
		return true
	default:
		return false
	}
}

// restrictToolsForTrustedOutcome enforces deny or narrow on top of the
// decision's configured tools policy, never in place of it. The gate returns
// before ordinary mode handling, so mode none strips here exactly what it
// strips when the gate allows: missing or stale availability evidence cannot
// keep tools the operator disabled. Deny then drops every remaining tool,
// including the Responses hosted image_generation tool that lives outside
// Request.Tools; its forced tool_choice falls to the no-tools cleanup. Narrow
// keeps only explicitly policy-filtered tools with no retrieval expansion.
func restrictToolsForTrustedOutcome(request *llmprotocol.Request, ctx *RequestContext, toolsCfg *config.ToolsPluginConfig, outcome llmprotocol.TrustedOutcome) {
	modeNone := toolsCfg.EffectiveMode() == config.ToolsPluginModeNone
	if modeNone {
		stripToolsForModeNone(request, ctx, toolsCfg)
	}
	switch {
	case outcome == llmprotocol.TrustedDeny:
		if len(request.Tools) > 0 || request.ImageGeneration != nil {
			request.Tools = nil
			request.ImageGeneration = nil
			request.Generation++
		}
	case !modeNone:
		filtered := filterToolsByDecisionPolicy(request.Tools, toolsCfg.AllowTools, toolsCfg.BlockTools)
		if len(filtered) != len(request.Tools) {
			request.Generation++
		}
		request.Tools = filtered
	}
	clearSemanticToolChoiceWhenNoTools(request)
	_ = commitToolSelection(request, ctx)
}

// trustedFactsToolsAvailable reports bounded runtime availability evidence:
// the tools database must be enabled and must have completed a successful
// load, and when freshnessSeconds is positive that load must be no older than
// the bound. An enabled flag alone, an unloaded database, or stale evidence
// is unavailable, which narrows and never allows.
func (r *OpenAIRouter) trustedFactsToolsAvailable(freshnessSeconds int) bool {
	if r == nil || r.ToolsDatabase == nil || !r.ToolsDatabase.IsEnabled() {
		return false
	}
	loadedAt := r.ToolsDatabase.LoadedAt()
	if loadedAt.IsZero() {
		return false
	}
	if freshnessSeconds <= 0 {
		return true
	}
	return trustedFactsNow().Sub(loadedAt) <= time.Duration(freshnessSeconds)*time.Second
}

// recordTrustedFactsOutcome records the bounded gate result: enforcement,
// stage, declared sources, and outcome only — never prompts, arguments,
// results, or credentials. The gate runs before Replay starts on both the
// ordinary and Looper paths, so without a record the outcome is retained on
// the request context and appended once Replay creates one. Recording never
// fails the request.
func (r *OpenAIRouter) recordTrustedFactsOutcome(ctx *RequestContext, toolsCfg *config.ToolsPluginConfig, facts llmprotocol.TrustedFacts, outcome llmprotocol.TrustedOutcome) {
	decision := ""
	if ctx != nil && ctx.VSRSelectedDecision != nil {
		decision = ctx.VSRSelectedDecision.Name
	}
	reason := routerreplay.NewTrustedFactsReason(string(facts.Enforcement), string(facts.Stage), append([]string(nil), toolsCfg.TrustedFacts.TrustSources...), string(outcome))
	logging.Infof("[ToolsPlugin] Decision %q trusted-facts outcome=%s enforcement=%s stage=%s", decision, reason.Outcome, reason.Enforcement, reason.Stage)
	if ctx == nil {
		return
	}
	replayOutcome := routerreplay.Outcome{
		Timestamp: time.Now().UTC(),
		Source:    "tools-plugin",
		Target:    "trusted_facts",
		Verdict:   string(outcome),
		Reason:    fmt.Sprintf("enforcement=%s stage=%s", reason.Enforcement, reason.Stage),
		Metadata: map[string]string{
			"enforcement": reason.Enforcement,
			"stage":       reason.Stage,
			"outcome":     reason.Outcome,
		},
	}
	if decision != "" {
		replayOutcome.Metadata["decision"] = decision
	}
	if ctx.RouterReplayID == "" {
		ctx.pendingTrustedFactsOutcomes = append(ctx.pendingTrustedFactsOutcomes, replayOutcome)
		return
	}
	recorder := ctx.RouterReplayRecorder
	if recorder == nil && r != nil {
		recorder = r.ReplayRecorder
	}
	appendTrustedFactsReplayOutcome(recorder, ctx.RouterReplayID, replayOutcome)
}

// appendPendingTrustedFactsOutcomes attaches gate outcomes retained before the
// Replay record existed. Each outcome is appended once.
func appendPendingTrustedFactsOutcomes(ctx *RequestContext, recorder *routerreplay.Recorder) {
	if ctx == nil || ctx.RouterReplayID == "" || len(ctx.pendingTrustedFactsOutcomes) == 0 {
		return
	}
	for _, outcome := range ctx.pendingTrustedFactsOutcomes {
		appendTrustedFactsReplayOutcome(recorder, ctx.RouterReplayID, outcome)
	}
	ctx.pendingTrustedFactsOutcomes = nil
}

func appendTrustedFactsReplayOutcome(recorder *routerreplay.Recorder, replayID string, outcome routerreplay.Outcome) {
	if recorder == nil {
		return
	}
	if err := recorder.AppendOutcome(replayID, outcome); err != nil {
		logging.Warnf("[ToolsPlugin] trusted-facts replay append failed: %v", err)
	}
}
