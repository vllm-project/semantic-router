package extproc

import (
	"fmt"
	"slices"
	"sync"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// trustedFactsRequestStage is the Looper role attributed to request-body tool
// selection. The request path proposes tool candidates before relevance or
// learned ranking; final-stage binding is a separate seam and stays denied
// unless the decision's stage_roles authorize it.
const trustedFactsRequestStage = llmprotocol.TrustedStageCandidate

// trustedFactsNow is the clock for availability freshness; tests override it.
var trustedFactsNow = time.Now

// resolveTrustedFactsGate builds the distinct capability, authorization,
// availability, and stage-role facts for the llmprotocol eligibility gate. It
// reports false when the decision does not opt in to trusted facts, in which
// case the caller must leave the request untouched.
func resolveTrustedFactsGate(request *llmprotocol.Request, toolsCfg *config.ToolsPluginConfig, toolsAvailable bool, stage llmprotocol.TrustedStage) (llmprotocol.TrustedFacts, bool) {
	if request == nil || !toolsCfg.TrustedFactsEnabled() {
		return llmprotocol.TrustedFacts{}, false
	}
	trusted := toolsCfg.Policy.TrustedFacts
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
// widen privileges. Both apply the decision's mode none first. Allow and
// observe leave the request untouched.
func (r *OpenAIRouter) applyTrustedFactsGate(request *llmprotocol.Request, ctx *RequestContext, toolsCfg *config.ToolsPluginConfig, stage llmprotocol.TrustedStage) bool {
	if !toolsCfg.TrustedFactsEnabled() {
		return false
	}
	policy := toolsCfg.Policy
	facts, ok := resolveTrustedFactsGate(request, toolsCfg, r.trustedFactsToolsAvailable(toolsCfg), stage)
	if !ok || facts.Enforcement == llmprotocol.TrustedDisabled {
		return false
	}
	outcome := llmprotocol.EvaluateTrustedFacts(facts)
	r.recordTrustedFactsOutcome(ctx, toolsCfg, facts, outcome)
	if outcome != llmprotocol.TrustedDeny && outcome != llmprotocol.TrustedNarrow {
		return false
	}
	// Returning here skips ordinary policy handling, so mode none is applied
	// first with the same strip it uses there. Deny and narrow then restrict
	// the configured policy further and never keep tools it removes.
	modeNone := policy.EffectiveMode() == config.ToolsPolicyModeNone
	if modeNone {
		if changed, _ := stripSemanticToolPolicy(request, policy.StripToolHistory); changed {
			request.Generation++
		}
	}
	switch {
	case outcome == llmprotocol.TrustedDeny:
		// Responses hosted image_generation lives outside Request.Tools, so
		// deny clears it too; its forced tool_choice then falls to the
		// no-tools cleanup below.
		if len(request.Tools) > 0 || request.ImageGeneration != nil {
			request.Tools = nil
			request.ImageGeneration = nil
			request.Generation++
		}
	case !modeNone:
		applyToolsPolicy(request, policy)
	}
	clearSemanticToolChoiceWhenNoTools(request)
	// A forced choice naming a tool narrowing removed would fail protocol
	// validation, so it is cleared; a choice whose tool survived is kept.
	if request.ToolChoice.Mode == llmprotocol.ToolChoiceNamed &&
		!slices.ContainsFunc(request.Tools, func(t llmprotocol.Tool) bool { return t.Name == request.ToolChoice.Name }) {
		request.ToolChoice = llmprotocol.ToolChoice{}
		request.Generation++
	}
	_ = commitToolSelection(request, ctx)
	return true
}

// looperToolStageGate judges each tool-bearing Looper model attempt by the
// decision's tools policy, then by its trusted facts. It returns nil when the
// decision neither restricts tools nor opts in to trusted facts.
func (r *OpenAIRouter) looperToolStageGate(request *llmprotocol.Request, decision *config.Decision, reqCtx *RequestContext) looper.ToolStageGate {
	if request == nil || decision == nil {
		return nil
	}
	toolsCfg := decision.GetToolsConfig()
	policy := looperToolPolicyGate(toolsCfg.ActivePolicy())
	trusted := r.looperTrustedFactsGate(request, toolsCfg, reqCtx)
	switch {
	case trusted == nil:
		return policy
	case policy == nil:
		return trusted
	}
	return func(stage llmprotocol.TrustedStage, req *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams {
		return trusted(stage, policy(stage, req))
	}
}

// looperToolPolicyGate holds each attempt to the decision's tools policy. A
// fresh request was narrowed before Looper ran; a paused workflow resumes
// from the requests it stored, which the policy in force now may narrow
// further.
func looperToolPolicyGate(policy *config.ToolsPolicyConfig) looper.ToolStageGate {
	switch policy.EffectiveMode() {
	case config.ToolsPolicyModeNone:
		return func(_ llmprotocol.TrustedStage, req *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams {
			return looper.WithoutToolUse(req)
		}
	case config.ToolsPolicyModeFiltered:
		return func(_ llmprotocol.TrustedStage, req *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams {
			return looper.NarrowToolUse(req, policy.AllowTools, policy.BlockTools)
		}
	}
	return nil
}

// looperTrustedFactsGate judges each attempt at its own trusted-facts stage,
// so a policy that authorizes only final never sends tools on a candidate or
// verifier call, and one that authorizes candidate does not widen the final
// stage. An attempt that names no stage matches no role, so an authoritative
// policy denies it rather than guessing. Each stage's outcome is evaluated
// and recorded once, then reused by concurrent attempts. It returns nil when
// the decision does not opt in.
func (r *OpenAIRouter) looperTrustedFactsGate(request *llmprotocol.Request, toolsCfg *config.ToolsPluginConfig, reqCtx *RequestContext) looper.ToolStageGate {
	if !toolsCfg.TrustedFactsEnabled() {
		return nil
	}
	base, ok := resolveTrustedFactsGate(request, toolsCfg, r.trustedFactsToolsAvailable(toolsCfg), "")
	if !ok || base.Enforcement == llmprotocol.TrustedDisabled {
		return nil
	}
	var mu sync.Mutex
	outcomes := make(map[llmprotocol.TrustedStage]llmprotocol.TrustedOutcome, 4)
	return func(stage llmprotocol.TrustedStage, req *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams {
		mu.Lock()
		outcome, seen := outcomes[stage]
		if !seen {
			facts := base
			facts.Stage = stage
			outcome = llmprotocol.EvaluateTrustedFacts(facts)
			outcomes[stage] = outcome
			r.recordTrustedFactsOutcome(reqCtx, toolsCfg, facts, outcome)
		}
		mu.Unlock()
		switch outcome {
		case llmprotocol.TrustedDeny:
			return looper.WithoutToolUse(req)
		case llmprotocol.TrustedNarrow:
			return looper.NarrowToolUse(req, toolsCfg.Policy.AllowTools, toolsCfg.Policy.BlockTools)
		}
		return req
	}
}

// trustedFactsToolsAvailable reports bounded runtime availability evidence:
// the decision's tool catalog, which its selection adds tools from, must
// have been read, and when freshness_seconds is positive that read must be
// no older than the bound. A decision without a catalog, or stale evidence,
// is unavailable, which narrows and never allows.
func (r *OpenAIRouter) trustedFactsToolsAvailable(toolsCfg *config.ToolsPluginConfig) bool {
	spec := toolsCfg.ActiveSelect()
	if r == nil || spec == nil || spec.Mode != config.ToolsSelectModeAdd {
		return false
	}
	catalog := r.toolCatalogs[spec.ToolsDBPath]
	if catalog == nil {
		return false
	}
	freshnessSeconds := toolsCfg.Policy.TrustedFacts.FreshnessSeconds
	if freshnessSeconds <= 0 {
		return true
	}
	return trustedFactsNow().Sub(catalog.loadedAt) <= time.Duration(freshnessSeconds)*time.Second
}

// recordTrustedFactsOutcome records the bounded gate result: enforcement,
// stage, declared sources, and outcome only — never prompts, arguments,
// results, or credentials. Recording never fails the request.
func (r *OpenAIRouter) recordTrustedFactsOutcome(ctx *RequestContext, toolsCfg *config.ToolsPluginConfig, facts llmprotocol.TrustedFacts, outcome llmprotocol.TrustedOutcome) {
	decision := ""
	if ctx != nil && ctx.VSRSelectedDecision != nil {
		decision = ctx.VSRSelectedDecision.Name
	}
	reason := routerreplay.NewTrustedFactsReason(string(facts.Enforcement), string(facts.Stage), append([]string(nil), toolsCfg.Policy.TrustedFacts.TrustSources...), string(outcome))
	logging.Infof("[ToolsPlugin] Decision %q trusted-facts outcome=%s enforcement=%s stage=%s", decision, reason.Outcome, reason.Enforcement, reason.Stage)
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
	r.recordToolsOutcome(ctx, replayOutcome)
}

// recordToolsOutcome adds an outcome of the tools plugin to the request's
// replay record. The plugin runs before Replay starts on both the ordinary
// and Looper paths, so without a record the outcome is retained on the
// request context and appended once Replay creates one.
func (r *OpenAIRouter) recordToolsOutcome(ctx *RequestContext, outcome routerreplay.Outcome) {
	if ctx == nil {
		return
	}
	if ctx.RouterReplayID == "" {
		ctx.pendingToolsOutcomes = append(ctx.pendingToolsOutcomes, outcome)
		return
	}
	recorder := ctx.RouterReplayRecorder
	if recorder == nil && r != nil {
		recorder = r.ReplayRecorder
	}
	appendToolsReplayOutcome(recorder, ctx.RouterReplayID, outcome)
}

// appendPendingToolsOutcomes attaches tools plugin outcomes retained before
// the Replay record existed. Each outcome is appended once.
func appendPendingToolsOutcomes(ctx *RequestContext, recorder *routerreplay.Recorder) {
	if ctx == nil || ctx.RouterReplayID == "" || len(ctx.pendingToolsOutcomes) == 0 {
		return
	}
	for _, outcome := range ctx.pendingToolsOutcomes {
		appendToolsReplayOutcome(recorder, ctx.RouterReplayID, outcome)
	}
	ctx.pendingToolsOutcomes = nil
}

func appendToolsReplayOutcome(recorder *routerreplay.Recorder, replayID string, outcome routerreplay.Outcome) {
	if recorder == nil {
		return
	}
	if err := recorder.AppendOutcome(replayID, outcome); err != nil {
		logging.Warnf("[ToolsPlugin] %s replay append failed: %v", outcome.Target, err)
	}
}
