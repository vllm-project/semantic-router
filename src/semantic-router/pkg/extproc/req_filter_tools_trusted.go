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
// unless the decision's stage_roles authorize it.
const trustedFactsRequestStage = llmprotocol.TrustedStageCandidate

// resolveTrustedFactsGate builds the distinct capability, authorization,
// availability, and stage-role facts for the llmprotocol eligibility gate. It
// reports false when the decision does not opt in to trusted facts, in which
// case the caller must leave the request untouched.
func resolveTrustedFactsGate(request *llmprotocol.Request, toolsCfg *config.ToolsPluginConfig, toolsAvailable bool) (llmprotocol.TrustedFacts, bool) {
	if request == nil || toolsCfg == nil || !toolsCfg.TrustedFactsEnabled() {
		return llmprotocol.TrustedFacts{}, false
	}
	trusted := toolsCfg.TrustedFacts
	return llmprotocol.TrustedFacts{
		Enforcement:   llmprotocol.TrustedEnforcement(trusted.EffectiveEnforcement()),
		Capable:       llmprotocol.RequiredCapabilities(*request).Supports(llmprotocol.CapabilityTools),
		Authorized:    trustedFactsAuthorized(trusted.TrustSources),
		Available:     toolsAvailable,
		Stage:         trustedFactsRequestStage,
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

// applyTrustedFactsGate evaluates the trusted-facts eligibility gate before
// relevance or learned ranking and enforces the outcome on the request tool
// set. It returns true when the caller must stop: deny strips all tools and
// narrow keeps only explicitly policy-filtered tools with no retrieval
// expansion, so neither outcome can widen privileges. Allow and observe leave
// the request untouched.
func (r *OpenAIRouter) applyTrustedFactsGate(request *llmprotocol.Request, ctx *RequestContext, toolsCfg *config.ToolsPluginConfig) bool {
	facts, ok := resolveTrustedFactsGate(request, toolsCfg, r.trustedFactsToolsAvailable())
	if !ok || facts.Enforcement == llmprotocol.TrustedDisabled {
		return false
	}
	outcome := llmprotocol.EvaluateTrustedFacts(facts)
	r.recordTrustedFactsOutcome(ctx, toolsCfg, facts, outcome)
	switch outcome {
	case llmprotocol.TrustedDeny:
		if len(request.Tools) > 0 {
			request.Tools = nil
			request.Generation++
		}
		_ = commitToolSelection(request, ctx)
		return true
	case llmprotocol.TrustedNarrow:
		filtered := filterToolsByDecisionPolicy(request.Tools, toolsCfg.AllowTools, toolsCfg.BlockTools)
		if len(filtered) != len(request.Tools) {
			request.Generation++
		}
		request.Tools = filtered
		_ = commitToolSelection(request, ctx)
		return true
	default:
		return false
	}
}

// trustedFactsToolsAvailable reports live tools-database availability: the
// request-time liveness probe is fresh by construction. FreshnessSeconds
// bounds cached availability snapshots once such evidence exists; until then
// only a live database counts as available.
func (r *OpenAIRouter) trustedFactsToolsAvailable() bool {
	return r != nil && r.ToolsDatabase != nil && r.ToolsDatabase.IsEnabled()
}

// recordTrustedFactsOutcome records the bounded gate result: enforcement,
// stage, declared sources, and outcome only — never prompts, arguments,
// results, or credentials. The outcome is appended to the replay record when
// one exists; recording never fails the request.
func (r *OpenAIRouter) recordTrustedFactsOutcome(ctx *RequestContext, toolsCfg *config.ToolsPluginConfig, facts llmprotocol.TrustedFacts, outcome llmprotocol.TrustedOutcome) {
	decision := ""
	if ctx != nil && ctx.VSRSelectedDecision != nil {
		decision = ctx.VSRSelectedDecision.Name
	}
	reason := routerreplay.NewTrustedFactsReason(string(facts.Enforcement), string(facts.Stage), append([]string(nil), toolsCfg.TrustedFacts.TrustSources...), string(outcome))
	logging.Infof("[ToolsPlugin] Decision %q trusted-facts outcome=%s enforcement=%s stage=%s", decision, reason.Outcome, reason.Enforcement, reason.Stage)
	if ctx == nil || ctx.RouterReplayID == "" {
		return
	}
	recorder := ctx.RouterReplayRecorder
	if recorder == nil && r != nil {
		recorder = r.ReplayRecorder
	}
	if recorder == nil {
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
	if err := recorder.AppendOutcome(ctx.RouterReplayID, replayOutcome); err != nil {
		logging.Warnf("[ToolsPlugin] trusted-facts replay append failed: %v", err)
	}
}
