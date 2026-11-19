package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

const (
	protectionReleaseReasonHonored            = "explicit_release"
	protectionReleaseReasonReleaseDisabled    = "release_disabled"
	protectionReleaseReasonNotBoundary        = "release_not_at_turn_boundary"
	protectionReleaseReasonInterval           = "release_interval"
	protectionReleaseReasonNotApplied         = "release_not_applied"
	protectionReleaseReasonObserveOnly        = "release_observe_only"
	protectionReleaseReasonKeptCurrent        = "release_kept_current"
	protectionReleaseReasonProtectionBypassed = "protection_bypassed"
	protectionReleaseReasonProtectionDisabled = "protection_disabled"

	switchGateReasonRelease = "bypassed_by_release"
)

type protectionReleaseVerdict struct {
	requested     bool
	honored       bool
	reason        string
	header        string
	previousModel string
	selectedModel string
}

// requestedProtectionRelease returns the unresolved verdict for a request that
// carries the release header.
func requestedProtectionRelease(ctx *RequestContext, cfg config.RouterLearningProtectionConfig) (protectionReleaseVerdict, bool) {
	header := cfg.Release.EffectiveHeader()
	// Looper hops re-enter with the client's headers; one release must not
	// fire once per hop.
	if ctx == nil || ctx.Hop != nil || strings.TrimSpace(headerValueCI(ctx, header)) == "" {
		return protectionReleaseVerdict{}, false
	}
	return protectionReleaseVerdict{requested: true, header: header}, true
}

// unavailableProtectionRelease refuses a release on a request that protection
// does not evaluate, so the client still learns why it was ignored.
func unavailableProtectionRelease(ctx *RequestContext, cfg config.RouterLearningProtectionConfig, reason string) protectionReleaseVerdict {
	verdict, ok := requestedProtectionRelease(ctx, cfg)
	if !ok {
		return protectionReleaseVerdict{}
	}
	verdict.reason = reason
	if !cfg.Release.EffectiveEnabled() {
		verdict.reason = protectionReleaseReasonReleaseDisabled
	}
	return verdict
}

// evaluateProtectionRelease decides whether this request's release header can
// release the hold on its protection key. It reads session memory but never
// writes it: the interval is consumed only when the decision commits.
func evaluateProtectionRelease(
	ctx *RequestContext,
	cfg config.RouterLearningProtectionConfig,
	session *selection.AgenticSessionContext,
	mode string,
) protectionReleaseVerdict {
	verdict, ok := requestedProtectionRelease(ctx, cfg)
	if !ok {
		return verdict
	}
	if !cfg.Release.EffectiveEnabled() {
		verdict.reason = protectionReleaseReasonReleaseDisabled
		return verdict
	}
	if session == nil || strings.TrimSpace(session.PreviousModel) == "" {
		// A new key has no hold to release; it gets a normal first decision.
		return protectionReleaseVerdict{}
	}
	verdict.previousModel = session.PreviousModel
	if session.ActiveToolLoop || session.HasNonPortableContext {
		verdict.reason = protectionReleaseReasonNotBoundary
		return verdict
	}
	// Memory has not counted this request yet; at a turn boundary it is the
	// next user turn.
	last := session.MemoryLastReleaseUserTurn
	if last > 0 && session.MemoryUserTurnCount+1-last <= cfg.Release.EffectiveMinUserTurnsBetween() {
		verdict.reason = protectionReleaseReasonInterval
		return verdict
	}
	if mode != config.DecisionAdaptationModeApply {
		verdict.reason = protectionReleaseReasonObserveOnly
		return verdict
	}
	verdict.honored = true
	verdict.reason = protectionReleaseReasonHonored
	return verdict
}

// protectionReleaseHeaderForRemoval names a renamed release header so it
// never reaches a backend, whether or not protection or release is enabled.
// The default name is already one of the headers every provider-bound
// request drops (headers.KeptFromProviders).
func (r *OpenAIRouter) protectionReleaseHeaderForRemoval() []string {
	if r == nil || r.Config == nil {
		return nil
	}
	header := r.Config.RouterLearning.Protection.Release.EffectiveHeader()
	if strings.EqualFold(header, headers.VSRProtectionRelease) {
		return nil
	}
	return []string{header}
}

func recordProtectionReleaseWarning(ctx *RequestContext, verdict protectionReleaseVerdict) {
	if ctx == nil || !verdict.requested || verdict.honored {
		return
	}
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, llmprotocol.Diagnostic{
		Source: ctx.SourceFormat,
		Target: ctx.TargetFormat,
		Field:  verdict.header,
		Action: llmprotocol.DiagnosticDropped,
		Reason: verdict.reason,
	})
}

func (v protectionReleaseVerdict) toPolicyMap() map[string]interface{} {
	out := map[string]interface{}{
		"requested": v.requested,
		"honored":   v.honored,
		"reason":    v.reason,
		"header":    v.header,
	}
	if v.previousModel != "" {
		out["previous_model"] = v.previousModel
	}
	if v.selectedModel != "" {
		out["selected_model"] = v.selectedModel
	}
	return out
}

// applyProtectionReleasePolicy records the verdict on the decision's policy and
// marks an applied release as such. A release counts as applied only when the
// decision carries the released learning context; fallbacks and failed
// selections keep their own action. A release that re-selects the held model
// still consumes the interval but is not a switch.
func applyProtectionReleasePolicy(ctx *RequestContext, decision *routerLearningDecision, verdict protectionReleaseVerdict) {
	if decision == nil || !verdict.requested || decision.policy.Empty() {
		return
	}
	if verdict.honored {
		session := agenticSessionFromContext(decision.selectionContext)
		if session != nil && session.ReleaseHonored {
			verdict.selectedModel = selectedModelName(decision.selectionResult)
			decision.policy.Action = routerLearningActionReleaseSwitch
			decision.policy.Reason = protectionReleaseReasonHonored
			if verdict.selectedModel == verdict.previousModel {
				decision.policy.Action = routerLearningActionHoldCurrent
				decision.policy.Reason = protectionReleaseReasonKeptCurrent
			}
		} else {
			verdict.honored = false
			verdict.reason = protectionReleaseReasonNotApplied
			recordProtectionReleaseWarning(ctx, verdict)
		}
	}
	if decision.policy.Details.Protection == nil {
		decision.policy.Details.Protection = newRouterLearningProtectionDiagnostics(nil, routerLearningIdentityDiagnostics{})
	}
	decision.policy.Details.Protection.release = &verdict
}
