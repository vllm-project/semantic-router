package extproc

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// Reject a conflict between hard request ownership and candidate admission.
// A lock cannot authorize an excluded model, and selector fallback must not
// silently transfer the bound request to a different model. Checks run before
// selection and after selector filters, before learning records a new owner.
//
// One carve-out: an active tool loop is portable by construction (the client
// resends the full conversation each turn), so its lock binds the loop to the
// decision state that admitted it. Decision drift is itself a continuity
// boundary - the drifted decision releases the lock instead of rejecting
// every candidate and deadlocking the loop, and the transfer is traced as
// decision_drift rather than silent. Opaque provider state keeps the stricter
// rule: it stays physically bound to the model that created it.
func (r *OpenAIRouter) validateProtectedCandidateOwnership(selCtx *selection.SelectionContext, ctx *RequestContext) error {
	if r == nil || r.Config == nil || selCtx == nil || ctx == nil ||
		!r.Config.RouterLearning.Enabled || protectionMode(ctx) != config.DecisionAdaptationModeApply {
		return nil
	}
	cfg := r.Config.RouterLearning.Protection
	if !cfg.EffectiveEnabled() {
		return nil
	}
	identity, ok := r.protectionIdentity(ctx, cfg)
	if !ok {
		return nil
	}
	learningCtx := r.protectionSelectionContext(selCtx, ctx, identity)
	session := learningCtx.AgenticSession
	if session == nil || (!session.ActiveToolLoop && !session.HasNonPortableContext) {
		return nil
	}
	current := currentLearningModel(learningCtx)
	if current == "" {
		return nil
	}
	if session.ActiveToolLoop && !session.HasNonPortableContext && selectionDecisionDrift(learningCtx) {
		return nil
	}
	if owner := session.PreviousCandidate; owner != nil {
		if modelRefInEligibility(*owner, learningCtx.CandidateModels) {
			return nil
		}
	} else if selectionContextContainsModel(learningCtx, current) {
		// Older session snapshots carry only model identity. Do not invent an
		// effort for those records, but never downgrade a recorded exact owner.
		return nil
	}
	boundary := "nonportable context"
	if session.ActiveToolLoop {
		boundary = "active tool loop"
	}
	ownerDescription := fmt.Sprintf("model %q", current)
	if owner := session.PreviousCandidate; owner != nil {
		ownerDescription = fmt.Sprintf("candidate %q (effort=%q, lora=%q)", owner.Model, owner.ReasoningEffort, owner.LoRAName)
	}
	err := fmt.Errorf("%w: %s is bound to %s outside the admitted candidate set", selection.ErrNoEligibleCandidates, boundary, ownerDescription)
	ctx.VSRSelectionReasoning = err.Error()
	return err
}
