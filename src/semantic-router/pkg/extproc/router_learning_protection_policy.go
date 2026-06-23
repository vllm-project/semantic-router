package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

const (
	routerLearningIdentityStatusMissing     routerLearningIdentityStatus = "missing"
	routerLearningIdentityStatusNotRequired routerLearningIdentityStatus = "not_required"
	routerLearningIdentityStatusPresent     routerLearningIdentityStatus = "present"
)

type routerLearningIdentityStatus string

type routerLearningProtectionDiagnostics struct {
	trace           *selection.SessionPolicyTrace
	identity        routerLearningIdentityDiagnostics
	samplingPolicy  string
	baseModel       string
	proposalModel   string
	finalModel      string
	switchCost      float64
	switchMargin    float64
	stabilityWeight float64
	rescue          bool
}

type routerLearningIdentityDiagnostics struct {
	scope         string
	sessionHeader string
	convoHeader   string
	session       routerLearningIdentityPart
	conversation  routerLearningIdentityPart
	memoryKeyHash string
}

type routerLearningIdentityPart struct {
	source   string
	required bool
	status   routerLearningIdentityStatus
	hash     string
}

func newProtectionPolicy(
	ctx *RequestContext,
	cfg config.RouterLearningProtectionConfig,
	mode string,
	action routerLearningAction,
	reason string,
	scope string,
) routerLearningPolicy {
	policy := newRouterLearningPolicy(routerLearningMethodProtection)
	policy.Mode = mode
	policy.Action = action
	policy.Reason = reason
	policy.Scope = scope
	policy.Details.Protection = newRouterLearningProtectionDiagnostics(
		nil,
		newRouterLearningIdentityDiagnostics(
			scope,
			cfg.HeaderName("session"),
			cfg.HeaderName("conversation"),
			strings.TrimSpace(headerValueCI(ctx, cfg.HeaderName("session"))),
			strings.TrimSpace(headerValueCI(ctx, cfg.HeaderName("conversation"))),
			"",
		),
	)
	return policy
}

func protectionPolicyFromSelectionResult(
	result *selection.SelectionResult,
	identity routerLearningIdentity,
	mode string,
	baseModel string,
	proposalModel string,
	finalModel string,
	cfg config.RouterLearningProtectionConfig,
) routerLearningPolicy {
	trace := protectionTraceFromResult(result)
	policy := newRouterLearningPolicy(routerLearningMethodProtection)
	policy.Mode = mode
	policy.Scope = identity.scope
	policy.Action = protectionLearningAction(trace, identity.scope, proposalModel, finalModel)
	policy.Reason = protectionLearningReason(trace, identity.scope, proposalModel, finalModel)
	policy.Details.Protection = newRouterLearningProtectionDiagnostics(
		trace,
		newRouterLearningIdentityDiagnostics(
			identity.scope,
			identity.sessionHeader,
			identity.conversationHeader,
			identity.sessionID,
			identity.conversationID,
			identity.memoryKey,
		),
	)
	policy.Details.Protection.baseModel = baseModel
	policy.Details.Protection.proposalModel = proposalModel
	policy.Details.Protection.finalModel = finalModel
	policy.Details.Protection.switchMargin = learningSwitchMargin(cfg)
	policy.Details.Protection.stabilityWeight = learningProtectionStabilityWeight(cfg)
	policy.Details.Protection.switchCost = protectionSwitchCost(trace)
	return policy
}

func protectionRescuePolicyFromSelectionResult(
	result *selection.SelectionResult,
	identity routerLearningIdentity,
	mode string,
	baseModel string,
	proposalModel string,
	finalModel string,
	cfg config.RouterLearningProtectionConfig,
) routerLearningPolicy {
	policy := protectionPolicyFromSelectionResult(
		result,
		identity,
		mode,
		baseModel,
		proposalModel,
		finalModel,
		cfg,
	)
	policy.Action = routerLearningActionRescueSwitch
	policy.Reason = "rescue_evidence"
	if policy.Details.Protection != nil {
		policy.Details.Protection.rescue = true
	}
	return policy
}

func newRouterLearningProtectionDiagnostics(
	trace *selection.SessionPolicyTrace,
	identity routerLearningIdentityDiagnostics,
) *routerLearningProtectionDiagnostics {
	return &routerLearningProtectionDiagnostics{
		trace:    trace,
		identity: identity,
	}
}

func protectionTraceFromResult(result *selection.SelectionResult) *selection.SessionPolicyTrace {
	if result == nil {
		return nil
	}
	return result.SessionPolicy
}

func protectionLearningAction(
	trace *selection.SessionPolicyTrace,
	scope string,
	proposalModel string,
	finalModel string,
) routerLearningAction {
	if protectionEstablishesBaseline(trace, scope) {
		return routerLearningActionEstablishBaseline
	}
	if trace == nil {
		if strings.TrimSpace(proposalModel) != "" && strings.TrimSpace(finalModel) == strings.TrimSpace(proposalModel) {
			return routerLearningActionAllowSwitch
		}
		return routerLearningActionHoldCurrent
	}
	if trace.HardLocked || (strings.TrimSpace(proposalModel) != "" && strings.TrimSpace(finalModel) != strings.TrimSpace(proposalModel)) {
		return routerLearningActionHoldCurrent
	}
	if strings.TrimSpace(trace.CurrentModel) == "" || strings.TrimSpace(proposalModel) == "" || strings.TrimSpace(finalModel) == strings.TrimSpace(proposalModel) {
		return routerLearningActionAllowSwitch
	}
	return routerLearningActionHoldCurrent
}

func protectionLearningReason(
	trace *selection.SessionPolicyTrace,
	scope string,
	proposalModel string,
	finalModel string,
) string {
	if protectionEstablishesBaseline(trace, scope) {
		return protectionBaselineReason(trace, scope)
	}
	if trace == nil {
		return ""
	}
	if trace.HardLocked {
		return normalizeProtectionReason(firstNonEmpty(trace.HardLockReason, trace.DecisionReason, "hold_current"))
	}
	if strings.TrimSpace(proposalModel) != "" && strings.TrimSpace(finalModel) != strings.TrimSpace(proposalModel) {
		return normalizeProtectionReason(firstNonEmpty(trace.DecisionReason, "switch_guard_hold"))
	}
	return normalizeProtectionReason(firstNonEmpty(trace.DecisionReason, "switch_allowed"))
}

func normalizeProtectionReason(reason string) string {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		return ""
	}
	if strings.Contains(reason, "tool_loop") || strings.Contains(reason, "context_portability") {
		return "tool_or_protocol_state"
	}
	switch reason {
	case "stay_has_best_adjusted_score", "switch_guard_hold", "hold_current":
		return "switch_margin_not_met"
	case "rescue_underpowered_model":
		return "rescue_evidence"
	default:
		return reason
	}
}

func protectionEstablishesBaseline(trace *selection.SessionPolicyTrace, scope string) bool {
	if trace == nil {
		return false
	}
	if trace.IdleExpired {
		return true
	}
	if strings.TrimSpace(trace.CurrentModel) == "" &&
		strings.TrimSpace(trace.SelectedModel) != "" &&
		(scope == config.RouterLearningScopeConversation || scope == config.RouterLearningScopeSession) {
		return true
	}
	return false
}

func protectionBaselineReason(trace *selection.SessionPolicyTrace, scope string) string {
	if trace != nil && trace.IdleExpired {
		return "idle_reset"
	}
	if scope == config.RouterLearningScopeSession {
		return "new_session"
	}
	return "new_conversation"
}

func (d *routerLearningProtectionDiagnostics) toPolicyMap() map[string]interface{} {
	if d == nil {
		return nil
	}
	out := map[string]interface{}{}
	if d.trace != nil {
		for key, value := range d.trace.ToMap() {
			field := routerLearningPolicyField(key)
			switch field {
			case "session_id", "user_id",
				learningPolicyFieldAction,
				learningPolicyFieldReason,
				learningPolicyFieldLearning,
				learningPolicyFieldMethod,
				learningPolicyFieldMode,
				learningPolicyFieldScope:
				continue
			default:
				setLearningPolicyValue(out, field, value)
			}
		}
	}
	setLearningPolicyValue(out, learningPolicyFieldIdentity, d.identity.toPolicyMap())
	setLearningPolicyString(out, learningPolicyFieldBaseModel, d.baseModel)
	setLearningPolicyString(out, learningPolicyFieldProposalModel, d.proposalModel)
	setLearningPolicyString(out, learningPolicyFieldFinalModel, d.finalModel)
	if d.switchMargin > 0 {
		setLearningPolicyNumber(out, learningPolicyFieldSwitchMargin, d.switchMargin)
	}
	if d.stabilityWeight > 0 {
		setLearningPolicyNumber(out, learningPolicyFieldStabilityWeight, d.stabilityWeight)
	}
	if d.switchCost > 0 {
		setLearningPolicyNumber(out, learningPolicyFieldSwitchCost, d.switchCost)
	}
	if d.samplingPolicy != "" {
		setLearningPolicyString(out, learningPolicyFieldSampling, d.samplingPolicy)
	}
	if d.rescue {
		setLearningPolicyValue(out, learningPolicyFieldRescue, map[string]interface{}{"active": true})
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func newRouterLearningIdentityDiagnostics(
	scope string,
	sessionHeader string,
	conversationHeader string,
	sessionID string,
	conversationID string,
	memoryKey string,
) routerLearningIdentityDiagnostics {
	conversationRequired := scope == config.RouterLearningScopeConversation
	return routerLearningIdentityDiagnostics{
		scope:         scope,
		sessionHeader: sessionHeader,
		convoHeader:   conversationHeader,
		session:       newRouterLearningIdentityPart(sessionHeader, sessionID, true),
		conversation:  newRouterLearningIdentityPart(conversationHeader, conversationID, conversationRequired),
		memoryKeyHash: shortLearningIdentityHash(memoryKey),
	}
}

func newRouterLearningIdentityPart(
	headerName string,
	value string,
	required bool,
) routerLearningIdentityPart {
	part := routerLearningIdentityPart{
		source:   "header:" + headerName,
		required: required,
		status:   routerLearningIdentityStatusNotRequired,
	}
	if required {
		part.status = routerLearningIdentityStatusMissing
	}
	if strings.TrimSpace(value) != "" {
		part.status = routerLearningIdentityStatusPresent
		part.hash = shortLearningIdentityHash(value)
	}
	return part
}

func (d routerLearningIdentityDiagnostics) toPolicyMap() map[string]interface{} {
	identity := map[string]interface{}{
		"scope": d.scope,
		"headers": map[string]interface{}{
			"session":      d.sessionHeader,
			"conversation": d.convoHeader,
		},
		"session":      d.session.toMap(),
		"conversation": d.conversation.toMap(),
	}
	if d.memoryKeyHash != "" {
		identity["memory_key_hash"] = d.memoryKeyHash
	}
	return identity
}

func (p routerLearningIdentityPart) toMap() map[string]interface{} {
	out := map[string]interface{}{
		"source":   p.source,
		"required": p.required,
		"status":   string(p.status),
	}
	if p.hash != "" {
		out["hash"] = p.hash
	}
	return out
}

func (d *routerLearningProtectionDiagnostics) stringField(field routerLearningPolicyField) string {
	if d == nil {
		return ""
	}
	if d.trace != nil {
		if value := traceProtectionStringField(d.trace, field); value != "" {
			return value
		}
	}
	return d.componentStringField(field)
}

func traceProtectionStringField(trace *selection.SessionPolicyTrace, field routerLearningPolicyField) string {
	if trace == nil {
		return ""
	}
	switch field {
	case learningPolicyFieldPhase:
		return strings.TrimSpace(string(trace.Phase))
	case learningPolicyFieldCurrentModel:
		return strings.TrimSpace(trace.CurrentModel)
	case learningPolicyFieldBaseSelectedModel:
		return strings.TrimSpace(trace.BaseSelectedModel)
	case learningPolicyFieldSelectedModel:
		return strings.TrimSpace(trace.SelectedModel)
	case learningPolicyFieldHardLockReason:
		return strings.TrimSpace(trace.HardLockReason)
	case learningPolicyFieldDecisionReason:
		return strings.TrimSpace(trace.DecisionReason)
	default:
		return ""
	}
}

func (d *routerLearningProtectionDiagnostics) componentStringField(field routerLearningPolicyField) string {
	switch field {
	case learningPolicyFieldBaseModel:
		return strings.TrimSpace(d.baseModel)
	case learningPolicyFieldProposalModel:
		return strings.TrimSpace(d.proposalModel)
	case learningPolicyFieldFinalModel:
		return strings.TrimSpace(d.finalModel)
	case learningPolicyFieldSampling:
		return strings.TrimSpace(d.samplingPolicy)
	default:
		return ""
	}
}

func (d *routerLearningProtectionDiagnostics) boolField(field routerLearningPolicyField) bool {
	if d == nil {
		return false
	}
	if d.trace != nil && field == learningPolicyFieldHardLocked {
		return d.trace.HardLocked
	}
	return false
}

func shortLearningIdentityHash(value string) string {
	if strings.TrimSpace(value) == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])[:16]
}

func protectionSwitchCost(trace *selection.SessionPolicyTrace) float64 {
	if trace == nil {
		return 0
	}
	best := 0.0
	for _, candidate := range trace.CandidateTraces {
		cost := candidate.HandoffPenalty +
			candidate.PrefixCachePenalty +
			candidate.ToolLoopPenalty +
			candidate.SwitchHistoryPenalty
		if cost > best {
			best = cost
		}
	}
	return best
}
