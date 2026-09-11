package extproc

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func progressGateConfig(tuning config.RouterLearningProtectionTuning) selection.ProgressGateConfig {
	return tuning.ProgressGate.EffectiveConfig()
}

// turnOutcomeFacts converts stored outcomes into the selection-side facts the
// evidence evaluator consumes.
func turnOutcomeFacts(window []sessiontelemetry.TurnOutcome) []selection.TurnOutcomeFact {
	if len(window) == 0 {
		return nil
	}
	facts := make([]selection.TurnOutcomeFact, 0, len(window))
	for _, o := range window {
		facts = append(facts, selection.TurnOutcomeFact{
			TurnIndex:         o.TurnIndex,
			Timestamp:         o.Time(),
			Model:             o.Model,
			Category:          string(o.Category),
			ModelAttributable: o.ModelAttributable,
			Confidence:        o.Confidence,
			ConfidenceKnown:   o.ConfidenceKnown,
			OutputTokens:      o.OutputTokens,
			LatencyMs:         o.LatencyMs,
			LatencyKnown:      o.LatencyKnown,
			Cost:              o.Cost,
			CostKnown:         o.CostKnown,
		})
	}
	return facts
}

func progressEvidenceStateKey(ctx *RequestContext) string {
	if ctx != nil && !ctx.Routing.IsPassthrough() && ctx.ResponseObjectState != nil {
		return config.RoutingNamespaceKey(ctx.Routing.RecipeName(), ctx.ResponseObjectState.SessionTrackingID)
	}
	return routingSessionStateKey(ctx)
}

func configureProgressEvidence(ctx *RequestContext, cfg config.ProgressGateConfig, now time.Time) {
	if ctx == nil || !cfg.Enabled {
		return
	}
	ctx.VSRProgressGateConfig = &cfg
	ttl := time.Duration(cfg.WindowTTLSeconds) * time.Second
	key, stateKey := progressEvidenceStateKey(ctx), routingLearningStateKey(ctx)
	sessiontelemetry.ConfigureTurnOutcomeWindow(key, cfg.WindowSize, ttl, now)
	if stateKey != key {
		sessiontelemetry.ConfigureTurnOutcomeWindow(stateKey, cfg.WindowSize, ttl, now)
	}
}

// switchGateVerdict evaluates a proposal; its caller commits the final result.
func (r *OpenAIRouter) switchGateVerdict(
	cfg config.RouterLearningProtectionConfig,
	ctx *RequestContext,
	learningCtx *selection.SelectionContext,
	currentModel string,
	proposedModel string,
	downgrade bool,
) (selection.SwitchGateDecision, *selection.SessionSwitchGateTrace, bool) {
	gateCfg := progressGateConfig(cfg.Tuning)
	if !gateCfg.Enabled || currentModel == "" || proposedModel == "" || currentModel == proposedModel {
		return selection.SwitchGateDecision{}, nil, false
	}
	sessionKey := progressEvidenceStateKey(ctx)
	if sessionKey == "" {
		return selection.SwitchGateDecision{}, nil, false
	}
	stateKey := routingLearningStateKey(ctx)
	now := time.Now()
	windowTTL := time.Duration(gateCfg.WindowTTLSeconds) * time.Second
	configureProgressEvidence(ctx, gateCfg, now)
	window := sessiontelemetry.RecentTurnOutcomesWithPolicy(sessionKey, now, gateCfg.WindowSize, windowTTL)
	evidence := selection.EvaluateProgressEvidence(turnOutcomeFacts(window))

	in := selection.SwitchGateInput{
		Evidence:  evidence,
		Downgrade: downgrade,
	}
	if snapshot, ok := sessiontelemetry.GetRouterSessionSnapshot(stateKey, now); ok {
		// The oscillation guard is window-scoped: count the model changes
		// inside the gate's own evidence window, not the session lifetime.
		in.SwitchesInWindow = sessiontelemetry.CountRecentSwitches(snapshot.SwitchTimestamps, windowTTL, now)
		// LastSwitchAt is zero until the first real model change, so
		// SecondsSince reports unknown and cooldown simply does not apply.
		if secs, known := selection.SecondsSince(snapshot.LastSwitchAt, now); known {
			in.SecondsSinceLastSwitch = secs
			in.LastSwitchKnown = true
		}
	}
	// Hard locks are authoritative: the gate may only suppress, so a locked
	// session short-circuits with the lock's own reason.
	if session := agenticSessionFromContext(learningCtx); session != nil {
		if session.ActiveToolLoop || session.HasNonPortableContext {
			in.HardConstraintConflict = true
			in.HardConstraintReason = selection.GateReasonHardConstraint
		}
	}

	decision := selection.EvaluateSwitchGate(gateCfg, in)
	trace := switchGateTrace(decision, in, len(window))
	trace.CurrentModel, trace.ProposedModel, trace.FinalModel = currentModel, proposedModel, proposedModel
	trace.Source = "selector"
	return decision, trace, true
}

func agenticSessionFromContext(learningCtx *selection.SelectionContext) *selection.AgenticSessionContext {
	if learningCtx == nil {
		return nil
	}
	return learningCtx.AgenticSession
}

func switchGateTrace(
	decision selection.SwitchGateDecision,
	in selection.SwitchGateInput,
	windowCount int,
) *selection.SessionSwitchGateTrace {
	evidence := in.Evidence
	return &selection.SessionSwitchGateTrace{
		CalibrationID:   decision.CalibrationID,
		ConfidenceTrend: evidence.ConfidenceTrend, ConfidenceTrendKnown: evidence.ConfidenceTrendKnown,
		CostTrend: evidence.CostTrend, CostTrendKnown: evidence.CostTrendKnown,
		LatencyTrend: evidence.LatencyTrend, LatencyTrendKnown: evidence.LatencyTrendKnown,
		EvidenceVersion:        decision.Version,
		Mode:                   decision.Mode,
		Decision:               decision.Decision,
		Reason:                 decision.Reason,
		Origin:                 decision.Origin,
		Enforced:               decision.Enforced,
		RegressionStreak:       evidence.RegressionStreak,
		RecoveryStreak:         evidence.RecoveryStreak,
		Trend:                  evidence.Trend,
		AttributableCount:      evidence.AttributableCount,
		MissingCount:           evidence.MissingCount,
		WindowCount:            windowCount,
		ColdStart:              evidence.ColdStart,
		SwitchesInWindow:       in.SwitchesInWindow,
		SecondsSinceLastSwitch: in.SecondsSinceLastSwitch,
		LastSwitchKnown:        in.LastSwitchKnown,
	}
}

// attachSwitchGateTrace records the gate verdict on the result's session policy
// so replay explains every switch and every suppression.
func attachSwitchGateTrace(result *selection.SelectionResult, trace *selection.SessionSwitchGateTrace) {
	if result == nil || trace == nil {
		return
	}
	if result.SessionPolicy == nil {
		result.SessionPolicy = &selection.SessionPolicyTrace{}
	}
	result.SessionPolicy.SwitchGate = trace
}

// logSwitchGateSuppression records an enforced suppression for operators.
func logSwitchGateSuppression(ctx *RequestContext, currentModel, proposedModel string, decision selection.SwitchGateDecision) {
	requestID := ""
	if ctx != nil {
		requestID = ctx.RequestID
	}
	logging.ComponentDebugEvent("extproc", "progress_gate_suppressed_switch", map[string]interface{}{
		"request_id":         requestID,
		"current_model":      currentModel,
		"proposed_model":     proposedModel,
		"suppression_reason": decision.Reason,
		"switch_origin":      decision.Origin,
		"evidence_version":   decision.Version,
	})
}
