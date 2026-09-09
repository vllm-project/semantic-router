package extproc

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// progressGateConfig converts protection tuning into the selection-side gate
// config. Omitted fields keep the packaged defaults, and an omitted section
// leaves the gate disabled.
func progressGateConfig(tuning config.RouterLearningProtectionTuning) selection.ProgressGateConfig {
	cfg := selection.DefaultProgressGateConfig()
	gate := tuning.ProgressGate
	if gate == nil {
		return cfg
	}
	if gate.Enabled != nil {
		cfg.Enabled = *gate.Enabled
	}
	if gate.Mode != "" {
		cfg.Mode = gate.Mode
	}
	if gate.MinWindowOutcomes != nil {
		cfg.MinWindowOutcomes = *gate.MinWindowOutcomes
	}
	if gate.MinConsecutiveRegressions != nil {
		cfg.MinConsecutiveRegressions = *gate.MinConsecutiveRegressions
	}
	if gate.MinConsecutiveRecoveries != nil {
		cfg.MinConsecutiveRecoveries = *gate.MinConsecutiveRecoveries
	}
	if gate.CooldownSeconds != nil {
		cfg.CooldownSeconds = *gate.CooldownSeconds
	}
	if gate.MaxSwitchesPerWindow != nil {
		cfg.MaxSwitchesPerWindow = *gate.MaxSwitchesPerWindow
	}
	return cfg
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
			OutputTokens:      o.OutputTokens,
			LatencyMs:         o.LatencyMs,
		})
	}
	return facts
}

// switchGateVerdict evaluates the evidence gate for a switch the selector or a
// rescue path already proposed. It returns the verdict plus the trace section
// so callers record the same reasoning they act on. Switches that do not change
// the model, and sessions the gate cannot key, are left untouched.
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
	sessionKey := routingSessionStateKey(ctx)
	if sessionKey == "" {
		return selection.SwitchGateDecision{}, nil, false
	}

	now := time.Now()
	window := sessiontelemetry.RecentTurnOutcomes(sessionKey, now)
	evidence := selection.EvaluateProgressEvidence(turnOutcomeFacts(window))

	in := selection.SwitchGateInput{
		Evidence:  evidence,
		Downgrade: downgrade,
	}
	if snapshot, ok := sessiontelemetry.GetRouterSessionSnapshot(sessionKey, now); ok {
		in.SwitchesInWindow = snapshot.SwitchCount
		if secs, known := selection.SecondsSince(snapshot.LastSeen, now); known && snapshot.SwitchCount > 0 {
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
	return decision, switchGateTrace(decision, evidence, in, len(window)), true
}

// isDowngradeSwitch is intentionally absent: quality-score knowledge lives in
// the selection package, so callers pass the already-resolved downgrade flag.

func agenticSessionFromContext(learningCtx *selection.SelectionContext) *selection.AgenticSessionContext {
	if learningCtx == nil {
		return nil
	}
	return learningCtx.AgenticSession
}

func switchGateTrace(
	decision selection.SwitchGateDecision,
	evidence selection.ProgressEvidence,
	in selection.SwitchGateInput,
	windowCount int,
) *selection.SessionSwitchGateTrace {
	return &selection.SessionSwitchGateTrace{
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
