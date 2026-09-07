package selection

import "time"

// Switch gate decisions and suppression reasons recorded in replay.
const (
	GateDecisionSwitch   = "switch"
	GateDecisionSuppress = "suppress"

	GateReasonColdStart            = "cold_start"
	GateReasonInsufficientEvidence = "insufficient_evidence"
	GateReasonCooldown             = "cooldown"
	GateReasonOscillationGuard     = "oscillation_guard"
	GateReasonHardConstraint       = "hard_constraint_conflict"

	GateModeObserve = "observe"
	GateModeEnforce = "enforce"

	SwitchOriginEscalation = "escalation"
	SwitchOriginDowngrade  = "downgrade"
)

// ProgressGateConfig tunes the evidence thresholds. A zero value disables the
// gate, matching the default-off requirement in issue #3377.
type ProgressGateConfig struct {
	Enabled bool
	Mode    string

	MinWindowOutcomes         int
	MinConsecutiveRegressions int
	MinConsecutiveRecoveries  int
	CooldownSeconds           float64
	MaxSwitchesPerWindow      int
}

// DefaultProgressGateConfig returns the calibration placeholders. Defaults stay
// in observe mode until PL-0041 TASK-10 calibrates them.
func DefaultProgressGateConfig() ProgressGateConfig {
	return ProgressGateConfig{
		Enabled:                   false,
		Mode:                      GateModeObserve,
		MinWindowOutcomes:         3,
		MinConsecutiveRegressions: 2,
		MinConsecutiveRecoveries:  2,
		CooldownSeconds:           120,
		MaxSwitchesPerWindow:      2,
	}
}

// SwitchGateInput is everything the gate needs beyond the evidence itself.
type SwitchGateInput struct {
	Evidence ProgressEvidence

	// Downgrade marks a switch toward a cheaper or weaker model, which uses the
	// recovery threshold instead of the regression threshold.
	Downgrade bool

	// HardConstraintConflict short-circuits the gate: hard locks, candidate
	// eligibility, and budget checks stay authoritative.
	HardConstraintConflict bool
	HardConstraintReason   string

	SwitchesInWindow       int
	SecondsSinceLastSwitch float64
	LastSwitchKnown        bool
}

// SwitchGateDecision explains one gate evaluation. Enforced reports whether the
// caller must honour it; observe mode records the same reasoning without
// intercepting.
type SwitchGateDecision struct {
	Version  string
	Mode     string
	Decision string
	Reason   string
	Origin   string
	Enforced bool

	RegressionStreak int
	RecoveryStreak   int
	Trend            float64
}

// EvaluateSwitchGate decides whether trajectory evidence justifies a switch the
// selector already proposed. It can only suppress: hard constraints and budget
// checks remain authoritative and are re-checked by the caller before commit.
func EvaluateSwitchGate(cfg ProgressGateConfig, in SwitchGateInput) SwitchGateDecision {
	decision := SwitchGateDecision{
		Version:          in.Evidence.Version,
		Mode:             gateMode(cfg.Mode),
		Decision:         GateDecisionSwitch,
		Origin:           switchOrigin(in.Downgrade),
		RegressionStreak: in.Evidence.RegressionStreak,
		RecoveryStreak:   in.Evidence.RecoveryStreak,
		Trend:            in.Evidence.Trend,
	}
	if decision.Version == "" {
		decision.Version = ProgressEvidenceVersion
	}
	if !cfg.Enabled {
		return decision
	}
	decision.Enforced = decision.Mode == GateModeEnforce

	if reason := gateSuppressionReason(cfg, in); reason != "" {
		decision.Decision = GateDecisionSuppress
		decision.Reason = reason
		if reason == GateReasonHardConstraint && in.HardConstraintReason != "" {
			decision.Reason = in.HardConstraintReason
		}
	}
	return decision
}

// gateSuppressionReason returns the first rule that blocks the switch, or "" when
// the evidence clears every threshold.
func gateSuppressionReason(cfg ProgressGateConfig, in SwitchGateInput) string {
	if in.HardConstraintConflict {
		return GateReasonHardConstraint
	}
	if in.Evidence.ColdStart {
		return GateReasonColdStart
	}
	if in.Evidence.AttributableCount < cfg.MinWindowOutcomes {
		return GateReasonInsufficientEvidence
	}
	if !streakMeetsThreshold(cfg, in) {
		return GateReasonInsufficientEvidence
	}
	if cfg.CooldownSeconds > 0 && in.LastSwitchKnown && in.SecondsSinceLastSwitch < cfg.CooldownSeconds {
		return GateReasonCooldown
	}
	if cfg.MaxSwitchesPerWindow > 0 && in.SwitchesInWindow >= cfg.MaxSwitchesPerWindow {
		return GateReasonOscillationGuard
	}
	return ""
}

// streakMeetsThreshold applies the hysteresis: escalation needs consecutive
// attributable regressions and a non-positive trend, downgrade needs consecutive
// recoveries.
func streakMeetsThreshold(cfg ProgressGateConfig, in SwitchGateInput) bool {
	if in.Downgrade {
		return cfg.MinConsecutiveRecoveries <= 0 ||
			in.Evidence.RecoveryStreak >= cfg.MinConsecutiveRecoveries
	}
	if cfg.MinConsecutiveRegressions > 0 && in.Evidence.RegressionStreak < cfg.MinConsecutiveRegressions {
		return false
	}
	return in.Evidence.Trend <= 0
}

func gateMode(mode string) string {
	if mode == GateModeEnforce {
		return GateModeEnforce
	}
	return GateModeObserve
}

func switchOrigin(downgrade bool) string {
	if downgrade {
		return SwitchOriginDowngrade
	}
	return SwitchOriginEscalation
}

// Suppressed reports whether the caller must hold the current model.
func (d SwitchGateDecision) Suppressed() bool {
	return d.Enforced && d.Decision == GateDecisionSuppress
}

// SecondsSince is a helper for callers holding timestamps rather than deltas.
func SecondsSince(from, to time.Time) (float64, bool) {
	if from.IsZero() || to.IsZero() {
		return 0, false
	}
	delta := to.Sub(from).Seconds()
	if delta < 0 {
		delta = 0
	}
	return delta, true
}
