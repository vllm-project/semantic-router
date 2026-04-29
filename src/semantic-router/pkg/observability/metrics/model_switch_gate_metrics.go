package metrics

import (
	"strconv"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
)

// ModelSwitchGateDecisions counts every gate evaluation, labeled so operators
// can diff shadow vs. enforce, would-switch vs. stay, and bucket the reasons
// emitted by the gate (e.g., switch_advantage_exceeds_threshold,
// stay_cost_not_exceeded, missing_signal_fallback, candidate_is_current_model).
var ModelSwitchGateDecisions = promauto.NewCounterVec(
	prometheus.CounterOpts{
		Name: "llm_model_switch_gate_decisions_total",
		Help: "Total model_switch_gate evaluations, labeled by mode, would_switch, enforced_stay, and reason.",
	},
	[]string{"mode", "would_switch", "enforced_stay", "reason"},
)

// ModelSwitchGateNetAdvantage tracks the distribution of net switch advantage
// (switch benefit minus handoff and cache penalties) so operators can decide
// whether to graduate from shadow to enforce based on real traffic.
var ModelSwitchGateNetAdvantage = promauto.NewHistogram(
	prometheus.HistogramOpts{
		Name:    "llm_model_switch_gate_net_advantage",
		Help:    "Distribution of net switch advantage produced by the model_switch_gate when evidence was collected.",
		Buckets: []float64{-0.5, -0.25, -0.1, -0.05, 0, 0.05, 0.1, 0.25, 0.5},
	},
)

// ModelSwitchGateMissingSignals counts which evidence signals were unavailable
// at gate time, labeled by signal name (previous_model, session_id,
// handoff_penalty, cache_warmth, etc.). Use this to spot rollout gaps before
// flipping the gate to enforce.
var ModelSwitchGateMissingSignals = promauto.NewCounterVec(
	prometheus.CounterOpts{
		Name: "llm_model_switch_gate_missing_signals_total",
		Help: "Total occurrences of each missing signal observed by the model_switch_gate.",
	},
	[]string{"signal"},
)

// RecordModelSwitchGateDecision increments the decision counter for one gate
// evaluation. Empty mode falls back to consts.UnknownLabel.
func RecordModelSwitchGateDecision(mode string, wouldSwitch, enforcedStay bool, reason string) {
	if mode == "" {
		mode = consts.UnknownLabel
	}
	if reason == "" {
		reason = consts.UnknownLabel
	}
	ModelSwitchGateDecisions.WithLabelValues(
		mode,
		strconv.FormatBool(wouldSwitch),
		strconv.FormatBool(enforcedStay),
		reason,
	).Inc()
}

// ObserveModelSwitchGateNetAdvantage records one net-advantage observation. Caller
// should only invoke this when evidence was collected (otherwise the value is
// meaningless and would skew the histogram).
func ObserveModelSwitchGateNetAdvantage(advantage float64) {
	ModelSwitchGateNetAdvantage.Observe(advantage)
}

// RecordModelSwitchGateMissingSignal increments the missing-signal counter for
// one signal name.
func RecordModelSwitchGateMissingSignal(signal string) {
	if signal == "" {
		signal = consts.UnknownLabel
	}
	ModelSwitchGateMissingSignals.WithLabelValues(signal).Inc()
}
