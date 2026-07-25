package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
)

// Fusion metrics give operators per-request visibility into the multi-model
// deliberation path (panel -> grounding -> analysis -> synthesis): how often it
// runs, how long each stage takes, which panel models fail, what the grounding
// scorer produces, and the token cost paid. Without these, fusion traffic cannot
// be SLO'd, budgeted, or alarmed on (see bench/grounded_fusion/FINDINGS.md for
// why cost/quality visibility matters here).
var (
	// FusionRequests counts fusion executions by decision and final status.
	FusionRequests = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "vsr_fusion_requests_total",
			Help: "The total number of Fusion looper executions by decision and status (success|error)",
		},
		[]string{"decision", "status"},
	)

	// FusionRequestDuration tracks end-to-end fusion execution latency. The
	// critical path is N parallel panel calls + analysis + synthesis, so this
	// runs in seconds-to-minutes.
	FusionRequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "vsr_fusion_request_duration_seconds",
			Help:    "End-to-end Fusion execution latency in seconds by decision",
			Buckets: []float64{0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300},
		},
		[]string{"decision"},
	)

	// FusionStageDuration tracks per-stage latency so operators can see where the
	// time goes (panel|grounding|analysis|synthesis).
	FusionStageDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "vsr_fusion_stage_duration_seconds",
			Help:    "Per-stage Fusion latency in seconds (stage=panel|grounding|analysis|synthesis)",
			Buckets: []float64{0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120},
		},
		[]string{"stage"},
	)

	// FusionPanelModels counts per-model panel outcomes, driving per-model
	// failure rate (status=success|failed).
	FusionPanelModels = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "vsr_fusion_panel_models_total",
			Help: "The total number of Fusion panel model calls by model and status (success|failed)",
		},
		[]string{"model", "status"},
	)

	// FusionGroundingScore is the distribution of per-response groundedness
	// scores [0,1] by reference mode and policy.
	FusionGroundingScore = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "vsr_fusion_grounding_score",
			Help:    "Distribution of Fusion grounding scores [0,1] by reference_mode (panel|context) and policy (weight|annotate|filter)",
			Buckets: prometheus.LinearBuckets(0, 0.1, 11),
		},
		[]string{"reference_mode", "policy"},
	)

	// FusionGroundingDropped counts panel responses dropped by the grounding
	// stage (only the `filter` policy drops; weight/annotate keep all).
	FusionGroundingDropped = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "vsr_fusion_grounding_dropped_total",
			Help: "The total number of panel responses dropped by the Fusion grounding stage by policy",
		},
		[]string{"policy"},
	)

	// FusionEarlyExit counts fusion executions that took the panel-agreement
	// early-exit path (skipped the analysis judge call because the panel was
	// unanimous).
	FusionEarlyExit = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "vsr_fusion_early_exit_total",
			Help: "The total number of Fusion executions that took the panel-agreement early-exit (skipped analysis) by decision",
		},
		[]string{"decision"},
	)

	// FusionEscalationBypass counts fusion executions that skipped the panel via
	// adaptive escalation (an easy query answered by a single judge-model call).
	FusionEscalationBypass = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "vsr_fusion_escalation_bypass_total",
			Help: "The total number of Fusion executions that skipped the panel via adaptive escalation (single-model fast path) by decision",
		},
		[]string{"decision"},
	)

	// FusionRequestTokens counts tokens consumed across all model calls in a
	// fusion request by decision and type (prompt|completion). This is the real
	// cost paid: ~N+2 LLM calls per request.
	FusionRequestTokens = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "vsr_fusion_request_tokens_total",
			Help: "The total tokens consumed by Fusion executions by decision and type (prompt|completion)",
		},
		[]string{"decision", "type"},
	)
)

func fusionDecisionLabel(decision string) string {
	if decision == "" {
		return consts.UnknownLabel
	}
	return decision
}

// RecordFusionRequest records a completed fusion execution and its status.
func RecordFusionRequest(decision, status string) {
	if status == "" {
		status = consts.UnknownLabel
	}
	FusionRequests.WithLabelValues(fusionDecisionLabel(decision), status).Inc()
}

// RecordFusionRequestDuration records the end-to-end fusion latency in seconds.
func RecordFusionRequestDuration(decision string, seconds float64) {
	FusionRequestDuration.WithLabelValues(fusionDecisionLabel(decision)).Observe(seconds)
}

// RecordFusionStageDuration records the latency of a single fusion stage.
func RecordFusionStageDuration(stage string, seconds float64) {
	if stage == "" {
		stage = consts.UnknownLabel
	}
	FusionStageDuration.WithLabelValues(stage).Observe(seconds)
}

// RecordFusionPanelModel records the outcome of a single panel model call.
func RecordFusionPanelModel(model, status string) {
	if model == "" {
		model = consts.UnknownLabel
	}
	if status == "" {
		status = consts.UnknownLabel
	}
	FusionPanelModels.WithLabelValues(model, status).Inc()
}

// RecordFusionGroundingScore records one per-response groundedness score.
func RecordFusionGroundingScore(referenceMode, policy string, score float64) {
	if referenceMode == "" {
		referenceMode = consts.UnknownLabel
	}
	if policy == "" {
		policy = consts.UnknownLabel
	}
	FusionGroundingScore.WithLabelValues(referenceMode, policy).Observe(score)
}

// RecordFusionGroundingDropped records the number of responses dropped by the
// grounding stage for a policy. A no-op when count <= 0 so the series only
// appears once grounding has actually dropped something.
func RecordFusionGroundingDropped(policy string, count int) {
	if count <= 0 {
		return
	}
	if policy == "" {
		policy = consts.UnknownLabel
	}
	FusionGroundingDropped.WithLabelValues(policy).Add(float64(count))
}

// RecordFusionEarlyExit records a fusion execution that skipped analysis via the
// panel-agreement early-exit path.
func RecordFusionEarlyExit(decision string) {
	FusionEarlyExit.WithLabelValues(fusionDecisionLabel(decision)).Inc()
}

// RecordFusionEscalationBypass records a fusion execution that skipped the panel
// via adaptive escalation (single-model fast path).
func RecordFusionEscalationBypass(decision string) {
	FusionEscalationBypass.WithLabelValues(fusionDecisionLabel(decision)).Inc()
}

// RecordFusionRequestTokens records the prompt/completion tokens for a fusion
// execution. Negative inputs are clamped to zero.
func RecordFusionRequestTokens(decision string, promptTokens, completionTokens int64) {
	label := fusionDecisionLabel(decision)
	if promptTokens > 0 {
		FusionRequestTokens.WithLabelValues(label, "prompt").Add(float64(promptTokens))
	}
	if completionTokens > 0 {
		FusionRequestTokens.WithLabelValues(label, "completion").Add(float64(completionTokens))
	}
}
