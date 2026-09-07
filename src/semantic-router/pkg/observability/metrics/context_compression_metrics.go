package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	contextCompressionTokensBefore = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_context_compression_tokens_before",
			Help:    "Estimated tool-result tokens before route-local compression.",
			Buckets: []float64{128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000},
		},
		[]string{"format"},
	)
	contextCompressionTokensAfter = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_context_compression_tokens_after",
			Help:    "Estimated tool-result tokens after route-local compression.",
			Buckets: []float64{64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000},
		},
		[]string{"format"},
	)
	contextCompressionRatio = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_context_compression_ratio",
			Help:    "Compressed-to-original estimated token ratio.",
			Buckets: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1},
		},
		[]string{"format"},
	)
	contextCompressionApplied = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_context_compression_applied_total",
			Help: "Context compression outcomes by decision, strategy, target, and status.",
		},
		[]string{"decision", "strategy", "target", "status"},
	)
	contextCompressionSavedTokens = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_context_compression_tokens_saved",
			Help:    "Estimated input tokens saved by context compression.",
			Buckets: []float64{0, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000},
		},
		[]string{"strategy", "token_counter_source"},
	)
	contextCompressionRecovery = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_context_compression_recovery_total",
			Help: "Recoverable compression events by status.",
		},
		[]string{"status"},
	)
	contextCompressionRecoveryAge = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "llm_context_compression_recovery_age_seconds",
			Help:    "Age of successfully retrieved compressed context.",
			Buckets: prometheus.ExponentialBuckets(1, 2, 12),
		},
	)
	contextCompressionRecoveryTTL = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "llm_context_compression_recovery_ttl_remaining_seconds",
			Help:    "Remaining TTL of successfully retrieved compressed context.",
			Buckets: prometheus.ExponentialBuckets(1, 2, 12),
		},
	)
	contextCompressionQuality = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_context_compression_quality_guard_total",
			Help: "Context compression quality-guard outcomes and bounded fallbacks.",
		},
		[]string{"quality", "fallback"},
	)
	contextCompressionCostSaved = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_context_compression_cost_saved_total",
			Help: "Estimated prompt cost saved by context compression.",
		},
		[]string{"model", "currency"},
	)
)

// RecordContextCompression records bounded, content-free compression
// diagnostics. Format is a low-cardinality value: text, json, or mixed.
func RecordContextCompression(before int, after int, format string) {
	if before <= 0 || after < 0 {
		return
	}
	if format == "" {
		format = "text"
	}
	contextCompressionTokensBefore.WithLabelValues(format).Observe(float64(before))
	contextCompressionTokensAfter.WithLabelValues(format).Observe(float64(after))
	contextCompressionRatio.WithLabelValues(format).Observe(float64(after) / float64(before))
}

func RecordContextCompressionRecovery(
	status string,
	ageSeconds float64,
	ttlRemainingSeconds float64,
) {
	if status == "" {
		status = "unknown"
	}
	contextCompressionRecovery.WithLabelValues(status).Inc()
	if status == "retrieved" {
		if ageSeconds >= 0 {
			contextCompressionRecoveryAge.Observe(ageSeconds)
		}
		if ttlRemainingSeconds >= 0 {
			contextCompressionRecoveryTTL.Observe(ttlRemainingSeconds)
		}
	}
}

func RecordContextCompressionCostSavings(
	model string,
	currency string,
	amount float64,
) {
	if amount <= 0 {
		return
	}
	if model == "" {
		model = "unknown"
	}
	if currency == "" {
		currency = "USD"
	}
	contextCompressionCostSaved.WithLabelValues(model, currency).Add(amount)
}

func RecordContextCompressionPlan(
	decision string,
	strategy string,
	target string,
	status string,
	tokenCounterSource string,
	before int,
	after int,
	recoveryKeys int,
	quality string,
	fallback string,
) {
	if decision == "" {
		decision = "unknown"
	}
	if strategy == "" {
		strategy = "extractive"
	}
	if target == "" {
		target = "mixed"
	}
	if status == "" {
		status = "unknown"
	}
	if tokenCounterSource == "" {
		tokenCounterSource = "heuristic"
	}
	if quality == "" {
		quality = "not_applied"
	}
	if fallback == "" {
		fallback = "none"
	}
	contextCompressionApplied.WithLabelValues(
		decision,
		strategy,
		target,
		status,
	).Inc()
	if before >= after && before > 0 {
		contextCompressionSavedTokens.WithLabelValues(
			strategy,
			tokenCounterSource,
		).Observe(float64(before - after))
	}
	if recoveryKeys > 0 {
		contextCompressionRecovery.WithLabelValues("stored").Add(float64(recoveryKeys))
	}
	contextCompressionQuality.WithLabelValues(quality, fallback).Inc()
}
