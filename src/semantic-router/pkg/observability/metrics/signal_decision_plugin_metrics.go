package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
)

var (
	// SignalPanicTotal tracks panics recovered inside signal evaluator goroutines.
	// A non-zero value indicates a classifier crashed mid-evaluation; the router
	// continued serving but that signal was skipped for the affected request.
	SignalPanicTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_signal_panic_total",
			Help: "Total number of panics recovered in signal evaluator goroutines",
		},
		[]string{"signal_type", "signal_name"},
	)

	// SignalExtractionTotal tracks the total number of signal extractions by type and name
	SignalExtractionTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_signal_extraction_total",
			Help: "Total number of signal extractions by type and name",
		},
		[]string{"signal_type", "signal_name"},
	)

	// SignalExtractionLatency tracks the latency of signal extraction by type
	SignalExtractionLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_signal_extraction_latency_seconds",
			Help:    "Latency of signal extraction by type in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"signal_type"},
	)

	// SignalMatchTotal tracks the total number of signal matches by type and name
	SignalMatchTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_signal_match_total",
			Help: "Total number of signal matches by type and name",
		},
		[]string{"signal_type", "signal_name"},
	)

	// DecisionEvaluationTotal tracks the total number of decision evaluations
	DecisionEvaluationTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "llm_decision_evaluation_total",
			Help: "Total number of decision evaluations",
		},
	)

	// DecisionEvaluationLatency tracks the latency of decision evaluation
	DecisionEvaluationLatency = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "llm_decision_evaluation_latency_seconds",
			Help:    "Latency of decision evaluation in seconds",
			Buckets: prometheus.DefBuckets,
		},
	)

	// DecisionMatchTotal tracks the total number of decision matches by decision name
	DecisionMatchTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_decision_match_total",
			Help: "Total number of decision matches by decision name",
		},
		[]string{"decision_name"},
	)

	// DecisionConfidence tracks the distribution of decision confidence scores
	DecisionConfidence = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_decision_confidence",
			Help:    "Distribution of decision confidence scores by decision name",
			Buckets: []float64{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
		},
		[]string{"decision_name"},
	)

	// PluginExecutionTotal tracks the total number of plugin executions by type, decision, and status
	PluginExecutionTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_plugin_execution_total",
			Help: "Total number of plugin executions by type, decision, and status",
		},
		[]string{"plugin_type", "decision_name", "status"},
	)

	// PluginExecutionLatency tracks the latency of plugin execution by type
	PluginExecutionLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_plugin_execution_latency_seconds",
			Help:    "Latency of plugin execution by type in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"plugin_type"},
	)

	// PluginExecutionErrors tracks the total number of plugin execution errors by type and reason
	PluginExecutionErrors = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_plugin_execution_errors_total",
			Help: "Total number of plugin execution errors by type and reason",
		},
		[]string{"plugin_type", "error_reason"},
	)
)

// RecordSignalPanic records a recovered panic from a signal evaluator goroutine.
func RecordSignalPanic(signalType, signalName string) {
	if signalType == "" {
		signalType = consts.UnknownLabel
	}
	if signalName == "" {
		signalName = consts.UnknownLabel
	}
	SignalPanicTotal.WithLabelValues(signalType, signalName).Inc()
}

// RecordSignalExtraction records a signal extraction event
func RecordSignalExtraction(signalType, signalName string, latencySeconds float64) {
	if signalType == "" {
		signalType = consts.UnknownLabel
	}
	if signalName == "" {
		signalName = consts.UnknownLabel
	}
	SignalExtractionTotal.WithLabelValues(signalType, signalName).Inc()
	SignalExtractionLatency.WithLabelValues(signalType).Observe(latencySeconds)
}

// RecordSignalMatch records a signal match event
func RecordSignalMatch(signalType, signalName string) {
	if signalType == "" {
		signalType = consts.UnknownLabel
	}
	if signalName == "" {
		signalName = consts.UnknownLabel
	}
	SignalMatchTotal.WithLabelValues(signalType, signalName).Inc()
}

// RecordDecisionEvaluation records a decision evaluation event
func RecordDecisionEvaluation(latencySeconds float64) {
	DecisionEvaluationTotal.Inc()
	DecisionEvaluationLatency.Observe(latencySeconds)
}

// RecordDecisionMatch records a decision match event with confidence
func RecordDecisionMatch(decisionName string, confidence float64) {
	if decisionName == "" {
		decisionName = consts.UnknownLabel
	}
	DecisionMatchTotal.WithLabelValues(decisionName).Inc()
	DecisionConfidence.WithLabelValues(decisionName).Observe(confidence)
}

// RecordPluginExecution records a plugin execution event
func RecordPluginExecution(pluginType, decisionName, status string, latencySeconds float64) {
	if pluginType == "" {
		pluginType = consts.UnknownLabel
	}
	if decisionName == "" {
		decisionName = consts.UnknownLabel
	}
	if status == "" {
		status = "unknown"
	}
	PluginExecutionTotal.WithLabelValues(pluginType, decisionName, status).Inc()
	PluginExecutionLatency.WithLabelValues(pluginType).Observe(latencySeconds)
}

// RecordPluginError records a plugin execution error
func RecordPluginError(pluginType, errorReason string) {
	if pluginType == "" {
		pluginType = consts.UnknownLabel
	}
	if errorReason == "" {
		errorReason = "unknown"
	}
	PluginExecutionErrors.WithLabelValues(pluginType, errorReason).Inc()
}
