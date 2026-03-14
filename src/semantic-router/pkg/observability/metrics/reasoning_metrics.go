package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
)

var (
	// ReasoningDecisions tracks the reasoning mode decision outcome by category, model, and effort
	ReasoningDecisions = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_reasoning_decisions_total",
			Help: "The total number of reasoning mode decisions by category, model, and effort",
		},
		[]string{"category", "model", "enabled", "effort"},
	)

	// ReasoningTemplateUsage tracks usage of model-family-specific template parameters
	ReasoningTemplateUsage = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_reasoning_template_usage_total",
			Help: "The total number of times a model family template parameter was applied",
		},
		[]string{"family", "param"},
	)

	// ReasoningEffortUsage tracks the distribution of reasoning efforts by model family
	ReasoningEffortUsage = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_reasoning_effort_usage_total",
			Help: "The total number of times a reasoning effort level was set per model family",
		},
		[]string{"family", "effort"},
	)
)

// RecordReasoningDecision records a reasoning-mode decision for a category, model and effort
func RecordReasoningDecision(category, model string, enabled bool, effort string) {
	status := "false"
	if enabled {
		status = "true"
	}
	ReasoningDecisions.WithLabelValues(category, model, status, effort).Inc()
}

// RecordReasoningTemplateUsage records usage of a model-family-specific template parameter
func RecordReasoningTemplateUsage(family, param string) {
	if family == "" {
		family = consts.UnknownLabel
	}
	if param == "" {
		param = "none"
	}
	ReasoningTemplateUsage.WithLabelValues(family, param).Inc()
}

// RecordReasoningEffortUsage records the effort usage by model family
func RecordReasoningEffortUsage(family, effort string) {
	if family == "" {
		family = consts.UnknownLabel
	}
	if effort == "" {
		effort = "unspecified"
	}
	ReasoningEffortUsage.WithLabelValues(family, effort).Inc()
}
