package classification

import (
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// A deployment whose model answers no span question and defines no pii or
// halu preset answers the PII and hallucination signals as yes/no verdicts:
// PII presence and categories, and a hallucination verdict. Their rules still
// match, but a detection carries no positions, so a configuration that masks
// PII is refused at load rather than failing its requests.

var spanVerdictFallbacks = promauto.NewCounterVec(prometheus.CounterOpts{
	Name: "vsr_signal_span_verdict_fallbacks_total",
	Help: "Configurations loaded whose span signal (pii_classifier, hallucination_detector) answers as a yes/no verdict because its deployment produces no spans.",
}, []string{"signal", "deployment"})

// reportSpanFallback makes a span signal's verdict fallback visible once per
// configuration that loads it: a warning and the metric.
func reportSpanFallback(signal string, judgment *decisionJudgment) {
	spanVerdictFallbacks.WithLabelValues(signal, judgment.deployment).Inc()
	logging.ComponentWarnEvent("classifier", "span_signal_verdict_fallback", map[string]interface{}{
		"signal":     signal,
		"deployment": judgment.deployment,
		"model":      judgment.card.ID,
		"effect":     "detections carry no positions; rules read the yes/no verdict (and PII categories)",
		"fix":        "bind the signal to a model that answers span questions to keep positions",
	})
}

// rejectMaskingWithoutSpans refuses a configuration that masks PII when the
// PII signal answers without positions.
func rejectMaskingWithoutSpans(cfg *config.RouterConfig, deployment string) error {
	if !cfg.UsesMaskingInReachableRouting() {
		return nil
	}
	return fmt.Errorf("%w: masking replaces PII at the positions pii_classifier detects, and its deployment %q answers PII as a yes/no verdict without positions; bind pii_classifier to a model that answers span questions, or remove masking", binding.ErrCapability, deployment)
}
