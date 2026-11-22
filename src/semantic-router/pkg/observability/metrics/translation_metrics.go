package metrics

import (
	"regexp"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// TranslationLossyTotal counts translation warnings observed at the
// response-header and streaming-body phases, partitioned by inbound/outbound protocol pair,
// severity, and reason. Used by post-deployment dashboards to detect
// protocol-pair regressions and to size lossiness across the fleet.
var TranslationLossyTotal = promauto.NewCounterVec(
	prometheus.CounterOpts{
		Name: "llm_translation_lossy_total",
		Help: "Total translation warnings emitted by the inbound parser, partitioned by protocol pair, severity, and reason.",
	},
	[]string{"from", "to", "severity", "reason"},
)

// reasonCount matches the numbers a diagnostic reason carries, such as
// "3 system messages sent as user text in place".
var reasonCount = regexp.MustCompile(`[0-9]+`)

// RecordTranslationWarning increments the translation-warning counter
// for one warning observed during response translation. Empty
// from/to/reason fall back to "unknown" so dashboard queries never see
// the empty-string label. Numbers in reason become N, so a count from one
// request does not open a series of its own.
func RecordTranslationWarning(from, to, severity, reason string) {
	if from == "" {
		from = "unknown"
	}
	if to == "" {
		to = "unknown"
	}
	if severity == "" {
		severity = "unknown"
	}
	if reason == "" {
		reason = "unknown"
	}
	TranslationLossyTotal.WithLabelValues(from, to, severity, reasonCount.ReplaceAllString(reason, "N")).Inc()
}
