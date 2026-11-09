package modelservice

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

var decisionMediaSkipped = promauto.NewCounterVec(prometheus.CounterOpts{
	Name: "vsr_decision_media_skipped_total", Help: "Attachments skipped when preparing Router decision calls.",
}, []string{"kind", "reason"})

// RecordMediaSkip accepts only bounded labels; media and URLs are never logged.
func RecordMediaSkip(kind, reason string) {
	if kind != "image" && kind != "video" {
		return
	}
	switch reason {
	case "remote_url", "unsupported_type", "too_large", "over_count", "malformed":
	default:
		return
	}
	decisionMediaSkipped.WithLabelValues(kind, reason).Inc()
	logging.Debugf("Decision media skipped: kind=%s reason=%s", kind, reason)
}
