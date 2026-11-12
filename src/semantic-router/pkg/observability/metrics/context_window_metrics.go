package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// The context window check counts a prompt with a model's own tokenizer
// (count "exact") or with the proxy tokenizer (count "proxy"), and places
// the count against the window and the band of that kind of count.
var (
	contextWindowChecks = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "llm_context_window_checks_total",
		Help: "Context window checks decided on the Router's prompt count, once per model and prompt shape of a request, by count (exact or proxy) and verdict (fits, in_band or exceeds).",
	}, []string{"count", "verdict"})
	contextWindowRefusals = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "llm_context_window_refusals_total",
		Help: "Requests the Router refused with context_length_exceeded on its own count, by the count of the widest window it named (exact or proxy).",
	}, []string{"count"})
	contextWindowMargins = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "llm_context_window_margin",
		Help: "Band around a context window in which the Router's count decides nothing, as a fraction of the window, by count: exact (global.router.context_window.margin) or proxy (proxy_margin).",
	}, []string{"count"})
)

// RecordContextWindowCheck counts one window check decided on a count.
func RecordContextWindowCheck(count, verdict string) {
	contextWindowChecks.WithLabelValues(count, verdict).Inc()
}

// RecordContextWindowRefusal counts one refusal for length.
func RecordContextWindowRefusal(count string) {
	contextWindowRefusals.WithLabelValues(count).Inc()
}

// SetContextWindowMargins publishes the configured bands.
func SetContextWindowMargins(exact, proxy float64) {
	contextWindowMargins.WithLabelValues("exact").Set(exact)
	contextWindowMargins.WithLabelValues("proxy").Set(proxy)
}
