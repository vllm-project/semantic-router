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
	contextWindowWait = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "llm_context_window_wait_seconds",
		Help:    "Time a request waited for its context window count after its signals; the count runs beside the signals, so this is about zero.",
		Buckets: []float64{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1},
	})
	contextWindowMargins = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "llm_context_window_margin",
		Help: "Band around a context window in which the Router's count decides nothing, as a fraction of the window, by count: exact (global.router.context_window.margin) or proxy (proxy_margin).",
	}, []string{"count"})
	contextWindowTokenizers = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "llm_context_window_tokenizer_loaded",
		Help: "1 once a tokenizer the context window check counts with has loaded; 0 while it is downloading, loading or failed, when its models count with the proxy tokenizer and its band.",
	}, []string{"tokenizer"})
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

// RecordContextWindowWait observes how long a request waited for its count.
func RecordContextWindowWait(seconds float64) {
	contextWindowWait.Observe(seconds)
}

// SetContextWindowTokenizerLoaded publishes whether a tokenizer has loaded.
func SetContextWindowTokenizerLoaded(tokenizer string, loaded bool) {
	value := 0.0
	if loaded {
		value = 1
	}
	contextWindowTokenizers.WithLabelValues(tokenizer).Set(value)
}
