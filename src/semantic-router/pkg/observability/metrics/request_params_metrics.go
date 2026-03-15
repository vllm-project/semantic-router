package metrics

import (
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	RequestParamsBlocked = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "sr_request_params_blocked_total",
			Help: "Total number of blocked request parameters",
		},
		[]string{"decision", "param"},
	)

	RequestParamsMaxTokensCapped = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "sr_request_params_max_tokens_capped_total",
			Help: "Total number of times max_tokens was capped",
		},
		[]string{"decision", "original", "capped"},
	)

	RequestParamsMaxNCapped = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "sr_request_params_max_n_capped_total",
			Help: "Total number of times n was capped",
		},
		[]string{"decision", "original", "capped"},
	)

	RequestParamsUnknownFieldStripped = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "sr_request_params_unknown_field_stripped_total",
			Help: "Total number of unknown fields stripped from requests",
		},
		[]string{"decision", "field"},
	)
)

func RecordBlockedParam(decision, param string) {
	RequestParamsBlocked.WithLabelValues(decision, param).Inc()
}

func RecordMaxTokensCapped(decision string, original, capped int) {
	RequestParamsMaxTokensCapped.WithLabelValues(decision, fmt.Sprintf("%d", original), fmt.Sprintf("%d", capped)).Inc()
}

func RecordMaxNCapped(decision string, original, capped int) {
	RequestParamsMaxNCapped.WithLabelValues(decision, fmt.Sprintf("%d", original), fmt.Sprintf("%d", capped)).Inc()
}

func RecordUnknownFieldStripped(decision, field string) {
	RequestParamsUnknownFieldStripped.WithLabelValues(decision, field).Inc()
}
