package classification

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	circuitBreakerStateGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "llm_classifier_circuit_breaker_state",
			Help: "Current circuit breaker state (0=closed, 1=open, 2=half_open) per remote backend",
		},
		[]string{"backend"},
	)

	circuitBreakerSkipsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_classifier_circuit_breaker_skips_total",
			Help: "Total number of requests skipped per remote backend due to an open circuit breaker",
		},
		[]string{"backend"},
	)
)
