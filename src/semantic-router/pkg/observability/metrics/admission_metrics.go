package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// ModelAdmissionOutcomes tracks admission gate outcomes per Router Model deployment.
	ModelAdmissionOutcomes = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_model_admission_outcomes_total",
			Help: "The total number of Router Model admission gate outcomes",
		},
		[]string{"deployment", "outcome"},
	)

	// ModelAdmissionWaitDuration tracks time spent acquiring an admission slot.
	ModelAdmissionWaitDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_model_admission_wait_seconds",
			Help:    "Time spent waiting for a Router Model admission slot in seconds",
			Buckets: []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5},
		},
		[]string{"deployment"},
	)
)

func RecordModelAdmission(deployment, outcome string, waitSeconds float64) {
	ModelAdmissionOutcomes.WithLabelValues(deployment, outcome).Inc()
	ModelAdmissionWaitDuration.WithLabelValues(deployment).Observe(waitSeconds)
}
