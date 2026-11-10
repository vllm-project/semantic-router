package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	SystemOneSignalCalls = promauto.NewCounter(prometheus.CounterOpts{
		Name: "sr_systemone_signal_model_calls_total",
		Help: "Physical inference exchanges used while evaluating System One signals, including transport retries",
	})
	SystemOneStageTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "sr_systemone_stage_total",
		Help: "System One auto stage attempts by configured algorithm, stage, model and outcome",
	}, []string{"algorithm", "stage", "model", "outcome"})
	SystemOneStageDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "sr_systemone_stage_duration_seconds",
		Help:    "System One auto stage wall time including transport and acceptance evaluation",
		Buckets: []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30},
	}, []string{"algorithm", "stage", "model"})
	SystemOneStageCalls = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "sr_systemone_stage_model_calls_total",
		Help: "Physical inference exchanges used by each System One auto stage, including transport retries",
	}, []string{"algorithm", "stage", "model"})
	SystemOneRequestTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "sr_systemone_auto_requests_total",
		Help: "System One auto requests by algorithm and final execution outcome",
	}, []string{"algorithm", "outcome"})
	SystemOneRequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "sr_systemone_auto_duration_seconds",
		Help:    "System One auto algorithm wall time, excluding preceding recipe signals",
		Buckets: []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30},
	}, []string{"algorithm"})
)

// RecordSystemOneStage uses only operator-declared labels and a fixed outcome
// vocabulary. User questions, state values and request IDs are never labels.
func RecordSystemOneStage(algorithm, stage, model, outcome string, seconds float64) {
	SystemOneStageTotal.WithLabelValues(algorithm, stage, model, outcome).Inc()
	SystemOneStageDuration.WithLabelValues(algorithm, stage, model).Observe(seconds)
}

func RecordSystemOneStageCalls(algorithm, stage, model string, calls int) {
	if calls > 0 {
		SystemOneStageCalls.WithLabelValues(algorithm, stage, model).Add(float64(calls))
	}
}

func RecordSystemOneSignalCalls(calls int) {
	if calls > 0 {
		SystemOneSignalCalls.Add(float64(calls))
	}
}

func RecordSystemOneRequest(algorithm, outcome string, seconds float64) {
	SystemOneRequestTotal.WithLabelValues(algorithm, outcome).Inc()
	SystemOneRequestDuration.WithLabelValues(algorithm).Observe(seconds)
}
