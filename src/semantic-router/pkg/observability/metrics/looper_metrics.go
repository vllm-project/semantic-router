/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	LooperAttemptsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "llm_looper_attempts_total",
		Help: "Total Looper attempts by bounded execution outcome.",
	}, []string{"algorithm", "stage", "status", "reason"})

	LooperAttemptDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "llm_looper_attempt_duration_seconds",
		Help:    "Looper attempt duration in seconds.",
		Buckets: prometheus.DefBuckets,
	}, []string{"algorithm", "stage", "status"})

	LooperAttemptFirstByte = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "llm_looper_attempt_first_byte_seconds",
		Help:    "Looper attempt time to first response byte in seconds.",
		Buckets: prometheus.DefBuckets,
	}, []string{"algorithm", "stage"})

	LooperAttemptTokens = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "llm_looper_attempt_tokens_total",
		Help: "Total prompt and completion tokens consumed by Looper attempts.",
	}, []string{"algorithm", "stage", "token_type"})

	LooperAttemptCost = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "llm_looper_attempt_cost_total",
		Help: "Total configured cost of Looper attempts by currency.",
	}, []string{"algorithm", "stage", "currency"})

	LooperExecutionDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "llm_looper_execution_duration_seconds",
		Help:    "End-to-end Looper execution duration in seconds.",
		Buckets: prometheus.DefBuckets,
	}, []string{"algorithm", "status"})
)

// RecordLooperAttempt records one terminal attempt. All labels are bounded by
// the Looper package; request-specific identifiers intentionally stay out.
func RecordLooperAttempt(
	algorithm, stage, status, reason string,
	totalLatencyMs int64,
	firstByteLatencyMs *int64,
	promptTokens, completionTokens int64,
	actualCost *float64,
	currency string,
) {
	LooperAttemptsTotal.WithLabelValues(algorithm, stage, status, reason).Inc()
	LooperAttemptDuration.WithLabelValues(algorithm, stage, status).Observe(float64(totalLatencyMs) / 1000)
	if firstByteLatencyMs != nil {
		LooperAttemptFirstByte.WithLabelValues(algorithm, stage).Observe(float64(*firstByteLatencyMs) / 1000)
	}
	if promptTokens > 0 {
		LooperAttemptTokens.WithLabelValues(algorithm, stage, "prompt").Add(float64(promptTokens))
	}
	if completionTokens > 0 {
		LooperAttemptTokens.WithLabelValues(algorithm, stage, "completion").Add(float64(completionTokens))
	}
	if actualCost != nil && currency != "" {
		LooperAttemptCost.WithLabelValues(algorithm, stage, currency).Add(*actualCost)
	}
}

func RecordLooperExecution(algorithm, status string, durationSeconds float64) {
	LooperExecutionDuration.WithLabelValues(algorithm, status).Observe(durationSeconds)
}
