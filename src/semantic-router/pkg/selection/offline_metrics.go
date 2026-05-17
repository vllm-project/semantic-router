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

package selection

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// OfflineUpdateTotal counts offline weight update runs.
	// Labels: status (success/error)
	OfflineUpdateTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_offline_weight_update_total",
			Help: "Total offline weight update runs by status.",
		},
		[]string{"status"},
	)

	// OfflineUpdateRecordsProcessed counts records consumed per offline run.
	OfflineUpdateRecordsProcessed = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "llm_offline_weight_update_records_processed_total",
			Help: "Total records processed across all offline weight update runs.",
		},
	)

	// PolicyVersionActivations counts policy version promotions.
	// Labels: version_id, source (offline_batch/online/manual/import)
	PolicyVersionActivations = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_policy_version_activations_total",
			Help: "Total policy version activations by version and source.",
		},
		[]string{"version_id", "source"},
	)

	// ShadowComparisonTotal counts shadow comparisons between active and shadow policies.
	// Labels: agreed (true/false)
	ShadowComparisonTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_shadow_comparison_total",
			Help: "Total shadow comparisons between active and candidate policies.",
		},
		[]string{"agreed"},
	)

	// ShadowComparisonScoreDelta tracks the score difference between active and shadow policies.
	ShadowComparisonScoreDelta = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "llm_shadow_comparison_score_delta",
			Help:    "Distribution of score deltas (shadow - active) in shadow comparisons.",
			Buckets: []float64{-0.5, -0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3, 0.5},
		},
	)
)

// RecordOfflineUpdate records the result of an offline weight update run.
func RecordOfflineUpdate(success bool, recordCount int) {
	status := "success"
	if !success {
		status = "error"
	}
	OfflineUpdateTotal.WithLabelValues(status).Inc()
	if success {
		OfflineUpdateRecordsProcessed.Add(float64(recordCount))
	}
}

// RecordPolicyActivation records a policy version activation event.
func RecordPolicyActivation(versionID, source string) {
	PolicyVersionActivations.WithLabelValues(versionID, source).Inc()
}

// RecordShadowComparison records a shadow comparison between active and candidate.
func RecordShadowComparison(comparison ShadowComparison) {
	agreed := "false"
	if comparison.Agreed {
		agreed = "true"
	}
	ShadowComparisonTotal.WithLabelValues(agreed).Inc()
	ShadowComparisonScoreDelta.Observe(comparison.ShadowScore - comparison.ActiveScore)
}
