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
	fusionQuorumFailures = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_fusion_quorum_failure_total",
			Help: "Fusion panels that ended below their usable-response quorum, by decision, selected policy, and final disposition.",
		},
		// Labels are bounded: policy and disposition are closed enumerations and
		// decision names come from configuration.
		[]string{"decision", "policy", "disposition"},
	)

	fusionQuorumFallbackTargets = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_fusion_quorum_fallback_total",
			Help: "Below-quorum Fusion panels routed to a configured fallback target, by decision, target, and disposition.",
		},
		// Target names come from configuration, so cardinality is bounded by the
		// recipe rather than by traffic.
		[]string{"decision", "target", "disposition"},
	)

	fusionQuorumRequired = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_fusion_quorum_required_responses",
			Help:    "Required usable-response quorum observed on below-quorum Fusion panels.",
			Buckets: []float64{1, 2, 3, 4, 5, 6, 8, 10},
		},
		[]string{"decision"},
	)

	fusionQuorumUsable = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_fusion_quorum_usable_responses",
			Help:    "Usable responses collected on below-quorum Fusion panels.",
			Buckets: []float64{0, 1, 2, 3, 4, 5, 6, 8, 10},
		},
		[]string{"decision"},
	)

	fusionPanelAttempts = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_fusion_panel_attempt_total",
			Help: "Per-attempt terminal states observed on below-quorum Fusion panels, by decision and failure class.",
		},
		// State is a closed enumeration of panel attempt outcomes.
		[]string{"decision", "state"},
	)
)

// FusionQuorumOutcome is the bounded, content-free description of one
// below-quorum Fusion panel. AttemptStates carries one closed-enumeration state
// per panel attempt, so failure classes stay retrievable from metrics rather
// than only from Replay.
type FusionQuorumOutcome struct {
	Decision       string
	Policy         string
	Disposition    string
	FallbackTarget string
	RequiredCount  int
	UsableCount    int
	AttemptStates  []string
}

// RecordFusionQuorumFailure records one below-quorum Fusion panel and the policy
// outcome applied to it. Exactly one call is expected per panel.
func RecordFusionQuorumFailure(outcome FusionQuorumOutcome) {
	decision := valueOrUnknown(outcome.Decision)
	policy := outcome.Policy
	if policy == "" {
		policy = "fail"
	}
	disposition := valueOrUnknown(outcome.Disposition)

	fusionQuorumFailures.WithLabelValues(decision, policy, disposition).Inc()
	fusionQuorumRequired.WithLabelValues(decision).Observe(float64(outcome.RequiredCount))
	fusionQuorumUsable.WithLabelValues(decision).Observe(float64(outcome.UsableCount))
	for _, state := range outcome.AttemptStates {
		fusionPanelAttempts.WithLabelValues(decision, valueOrUnknown(state)).Inc()
	}
	if outcome.FallbackTarget != "" {
		fusionQuorumFallbackTargets.
			WithLabelValues(decision, outcome.FallbackTarget, disposition).Inc()
	}
}

func valueOrUnknown(value string) string {
	if value == "" {
		return "unknown"
	}
	return value
}
