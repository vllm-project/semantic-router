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

package extproc

import (
	"sort"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

// quorumMetricSamples reads real Prometheus state rather than trusting an
// in-memory field. The metric vectors stay unexported, so tests gather from the
// default registry and select by metric name and label values.
//
// Counters return their value; histograms return their observation count.
func quorumMetricSamples(t *testing.T, metricName string, match map[string]string) float64 {
	t.Helper()
	families, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatalf("gather metrics: %v", err)
	}
	total := 0.0
	for _, family := range families {
		if family.GetName() != metricName {
			continue
		}
		for _, metric := range family.GetMetric() {
			if !quorumLabelsMatch(metric, match) {
				continue
			}
			switch {
			case metric.GetCounter() != nil:
				total += metric.GetCounter().GetValue()
			case metric.GetHistogram() != nil:
				total += float64(metric.GetHistogram().GetSampleCount())
			}
		}
	}
	return total
}

func quorumLabelsMatch(metric *dto.Metric, match map[string]string) bool {
	for name, want := range match {
		found := false
		for _, pair := range metric.GetLabel() {
			if pair.GetName() == name && pair.GetValue() == want {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// quorumDispositionsSeen returns every disposition label value currently present
// for a decision, so a test can assert that an internal state never escapes.
func quorumDispositionsSeen(t *testing.T, decision string) map[string]bool {
	t.Helper()
	families, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatalf("gather metrics: %v", err)
	}
	seen := map[string]bool{}
	for _, family := range families {
		name := family.GetName()
		if name != "llm_fusion_quorum_failure_total" && name != "llm_fusion_quorum_fallback_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			if !quorumLabelsMatch(metric, map[string]string{"decision": decision}) {
				continue
			}
			for _, pair := range metric.GetLabel() {
				if pair.GetName() == "disposition" {
					seen[pair.GetValue()] = true
				}
			}
		}
	}
	return seen
}

// quorumSeries names one metric plus the exact label set that must carry it.
// Matching on the full label set is the point: matching only decision and
// disposition would still pass if policy or target were emitted with the wrong
// value, and those are part of #3375's telemetry contract.
type quorumSeries struct {
	metric string
	labels map[string]string
}

func quorumFailureSeries(decision, policy, disposition string) quorumSeries {
	return quorumSeries{
		metric: "llm_fusion_quorum_failure_total",
		labels: map[string]string{"decision": decision, "policy": policy, "disposition": disposition},
	}
}

func quorumFallbackSeries(decision, target, disposition string) quorumSeries {
	return quorumSeries{
		metric: "llm_fusion_quorum_fallback_total",
		labels: map[string]string{"decision": decision, "target": target, "disposition": disposition},
	}
}

func quorumAttemptSeries(decision, state string) quorumSeries {
	return quorumSeries{
		metric: "llm_fusion_panel_attempt_total",
		labels: map[string]string{"decision": decision, "state": state},
	}
}

func quorumHistogramSeries(decision, metric string) quorumSeries {
	return quorumSeries{metric: metric, labels: map[string]string{"decision": decision}}
}

// quorumBaseline snapshots a set of series so tests assert deltas. The vectors
// are process-global, so absolute values only hold on a fresh process.
type quorumBaseline map[quorumSeriesKey]float64

type quorumSeriesKey struct {
	metric string
	labels string
}

func (s quorumSeries) key() quorumSeriesKey {
	names := make([]string, 0, len(s.labels))
	for name := range s.labels {
		names = append(names, name)
	}
	sort.Strings(names)
	encoded := ""
	for _, name := range names {
		encoded += name + "=" + s.labels[name] + ";"
	}
	return quorumSeriesKey{metric: s.metric, labels: encoded}
}

func captureQuorumSeries(t *testing.T, series ...quorumSeries) quorumBaseline {
	t.Helper()
	baseline := quorumBaseline{}
	for _, one := range series {
		baseline[one.key()] = quorumMetricSamples(t, one.metric, one.labels)
	}
	return baseline
}

// delta returns the change in one series since the baseline was captured.
func (b quorumBaseline) delta(t *testing.T, series quorumSeries) float64 {
	t.Helper()
	return quorumMetricSamples(t, series.metric, series.labels) - b[series.key()]
}
