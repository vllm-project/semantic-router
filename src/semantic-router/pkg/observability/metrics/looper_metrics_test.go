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
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestLooperMetricsRecordBoundedDimensions(t *testing.T) {
	firstByte := int64(125)
	cost := 0.004
	before := testutil.ToFloat64(LooperAttemptsTotal.WithLabelValues(
		"confidence", "candidate", "succeeded", "threshold_met",
	))

	RecordLooperAttempt(
		"confidence", "candidate", "succeeded", "threshold_met",
		250, &firstByte, 10, 5, &cost, "USD",
	)

	after := testutil.ToFloat64(LooperAttemptsTotal.WithLabelValues(
		"confidence", "candidate", "succeeded", "threshold_met",
	))
	if after != before+1 {
		t.Fatalf("attempt counter = %v, want %v", after, before+1)
	}
	if got := testutil.ToFloat64(LooperAttemptTokens.WithLabelValues("confidence", "candidate", "prompt")); got < 10 {
		t.Fatalf("prompt token counter = %v, want at least 10", got)
	}

	assertMetricHasNoLabels(t, LooperAttemptsTotal, "model", "decision", "recipe", "ordinal", "request_id", "trace_id")
}

func assertMetricHasNoLabels(t *testing.T, collector prometheus.Collector, forbidden ...string) {
	t.Helper()
	descriptions := make(chan *prometheus.Desc, 10)
	go func() {
		collector.Describe(descriptions)
		close(descriptions)
	}()
	for description := range descriptions {
		text := description.String()
		for _, label := range forbidden {
			if strings.Contains(text, "variableLabels: {"+label+",") ||
				strings.Contains(text, ","+label+",") || strings.Contains(text, ","+label+"}") {
				t.Fatalf("metric descriptor contains forbidden label %q: %s", label, text)
			}
		}
	}
}
