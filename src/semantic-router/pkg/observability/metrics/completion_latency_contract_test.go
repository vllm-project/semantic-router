package metrics

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

func TestCompletionLatencyBucketsCoverMaintainedAlerts(t *testing.T) {
	model := t.Name()
	RecordModelCompletionLatency(model, 45)
	RecordModelCompletionLatency(model, 600)
	var metric dto.Metric
	if err := ModelCompletionLatency.WithLabelValues(model).(prometheus.Metric).Write(&metric); err != nil {
		t.Fatal(err)
	}
	histogram := metric.GetHistogram()
	if histogram.GetSampleCount() != 2 || histogram.GetSampleSum() != 645 {
		t.Fatalf("unexpected completion observations: %v", histogram)
	}
	buckets := make(map[float64]uint64)
	for _, bucket := range histogram.GetBucket() {
		buckets[bucket.GetUpperBound()] = bucket.GetCumulativeCount()
	}
	if buckets[30] != 0 || buckets[60] != 1 || buckets[600] != 2 || buckets[1800] != 2 {
		t.Fatalf("slow completions must occupy finite buckets beyond 30s: %v", buckets)
	}
	for _, deployment := range []string{
		"deploy/kubernetes/observability/prometheus/rules.yaml",
		"deploy/openshift/observability/prometheus/rules.yaml",
		"deploy/helm/semantic-router/templates/prometheus-rule.yaml",
	} {
		t.Run(deployment, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join("../../../../..", deployment))
			if err != nil {
				t.Fatal(err)
			}
			threshold := completionAlertThreshold(t, string(raw))
			var aboveThreshold bool
			for upper := range buckets {
				aboveThreshold = aboveThreshold || upper > threshold
			}
			if !aboveThreshold {
				t.Fatalf("completion P95 can never exceed maintained alert threshold %gs", threshold)
			}
		})
	}
}

func completionAlertThreshold(t *testing.T, rules string) float64 {
	t.Helper()
	_, alert, found := strings.Cut(rules, "- alert: HighCompletionLatencyP95")
	if !found {
		t.Fatal("completion latency alert is missing")
	}
	alert, _, _ = strings.Cut(alert, "- alert:")
	expression := regexp.MustCompile(`histogram_quantile\(0\.95, sum\(rate\(llm_model_completion_latency_seconds_bucket\[5m\]\)\) by \(le\)\) > (.+)`).FindStringSubmatch(alert)
	if len(expression) != 2 {
		t.Fatal("completion latency alert must consume the model completion histogram")
	}
	value := strings.TrimSpace(expression[1])
	if strings.HasPrefix(value, "{{") {
		match := regexp.MustCompile(`"completionLatencyP95Seconds" \| default ([0-9.]+)`).FindStringSubmatch(value)
		if len(match) != 2 {
			t.Fatal("completion latency alert has no explicit default threshold")
		}
		value = match[1]
	}
	threshold, err := strconv.ParseFloat(value, 64)
	if err != nil || threshold <= 0 {
		t.Fatalf("invalid completion latency threshold %q: %v", value, err)
	}
	return threshold
}
