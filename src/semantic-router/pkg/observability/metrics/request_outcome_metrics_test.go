package metrics

import (
	"math"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

func TestRequestOutcomeMetricsExposeZeroBaselineBeforeFirstRequest(t *testing.T) {
	registry := prometheus.NewRegistry()
	newRequestOutcomeMetrics(registry)
	counters := gatherRequestMetricFamily(t, registry, "llm_request_outcomes_total")
	histograms := gatherRequestMetricFamily(t, registry, "llm_request_duration_seconds")
	if len(counters) != 42 || len(histograms) != 42 {
		t.Fatalf("startup series: counters=%d histograms=%d; want 6 kinds x 7 outcomes", len(counters), len(histograms))
	}
	for _, kind := range []string{"inference", "inference_internal", "catalog", "response_object", "health", "other"} {
		for _, outcome := range []string{"success", "client_error", "server_error", "canceled", "timeout", "incomplete", "error"} {
			key := kind + "/" + outcome
			counter, histogram := counters[key], histograms[key]
			if counter == nil || histogram == nil {
				t.Fatalf("startup scrape lacks %s", key)
			}
			if counter.GetCounter().GetValue() != 0 {
				t.Fatalf("startup counter %s is not zero", key)
			}
			assertRequestHistogram(t, histogram.GetHistogram(), 0, 0)
		}
	}
}

func TestRequestOutcomeMetricsFirstEventsIncrementExistingSeries(t *testing.T) {
	registry := prometheus.NewRegistry()
	metrics := newRequestOutcomeMetrics(registry)
	before := gatherRequestMetricFamily(t, registry, "llm_request_outcomes_total")
	metrics.record("inference", "success", 1.5)
	metrics.record("inference", "success", 2.5)
	metrics.record("private-path", "raw-provider-error", 600)
	after := gatherRequestMetricFamily(t, registry, "llm_request_outcomes_total")
	histograms := gatherRequestMetricFamily(t, registry, "llm_request_duration_seconds")
	if len(after) != len(before) || len(histograms) != len(before) {
		t.Fatal("recording requests changed the bounded startup series set")
	}
	for key, counter := range after {
		var count uint64
		var sum float64
		switch key {
		case "inference/success":
			count, sum = 2, 4
		case "other/error":
			count, sum = 1, 600
		}
		baseline, exists := before[key]
		if !exists || counter.GetCounter().GetValue()-baseline.GetCounter().GetValue() != float64(count) {
			t.Fatalf("%s lacks its zero baseline or has an incorrect first-scrape delta", key)
		}
		assertRequestHistogram(t, histograms[key].GetHistogram(), count, sum)
	}
}

func TestRequestOutcomeMetricsDoNotInventDurationObservations(t *testing.T) {
	for name, seconds := range map[string]float64{
		"zero": 0, "negative": -1, "nan": math.NaN(), "positive infinity": math.Inf(1), "negative infinity": math.Inf(-1),
	} {
		t.Run(name, func(t *testing.T) {
			registry := prometheus.NewRegistry()
			metrics := newRequestOutcomeMetrics(registry)
			metrics.record("inference", "incomplete", seconds)
			counter := gatherRequestMetricFamily(t, registry, "llm_request_outcomes_total")["inference/incomplete"]
			if counter.GetCounter().GetValue() != 1 {
				t.Fatal("terminal outcome missing despite unavailable duration")
			}
			histogram := gatherRequestMetricFamily(t, registry, "llm_request_duration_seconds")["inference/incomplete"]
			assertRequestHistogram(t, histogram.GetHistogram(), 0, 0)
		})
	}
}

func gatherRequestMetricFamily(t *testing.T, registry *prometheus.Registry, name string) map[string]*dto.Metric {
	t.Helper()
	families, err := registry.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
		samples := make(map[string]*dto.Metric, len(family.Metric))
		for _, sample := range family.Metric {
			labels := make(map[string]string, len(sample.Label))
			for _, label := range sample.Label {
				labels[label.GetName()] = label.GetValue()
			}
			if len(labels) != 2 || labels["traffic_kind"] == "" || labels["outcome"] == "" {
				t.Fatalf("unexpected labels in %s", name)
			}
			key := labels["traffic_kind"] + "/" + labels["outcome"]
			if _, exists := samples[key]; exists {
				t.Fatalf("duplicate metric sample %s", key)
			}
			samples[key] = sample
		}
		return samples
	}
	t.Fatalf("scrape lacks metric family %s", name)
	return nil
}

func assertRequestHistogram(t *testing.T, histogram *dto.Histogram, count uint64, sum float64) {
	t.Helper()
	if histogram.GetSampleCount() != count || histogram.GetSampleSum() != sum {
		t.Fatalf("duration observations: count=%d sum=%g; want count=%d sum=%g", histogram.GetSampleCount(), histogram.GetSampleSum(), count, sum)
	}
	buckets := histogram.GetBucket()
	if len(buckets) != 18 || buckets[len(buckets)-1].GetUpperBound() != 1800 {
		t.Fatal("long response durations lack the expected finite buckets")
	}
	for _, bucket := range buckets {
		if bucket.GetCumulativeCount() > count {
			t.Fatal("histogram contains unrecorded duration observations")
		}
	}
	if buckets[len(buckets)-1].GetCumulativeCount() != count {
		t.Fatal("long response duration was not observed in the 1800-second bucket")
	}
}
