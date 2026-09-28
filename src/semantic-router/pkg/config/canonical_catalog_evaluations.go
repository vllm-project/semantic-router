package config

import (
	"fmt"
	"math"
	"regexp"
	"strings"
	"time"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

var (
	operatorResourceID = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)+@[0-9]+(?:\.[0-9]+\.[0-9]+)?$`)
	operatorMetricID   = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]*$`)
)

// catalogIndexableEvaluationRecords adapts operator measurements whose
// benchmark semantics are built in or declared in evaluation.benchmarks into
// the typed scoring graph. Unknown namespaced benchmarks remain in the
// canonical evaluation records and round-trip through export, but cannot
// safely enter an index until their metric range, direction, and profiles are
// declared.
func catalogIndexableEvaluationRecords(
	records []CanonicalEvaluationRecord,
	benchmarks map[string]modelcatalog.BenchmarkDefinition,
	configuredCards map[string]struct{},
) ([]modelcatalog.EvaluationRecord, error) {
	indexable := make([]modelcatalog.EvaluationRecord, 0, len(records))
	for recordIndex, record := range records {
		path := fmt.Sprintf("evaluation.records[%d]", recordIndex)
		if err := validateCanonicalEvaluationRecord(record, path); err != nil {
			return nil, err
		}
		if _, exists := configuredCards[record.Model]; !exists {
			return nil, fmt.Errorf("%s.model %q does not reference a configured Model Card identity", path, record.Model)
		}
		benchmark, known := benchmarks[record.Benchmark]
		if !known {
			continue
		}
		knownMetrics := make(map[string]struct{}, len(benchmark.Metrics))
		for _, metric := range benchmark.Metrics {
			knownMetrics[metric.ID] = struct{}{}
		}
		metrics := make(map[string]float64, len(record.Metrics))
		for metric, value := range record.Metrics {
			if _, ok := knownMetrics[metric]; !ok {
				return nil, fmt.Errorf("%s.metrics.%s is not defined by benchmark %q", path, metric, record.Benchmark)
			}
			metrics[metric] = value
		}
		benchmarkProfile := record.BenchmarkProfile
		if benchmarkProfile == "" {
			benchmarkProfile = benchmark.DefaultProfile
		}
		reasoningEffort := record.ReasoningEffort
		if reasoningEffort == "" {
			reasoningEffort = "default"
		}
		subject := modelcatalog.EvaluationSubject{}
		if len(record.Metadata) > 0 {
			subject["parameters"] = cloneAnyMap(record.Metadata)
		}
		indexable = append(indexable, modelcatalog.EvaluationRecord{
			ID:    fmt.Sprintf("operator/%d/%s", recordIndex, sanitizeCatalogID(record.Model)),
			Model: record.Model, Benchmark: record.Benchmark, BenchmarkProfile: benchmarkProfile,
			ReasoningEffort: reasoningEffort, Metrics: metrics, Status: "available", MeasuredAt: record.MeasuredAt,
			Subject: subject,
			Evidence: modelcatalog.EvaluationEvidence{
				Provenance: "operator", Verification: "claimed", Source: record.Source, Redistributable: true,
			},
		})
	}
	return indexable, nil
}

func validateCanonicalEvaluationRecord(record CanonicalEvaluationRecord, path string) error {
	if strings.TrimSpace(record.Model) == "" {
		return fmt.Errorf("%s.model cannot be empty", path)
	}
	if !operatorResourceID.MatchString(record.Benchmark) {
		return fmt.Errorf("%s.benchmark must be a namespaced, versioned identity", path)
	}
	if len(record.Metrics) == 0 {
		return fmt.Errorf("%s.metrics cannot be empty", path)
	}
	for metric, value := range record.Metrics {
		if !operatorMetricID.MatchString(metric) || math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("%s.metrics.%s must be a finite numeric metric", path, metric)
		}
	}
	if record.MeasuredAt != "" {
		if _, err := time.Parse("2006-01-02", record.MeasuredAt); err != nil {
			return fmt.Errorf("%s.measured_at must use YYYY-MM-DD", path)
		}
	}
	for key, value := range record.Metadata {
		if strings.TrimSpace(key) == "" || !catalogMetadataScalar(value) {
			return fmt.Errorf("%s.metadata must contain non-empty scalar key/value pairs", path)
		}
	}
	return nil
}

func catalogMetadataScalar(value any) bool {
	switch typed := value.(type) {
	case nil, string, bool, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return true
	case float32:
		return !math.IsNaN(float64(typed)) && !math.IsInf(float64(typed), 0)
	case float64:
		return !math.IsNaN(typed) && !math.IsInf(typed, 0)
	default:
		return false
	}
}

func cloneAnyMap(values map[string]any) map[string]any {
	if values == nil {
		return nil
	}
	result := make(map[string]any, len(values))
	for key, value := range values {
		result[key] = value
	}
	return result
}

func sanitizeCatalogID(value string) string {
	replacer := strings.NewReplacer("/", "-", "@", "-", ":", "-", ".", "-")
	return strings.ToLower(replacer.Replace(value))
}
