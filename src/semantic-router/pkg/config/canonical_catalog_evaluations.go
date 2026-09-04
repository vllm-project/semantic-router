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
	operatorBenchmarkID = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)+@[0-9]+(?:\.[0-9]+\.[0-9]+)?$`)
	operatorMetricID    = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]*$`)
)

func catalogEvaluationRecords(card RoutingModel, modelIndex int, builtIn *modelcatalog.Registry) ([]modelcatalog.EvaluationRecord, error) {
	records := make([]modelcatalog.EvaluationRecord, 0, len(card.Evaluations))
	for evaluationIndex, evaluation := range card.Evaluations {
		path := fmt.Sprintf("routing.modelCards[%s].evaluations[%d]", card.Name, evaluationIndex)
		if err := validateUserEvaluation(evaluation, path); err != nil {
			return nil, err
		}
		benchmark, known := builtIn.Benchmark(evaluation.Benchmark)
		if !known {
			continue
		}
		knownMetrics := make(map[string]struct{}, len(benchmark.Metrics))
		for _, metric := range benchmark.Metrics {
			knownMetrics[metric.ID] = struct{}{}
		}
		metrics := make(map[string]float64, len(evaluation.Metrics))
		for metric, value := range evaluation.Metrics {
			if _, ok := knownMetrics[metric]; !ok {
				return nil, fmt.Errorf("%s.metrics.%s is not defined by benchmark %q", path, metric, evaluation.Benchmark)
			}
			metrics[evaluation.Benchmark+"#"+metric] = value
		}
		records = append(records, modelcatalog.EvaluationRecord{
			ID:    fmt.Sprintf("operator/%d/%d/%s", modelIndex, evaluationIndex, sanitizeCatalogID(card.Name)),
			Model: card.Name, Metrics: metrics, Status: "available", MeasuredAt: evaluation.MeasuredAt,
			Subject: modelcatalog.EvaluationSubject{Parameters: cloneAnyMap(evaluation.Metadata)},
			Evidence: modelcatalog.EvaluationEvidence{
				Provenance: "operator", Verification: "claimed", Source: evaluation.Source, Redistributable: true,
			},
		})
	}
	return records, nil
}

func validateUserEvaluation(evaluation modelcatalog.UserEvaluation, path string) error {
	if !operatorBenchmarkID.MatchString(evaluation.Benchmark) {
		return fmt.Errorf("%s.benchmark must be a namespaced, versioned identity", path)
	}
	if len(evaluation.Metrics) == 0 {
		return fmt.Errorf("%s.metrics cannot be empty", path)
	}
	for metric, value := range evaluation.Metrics {
		if !operatorMetricID.MatchString(metric) || math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("%s.metrics.%s must be a finite numeric metric", path, metric)
		}
	}
	if evaluation.MeasuredAt != "" {
		if _, err := time.Parse("2006-01-02", evaluation.MeasuredAt); err != nil {
			return fmt.Errorf("%s.measured_at must use YYYY-MM-DD", path)
		}
	}
	for key, value := range evaluation.Metadata {
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

func cloneUserEvaluations(values []modelcatalog.UserEvaluation) []modelcatalog.UserEvaluation {
	if len(values) == 0 {
		return nil
	}
	result := make([]modelcatalog.UserEvaluation, len(values))
	for index, value := range values {
		result[index] = value
		result[index].Metrics = make(map[string]float64, len(value.Metrics))
		for key, metric := range value.Metrics {
			result[index].Metrics[key] = metric
		}
		result[index].Metadata = cloneAnyMap(value.Metadata)
	}
	return result
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
