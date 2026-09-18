package evaluationplane

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
)

type capacityProfileAssessment struct {
	QualifiedConcurrency  json.RawMessage `json:"qualified_concurrency"`
	SaturationConcurrency json.RawMessage `json:"saturation_concurrency"`
	SLOHeadroom           *int64          `json:"slo_headroom"`
	Verdict               string          `json:"verdict"`
	FailureReasons        []string        `json:"failure_reasons"`
}

type capacitySLOAttestation struct {
	Headroom                     float64
	LevelCount                   int
	MeasurementClusterCount      int
	MinimumClustersPerLevel      int64
	RequiredClustersPerLevel     int64
	WorstErrorRateUpperBound     float64
	WorstErrorRateClusterRange   float64
	ReleaseErrorRateUpperBound   float64
	ReleaseErrorRateClusterRange float64
	MaxErrorRate                 float64
	MaxErrorRateClusterRange     float64
	MeanErrorRate                float64
}

func validateCapacitySLOMetric(report Report, attestation *capacitySLOAttestation) error {
	metrics := make(map[string]Metric, len(report.Metrics))
	for _, metric := range report.Metrics {
		metrics[metric.ID] = metric
	}
	selected := containsTrack(report.Run.TrackIDs, "capacity")
	if !selected {
		for _, metricID := range capacityClusterAttestedMetricIDs() {
			if _, present := metrics[metricID]; present {
				return fmt.Errorf("%w: capacity metric is published for an unselected track", ErrInvalid)
			}
		}
		return nil
	}
	if attestation != nil {
		if report.Run.CapacitySLO == nil || report.Run.CapacityLoadProtocol == nil ||
			attestation.RequiredClustersPerLevel != report.Run.CapacityLoadProtocol.MinimumMeasurementClustersPerLevel ||
			attestation.MaxErrorRate != report.Run.CapacitySLO.MaxErrorRate ||
			attestation.MaxErrorRateClusterRange != report.Run.CapacityLoadProtocol.MaxErrorRateClusterRange {
			return fmt.Errorf("%w: capacity cluster attestation differs from the frozen run contract", ErrInvalid)
		}
		if attestation.Headroom >= 0 &&
			(attestation.MinimumClustersPerLevel < attestation.RequiredClustersPerLevel ||
				attestation.ReleaseErrorRateUpperBound > report.Run.CapacitySLO.MaxErrorRate ||
				attestation.ReleaseErrorRateClusterRange > attestation.MaxErrorRateClusterRange) {
			return fmt.Errorf("%w: capacity headroom is not supported by independent-cluster evidence", ErrInvalid)
		}
	}
	for _, expected := range capacityClusterAttestedMetrics(attestation) {
		actual, present := metrics[expected.ID]
		if !present || actual.Name != expected.Name || actual.TrackID != "capacity" ||
			actual.Unit != expected.Unit || actual.Direction != expected.Direction ||
			actual.SampleCount != expected.SampleCount || actual.ConfidenceInterval != nil ||
			(actual.Value == nil) != (expected.Value == nil) ||
			(actual.Value != nil && !reducedFloatsEqual(*actual.Value, *expected.Value)) {
			return fmt.Errorf("%w: capacity metric %s does not match the server-reduced cluster profile", ErrInvalid, expected.ID)
		}
	}
	return nil
}

func capacityClusterAttestedMetricIDs() []string {
	return []string{
		"capacity.error_rate", "capacity.error_rate_cluster_range_max",
		"capacity.error_rate_upper_bound", "capacity.measurement_cluster_count_min",
		"capacity.slo_headroom", "capacity.success_rate",
	}
}

func capacityClusterAttestedMetrics(attestation *capacitySLOAttestation) []Metric {
	metrics := []Metric{
		{ID: "capacity.error_rate", Name: "Mean independent-cluster error rate", TrackID: "capacity", Unit: "fraction", Direction: "lower_is_better"},
		{ID: "capacity.error_rate_cluster_range_max", Name: "Worst independent-cluster error-rate range", TrackID: "capacity", Unit: "fraction", Direction: "lower_is_better"},
		{ID: "capacity.error_rate_upper_bound", Name: "Worst independent-cluster one-sided 95% error-rate upper bound", TrackID: "capacity", Unit: "fraction", Direction: "lower_is_better"},
		{ID: "capacity.measurement_cluster_count_min", Name: "Minimum independent clusters per concurrency level", TrackID: "capacity", Unit: "clusters", Direction: "target"},
		{ID: "capacity.slo_headroom", Name: "Qualified concurrency above the frozen SLO requirement", TrackID: "capacity", Unit: "concurrency", Direction: "higher_is_better"},
		{ID: "capacity.success_rate", Name: "Mean independent-cluster success rate", TrackID: "capacity", Unit: "fraction", Direction: "higher_is_better"},
	}
	if attestation == nil {
		return metrics
	}
	metrics[0].Value, metrics[0].SampleCount = capacityFloatPointer(attestation.MeanErrorRate), attestation.MeasurementClusterCount
	metrics[1].Value, metrics[1].SampleCount = capacityFloatPointer(attestation.WorstErrorRateClusterRange), attestation.LevelCount
	metrics[2].Value, metrics[2].SampleCount = capacityFloatPointer(attestation.WorstErrorRateUpperBound), attestation.MeasurementClusterCount
	metrics[3].Value, metrics[3].SampleCount = capacityFloatPointer(float64(attestation.MinimumClustersPerLevel)), attestation.LevelCount
	metrics[4].Value, metrics[4].SampleCount = capacityFloatPointer(attestation.Headroom), attestation.LevelCount
	metrics[5].Value, metrics[5].SampleCount = capacityFloatPointer(1-attestation.MeanErrorRate), attestation.MeasurementClusterCount
	return metrics
}

func validateCapacityAssessment(
	profile capacityProfileEvidence,
	levels []reducedCapacityLevel,
) (int64, error) {
	if profile.Assessment.SLOHeadroom == nil || profile.Assessment.FailureReasons == nil {
		return 0, fmt.Errorf("assessment requires headroom and failure reasons")
	}
	actualQualified, qualifiedPresent, err := decodeCapacityOptionalInt(
		"assessment.qualified_concurrency",
		profile.Assessment.QualifiedConcurrency,
	)
	if err != nil {
		return 0, err
	}
	actualSaturation, saturationPresent, err := decodeCapacityOptionalInt(
		"assessment.saturation_concurrency",
		profile.Assessment.SaturationConcurrency,
	)
	if err != nil {
		return 0, err
	}

	qualified := int64(0)
	expectedQualifiedPresent := false
	saturation := int64(0)
	expectedSaturationPresent := false
	for _, level := range levels {
		if level.qualified {
			qualified = level.concurrency
			expectedQualifiedPresent = true
		} else if !expectedSaturationPresent {
			saturation = level.concurrency
			expectedSaturationPresent = true
		}
	}
	headroom := -profile.SLO.RequiredConcurrency
	if expectedQualifiedPresent {
		headroom = qualified - profile.SLO.RequiredConcurrency
	}
	verdict := "fail"
	if headroom >= 0 {
		verdict = "pass"
	}
	reasons := capacityFailureReasons(levels, profile.SLO, qualified, expectedQualifiedPresent)
	if qualifiedPresent != expectedQualifiedPresent ||
		(qualifiedPresent && actualQualified != qualified) ||
		saturationPresent != expectedSaturationPresent ||
		(saturationPresent && actualSaturation != saturation) ||
		*profile.Assessment.SLOHeadroom != headroom ||
		profile.Assessment.Verdict != verdict ||
		!reflect.DeepEqual(profile.Assessment.FailureReasons, reasons) {
		return 0, fmt.Errorf("assessment does not match the server-reduced capacity envelope")
	}
	return headroom, nil
}

func capacityFailureReasons(
	levels []reducedCapacityLevel,
	slo *CapacitySLO,
	qualified int64,
	qualifiedPresent bool,
) []string {
	if qualifiedPresent && qualified >= slo.RequiredConcurrency {
		return []string{}
	}
	var target *reducedCapacityLevel
	for index := range levels {
		if levels[index].concurrency >= slo.RequiredConcurrency {
			target = &levels[index]
			break
		}
	}
	if target == nil {
		return []string{"required_concurrency"}
	}
	reasons := make([]string, 0, 9)
	checks := []struct {
		passed bool
		reason string
	}{
		{target.warmupPassed, "warmup_errors"},
		{target.latencyPassed, "latency_p95"},
		{target.clusterCoverage, "measurement_cluster_coverage"},
		{target.errorRateStable, "error_rate_cluster_stability"},
		{target.errorPassed, "error_rate_upper_bound"},
		{target.throughputPassed, "throughput"},
		{target.scalingPassed, "throughput_scaling"},
		{target.throughputStable, "throughput_stability"},
		{target.latencyStable, "latency_stability"},
	}
	for _, check := range checks {
		if !check.passed {
			reasons = append(reasons, check.reason)
		}
	}
	if len(reasons) == 0 {
		reasons = append(reasons, "required_concurrency")
	}
	return reasons
}

func decodeCapacityOptionalInt(name string, raw json.RawMessage) (int64, bool, error) {
	if len(raw) == 0 {
		return 0, false, fmt.Errorf("%s is required", name)
	}
	trimmed := bytes.TrimSpace(raw)
	if bytes.Equal(trimmed, []byte("null")) {
		return 0, false, nil
	}
	var value int64
	decoder := json.NewDecoder(bytes.NewReader(trimmed))
	if err := decoder.Decode(&value); err != nil {
		return 0, false, fmt.Errorf("%s must be a positive integer or null", name)
	}
	if err := ensureJSONEOF(decoder); err != nil || value <= 0 {
		return 0, false, fmt.Errorf("%s must be a positive integer or null", name)
	}
	return value, true, nil
}
