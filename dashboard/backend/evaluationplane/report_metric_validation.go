package evaluationplane

import (
	"fmt"
	"strings"
)

func validateReportMetrics(metrics []Metric, selectedTrackIDs []TrackID) error {
	selectedTracks := make(map[TrackID]struct{}, len(selectedTrackIDs))
	for _, trackID := range selectedTrackIDs {
		selectedTracks[trackID] = struct{}{}
	}
	metricIDs := make(map[string]struct{}, len(metrics))
	for _, metric := range metrics {
		if strings.TrimSpace(metric.ID) == "" {
			return fmt.Errorf("evaluation report contains a blank metric id")
		}
		if _, duplicate := metricIDs[metric.ID]; duplicate {
			return fmt.Errorf("evaluation report contains duplicate metric id %q", metric.ID)
		}
		metricIDs[metric.ID] = struct{}{}
		if strings.TrimSpace(metric.Name) == "" {
			return fmt.Errorf("evaluation metric %q has a blank name", metric.ID)
		}
		if strings.TrimSpace(metric.Unit) == "" {
			return fmt.Errorf("evaluation metric %q has a blank unit", metric.ID)
		}
		if metric.TrackID != "" {
			if _, selected := selectedTracks[metric.TrackID]; !selected {
				return fmt.Errorf("evaluation metric %q track_id %q is not selected by the run", metric.ID, metric.TrackID)
			}
		}
		if !validMetricDirection(metric.Direction) {
			return fmt.Errorf("evaluation metric %q has invalid direction", metric.ID)
		}
		if metric.SampleCount < 0 {
			return fmt.Errorf("evaluation metric %q sample_count cannot be negative", metric.ID)
		}
		for _, value := range []struct {
			name  string
			value *float64
		}{
			{name: "value", value: metric.Value},
			{name: "baseline_value", value: metric.BaselineValue},
			{name: "delta", value: metric.Delta},
		} {
			if value.value != nil && !finiteFloat(*value.value) {
				return fmt.Errorf("evaluation metric %q %s must be finite", metric.ID, value.name)
			}
		}
		if metric.ConfidenceInterval != nil {
			if len(metric.ConfidenceInterval) != 2 {
				return fmt.Errorf("evaluation metric %q confidence_interval must contain exactly two bounds", metric.ID)
			}
			lower, upper := metric.ConfidenceInterval[0], metric.ConfidenceInterval[1]
			if !finiteFloat(lower) || !finiteFloat(upper) {
				return fmt.Errorf("evaluation metric %q confidence_interval bounds must be finite", metric.ID)
			}
			if lower > upper {
				return fmt.Errorf("evaluation metric %q confidence_interval bounds are reversed", metric.ID)
			}
			if metric.Value == nil || metric.SampleCount == 0 {
				return fmt.Errorf("evaluation metric %q confidence_interval requires an estimate and samples", metric.ID)
			}
		}
		if (metric.BaselineValue == nil) != (metric.Delta == nil) {
			return fmt.Errorf("evaluation metric %q baseline_value and delta must be published together", metric.ID)
		}
		if metric.BaselineValue != nil {
			if metric.Value == nil {
				return fmt.Errorf("evaluation metric %q comparison requires a candidate value", metric.ID)
			}
			if !reducedFloatsEqual(*metric.Delta, *metric.Value-*metric.BaselineValue) &&
				!reducedFloatsEqual(*metric.Value, *metric.BaselineValue+*metric.Delta) {
				return fmt.Errorf("evaluation metric %q delta does not match value minus baseline_value", metric.ID)
			}
		}
		if err := validateMetricAnalysisProvenance(metric.ID, metric.AnalysisProvenance); err != nil {
			return err
		}
	}
	return nil
}

type metricAnalysisSpec struct {
	estimatorID, estimatorVersion, analysisUnit, clusterUnit string
	weighting, missingness, exclusionPolicy                  string
}

func registeredMetricAnalysisSpec(metricID string) (metricAnalysisSpec, error) {
	match, err := ResolveMetricAnalysisCatalog(metricID)
	if err != nil {
		return metricAnalysisSpec{}, err
	}
	specification := match.Specification
	return metricAnalysisSpec{
		estimatorID: specification.EstimatorID, estimatorVersion: specification.EstimatorVersion,
		analysisUnit: specification.AnalysisUnit, clusterUnit: specification.ClusterUnit,
		weighting: specification.Weighting, missingness: specification.Missingness,
		exclusionPolicy: specification.ExclusionPolicy,
	}, nil
}

func validateMetricAnalysisProvenance(metricID string, provenance MetricAnalysisProvenance) error {
	expected, err := registeredMetricAnalysisSpec(metricID)
	if err != nil {
		return fmt.Errorf("evaluation metric %q analysis_provenance metric id is not registered: %w", metricID, err)
	}
	if provenance.ContractVersion != MetricAnalysisContractVersion {
		return fmt.Errorf("evaluation metric %q analysis_provenance contract_version is invalid", metricID)
	}
	if provenance.EstimatorVersion != expected.estimatorVersion || provenance.EstimatorID != expected.estimatorID ||
		provenance.AnalysisUnit != expected.analysisUnit || provenance.ClusterUnit != expected.clusterUnit ||
		provenance.Weighting != expected.weighting || provenance.Missingness != expected.missingness ||
		provenance.ExclusionPolicy != expected.exclusionPolicy {
		return fmt.Errorf("evaluation metric %q analysis_provenance does not match its registered estimator", metricID)
	}
	if provenance.ObservedExclusions == nil || *provenance.ObservedExclusions < 0 {
		return fmt.Errorf("evaluation metric %q analysis_provenance observed_exclusions is required and non-negative", metricID)
	}
	return nil
}
