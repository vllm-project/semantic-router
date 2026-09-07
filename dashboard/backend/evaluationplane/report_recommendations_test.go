package evaluationplane

import (
	"math"
	"strings"
	"testing"
)

func TestServerReportRecommendationsUseOnlyQualifiedVerifiedEvidence(t *testing.T) {
	fallback := 0.25
	report := Report{
		Metrics: []Metric{{ID: "routing.fallback_rate", TrackID: "routing", Value: &fallback}},
		Gates:   []Gate{{ID: "G4", Name: "Workload-shift robustness", Owner: "evaluation-workload", Verdict: "pass"}},
	}
	qualified := mustServerReportRecommendations(t, report, sealedEvidenceLevels{
		Run: "E4", ByTrack: map[TrackID]EvidenceLevel{"routing": "E4"},
	})
	if len(qualified) != 1 || !strings.Contains(qualified[0], "observed value=0.250") {
		t.Fatalf("qualified recommendations = %#v", qualified)
	}

	diagnostic := mustServerReportRecommendations(t, report, sealedEvidenceLevels{
		Run: "E0", ByTrack: map[TrackID]EvidenceLevel{"routing": "E0"},
	})
	if len(diagnostic) != 1 || !strings.Contains(diagnostic[0], "promotion-grade evidence is not yet available") ||
		strings.Contains(diagnostic[0], "observed value=0.250") {
		t.Fatalf("diagnostic recommendations = %#v", diagnostic)
	}
}

func TestServerReportRecommendationsSeparateOverallAndSliceModalityFindings(t *testing.T) {
	overall, image := 0.5, 0.25
	recommendations := mustServerReportRecommendations(t, Report{Metrics: []Metric{
		{ID: "multimodal.support_rate", TrackID: "multimodal", Value: &overall},
		{ID: "multimodal.image.support_rate", TrackID: "multimodal", Value: &image},
	}}, sealedEvidenceLevels{
		Run: "E4", ByTrack: map[TrackID]EvidenceLevel{"multimodal": "E4"},
	})
	joined := strings.Join(recommendations, "\n")
	if !strings.Contains(joined, "observed value=0.500") ||
		!strings.Contains(joined, "image support=0.250") ||
		strings.Contains(joined, "[AF-") {
		t.Fatalf("modality recommendations = %#v", recommendations)
	}
}

func TestReportMetricRecommendationRuleInventoryIsValidAndExplicit(t *testing.T) {
	if err := validateReportMetricRecommendationRules(reportMetricRecommendationRules); err != nil {
		t.Fatalf("built-in report metric recommendation rules are invalid: %v", err)
	}
	capacityFound := false
	for _, rule := range reportMetricRecommendationRules {
		if rule.metricID != "capacity.saturation_concurrency" {
			continue
		}
		capacityFound = true
		if rule.comparator != reportMetricRecommendationAlways ||
			!finiteFloat(rule.threshold) ||
			!rule.comparator.compare(0, rule.threshold) {
			t.Fatalf("capacity recommendation does not use an explicit always comparator: %+v", rule)
		}
	}
	if !capacityFound {
		t.Fatal("capacity recommendation rule is missing")
	}
}

func TestReportMetricRecommendationComparatorsHaveExactBoundaries(t *testing.T) {
	tests := []struct {
		name       string
		comparator reportMetricRecommendationComparator
		value      float64
		threshold  float64
		want       bool
	}{
		{name: "less below", comparator: reportMetricRecommendationLessThan, value: 0.9, threshold: 1, want: true},
		{name: "less equal", comparator: reportMetricRecommendationLessThan, value: 1, threshold: 1},
		{name: "less or equal", comparator: reportMetricRecommendationLessOrEqual, value: 1, threshold: 1, want: true},
		{name: "greater above", comparator: reportMetricRecommendationGreaterThan, value: 1.1, threshold: 1, want: true},
		{name: "greater equal", comparator: reportMetricRecommendationGreaterThan, value: 1, threshold: 1},
		{name: "always", comparator: reportMetricRecommendationAlways, value: -1, threshold: 1, want: true},
		{name: "unknown", comparator: reportMetricRecommendationComparator("unknown"), value: 1, threshold: 1},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.comparator.compare(test.value, test.threshold); got != test.want {
				t.Fatalf("compare(%v, %v)=%t, want %t", test.value, test.threshold, got, test.want)
			}
		})
	}
}

func TestReportMetricRecommendationRuleValidationFailsClosed(t *testing.T) {
	tests := []struct {
		name   string
		mutate func([]reportMetricRecommendationRule) []reportMetricRecommendationRule
		match  string
	}{
		{
			name: "empty inventory",
			mutate: func([]reportMetricRecommendationRule) []reportMetricRecommendationRule {
				return nil
			},
			match: "empty",
		},
		{
			name: "duplicate rule id",
			mutate: func(rules []reportMetricRecommendationRule) []reportMetricRecommendationRule {
				rules[1].id = rules[0].id
				return rules
			},
			match: "rule id",
		},
		{
			name: "duplicate metric id",
			mutate: func(rules []reportMetricRecommendationRule) []reportMetricRecommendationRule {
				rules[1].metricID = rules[0].metricID
				return rules
			},
			match: "metric id",
		},
		{
			name: "unknown metric id",
			mutate: func(rules []reportMetricRecommendationRule) []reportMetricRecommendationRule {
				rules[0].metricID = "unknown.metric"
				return rules
			},
			match: "unknown evaluation metric id",
		},
		{
			name: "unknown comparator",
			mutate: func(rules []reportMetricRecommendationRule) []reportMetricRecommendationRule {
				rules[0].comparator = reportMetricRecommendationComparator("unknown")
				return rules
			},
			match: "comparator",
		},
		{
			name: "non-finite threshold",
			mutate: func(rules []reportMetricRecommendationRule) []reportMetricRecommendationRule {
				rules[0].threshold = math.Inf(1)
				return rules
			},
			match: "non-finite threshold",
		},
		{
			name: "missing owner",
			mutate: func(rules []reportMetricRecommendationRule) []reportMetricRecommendationRule {
				rules[0].owner = ""
				return rules
			},
			match: "ownership or action metadata",
		},
		{
			name: "missing surface",
			mutate: func(rules []reportMetricRecommendationRule) []reportMetricRecommendationRule {
				rules[0].surface = ""
				return rules
			},
			match: "ownership or action metadata",
		},
		{
			name: "missing action",
			mutate: func(rules []reportMetricRecommendationRule) []reportMetricRecommendationRule {
				rules[0].action = ""
				return rules
			},
			match: "ownership or action metadata",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			rules := append([]reportMetricRecommendationRule(nil), reportMetricRecommendationRules...)
			err := validateReportMetricRecommendationRules(test.mutate(rules))
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("rule validation error=%v, want %q", err, test.match)
			}
		})
	}
}

func TestServerReportRecommendationsFailClosedBeforePublication(t *testing.T) {
	rules := append([]reportMetricRecommendationRule(nil), reportMetricRecommendationRules...)
	rules[0].metricID = "unknown.metric"
	if _, err := serverReportRecommendationsWithRules(
		Report{},
		sealedEvidenceLevels{},
		rules,
	); err == nil {
		t.Fatal("report recommendation derivation accepted an invalid built-in rule")
	}
}

func mustServerReportRecommendations(
	t *testing.T,
	report Report,
	sealedLevels sealedEvidenceLevels,
) []string {
	t.Helper()
	recommendations, err := serverReportRecommendations(report, sealedLevels)
	if err != nil {
		t.Fatalf("derive server report recommendations: %v", err)
	}
	return recommendations
}
