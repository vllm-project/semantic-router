package evaluationplane

import "testing"

func TestPromotionSummaryPreservesPrimaryMetricIdentity(t *testing.T) {
	value := 0.82
	report := Report{
		Run: Run{EvidenceLevel: "E2"},
		Metrics: []Metric{{
			ID:                 "routing.accuracy",
			Value:              &value,
			Unit:               "fraction",
			ConfidenceInterval: []float64{0.78, 0.86},
		}},
		Summary: ReportSummary{PrimaryMetric: &ReportPrimaryMetric{
			ID:                 "routing.accuracy",
			Value:              value,
			Unit:               "fraction",
			ConfidenceInterval: []float64{0.78, 0.86},
		}},
	}

	if err := validatePromotionSummary(report); err != nil {
		t.Fatalf("valid primary metric rejected: %v", err)
	}

	report.Summary.PrimaryMetric.ID = "joint.realized_quality"
	if err := validatePromotionSummary(report); err == nil {
		t.Fatal("summary with a forged primary metric identity was accepted")
	}
}

func TestPromotionSummaryRejectsPrimaryMetricAtEvidenceLevelE0(t *testing.T) {
	report := Report{
		Run: Run{EvidenceLevel: "E0"},
		Summary: ReportSummary{PrimaryMetric: &ReportPrimaryMetric{
			ID: "routing.accuracy", Value: 0.82, Unit: "fraction",
		}},
	}

	if err := validatePromotionSummary(report); err == nil {
		t.Fatal("E0 report published a primary metric")
	}
}
