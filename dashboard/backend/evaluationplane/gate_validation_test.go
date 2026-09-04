package evaluationplane

import (
	"errors"
	"strings"
	"testing"
	"time"
)

func diagnosticGateReport() Report {
	run := Run{EvidenceLevel: "E0", ChangeProfile: "schema_adapter"}
	return Report{
		Run: run,
		Summary: ReportSummary{
			Verdict: "unavailable", PassedGates: 2, UnavailableGates: 5,
		},
		Gates: testReleaseGates(run.ChangeProfile, time.Now().UTC()),
	}
}

func diagnosticRecordAttestation() recordAttestation {
	return recordAttestation{
		validated: true, Total: 1, Succeeded: 1,
		ByTrack: map[TrackID]recordStatusCounts{"routing": {Succeeded: 1}},
	}
}

func TestServerOwnedGateReducerRejectsForgedPassesAndUnknownThresholds(t *testing.T) {
	valid := diagnosticGateReport()
	records := diagnosticRecordAttestation()
	if err := validateServerOwnedGateSemantics(valid, records); err != nil {
		t.Fatalf("canonical diagnostic gates rejected: %v", err)
	}

	t.Run("missing records attestation", func(t *testing.T) {
		if err := validateServerOwnedGateSemantics(diagnosticGateReport(), recordAttestation{}); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "records attestation") {
			t.Fatalf("missing records attestation error=%v, want ErrInvalid", err)
		}
	})

	t.Run("forged records coverage", func(t *testing.T) {
		forged := diagnosticGateReport()
		forged.Gates[0].Coverage.Total = 2
		if err := validateServerOwnedGateSemantics(forged, records); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "records attestation") {
			t.Fatalf("forged records coverage error=%v, want ErrInvalid", err)
		}
	})

	t.Run("unknown operator", func(t *testing.T) {
		forged := diagnosticGateReport()
		forged.Gates[0].Threshold.Operator = "approximately"
		if err := validateServerOwnedGateSemantics(forged, records); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "threshold") {
			t.Fatalf("unknown operator error=%v, want threshold ErrInvalid", err)
		}
	})

	t.Run("low safety metric marked pass", func(t *testing.T) {
		forged := diagnosticGateReport()
		violationRate := 0.5
		forged.Metrics = []Metric{{ID: "safety.violation_rate", Value: &violationRate}}
		forged.Gates[2].Verdict = "pass"
		forged.Gates[2].Observed = &violationRate
		forged.Gates[2].Threshold = &GateThreshold{Operator: "<=", Value: 0, Unit: "violations/case"}
		if err := validateServerOwnedGateSemantics(forged, records); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "contradicts") {
			t.Fatalf("forged low-metric pass error=%v, want contradiction ErrInvalid", err)
		}
	})

	t.Run("observed does not match metric", func(t *testing.T) {
		forged := diagnosticGateReport()
		metricValue, claimed := 0.5, 0.25
		forged.Metrics = []Metric{{ID: "safety.violation_rate", Value: &metricValue}}
		forged.Gates[2].Verdict = "fail"
		forged.Gates[2].Observed = &claimed
		forged.Gates[2].Threshold = &GateThreshold{Operator: "<=", Value: 0, Unit: "violations/case"}
		if err := validateServerOwnedGateSemantics(forged, records); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "does not match metric") {
			t.Fatalf("forged observed error=%v, want metric mismatch ErrInvalid", err)
		}
	})

	t.Run("E0 promotion pass", func(t *testing.T) {
		forged := diagnosticGateReport()
		forged.Summary.Verdict = "pass"
		if err := validateServerOwnedGateSemantics(forged, records); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "E0") {
			t.Fatalf("E0 pass error=%v, want E0 ErrInvalid", err)
		}
	})

	t.Run("unattested promotion gate", func(t *testing.T) {
		forged := diagnosticGateReport()
		observed := 1.0
		forged.Gates[4].Verdict = "pass"
		forged.Gates[4].Observed = &observed
		forged.Gates[4].Threshold = &GateThreshold{Operator: ">=", Value: 1, Unit: "boolean"}
		if err := validateServerOwnedGateSemantics(forged, records); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "server-owned") {
			t.Fatalf("unattested pass error=%v, want attestation ErrInvalid", err)
		}
	})
}
