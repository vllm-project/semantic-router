package evaluationplane

import (
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestVerdictDomainsRejectWaivedAndSeparateGateApplicability(t *testing.T) {
	for _, verdict := range []GateVerdict{
		GateVerdictPass, GateVerdictFail, GateVerdictUnavailable, GateVerdictNotApplicable,
	} {
		if !validGateVerdict(verdict) {
			t.Errorf("canonical gate verdict %q was rejected", verdict)
		}
	}
	for _, verdict := range []DecisionVerdict{
		DecisionVerdictPass, DecisionVerdictFail, DecisionVerdictUnavailable,
	} {
		if !validDecisionVerdict(verdict) {
			t.Errorf("canonical decision verdict %q was rejected", verdict)
		}
	}
	if validGateVerdict("waived") || validDecisionVerdict("waived") ||
		validDecisionVerdict("not_applicable") {
		t.Fatal("waived or gate-only applicability entered a verdict domain")
	}

	gate := Gate{
		ID: "G0", Disposition: GateDispositionRequired, Verdict: "waived",
		ChangeProfile: "schema_adapter", ContractVersion: GateContractVersion,
		EvidenceRefs: []string{"provenance.json"},
	}
	if err := validateReportGate(gate, gate.ChangeProfile); err == nil || !strings.Contains(err.Error(), "verdict") {
		t.Fatalf("waived gate error=%v, want verdict rejection", err)
	}
}

func diagnosticGateReport() Report {
	run := Run{EvidenceLevel: "E0", ChangeProfile: "schema_adapter"}
	gates := testReleaseGates(run.ChangeProfile, time.Now().UTC())
	setTestGatePlanCoverage(gates, "routing", 1, 1)
	return Report{
		Run: run,
		Summary: ReportSummary{
			Verdict: "unavailable", PassedGates: 2, UnavailableGates: 5,
		},
		Gates: gates,
	}
}

func diagnosticRecordAttestation() recordAttestation {
	return recordAttestation{
		validated: true, Total: 1, Succeeded: 1,
		ByTrack: map[TrackID]recordStatusCounts{"routing": {Succeeded: 1}},
		PlannedCaseIDsByTrack: map[TrackID]map[string]struct{}{
			"routing": {"case-1": {}},
		},
		EvaluatedCaseIDsByTrack: map[TrackID]map[string]struct{}{
			"routing": {"case-1": {}},
		},
	}
}

func TestServerOwnedGateReducerRejectsForgedPassesAndUnknownThresholds(t *testing.T) {
	valid := diagnosticGateReport()
	records := diagnosticRecordAttestation()
	if err := validateServerOwnedGateSemantics(valid, records, suiteGateQualification{}, nil); err != nil {
		t.Fatalf("canonical diagnostic gates rejected: %v", err)
	}

	t.Run("missing records attestation", func(t *testing.T) {
		if err := validateServerOwnedGateSemantics(diagnosticGateReport(), recordAttestation{}, suiteGateQualification{}, nil); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "records attestation") {
			t.Fatalf("missing records attestation error=%v, want ErrInvalid", err)
		}
	})

	t.Run("forged records coverage", func(t *testing.T) {
		forged := diagnosticGateReport()
		forged.Gates[0].Coverage.Total = 2
		if err := validateServerOwnedGateSemantics(forged, records, suiteGateQualification{}, nil); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "records attestation") {
			t.Fatalf("forged records coverage error=%v, want ErrInvalid", err)
		}
	})

	t.Run("unknown operator", func(t *testing.T) {
		forged := diagnosticGateReport()
		forged.Gates[0].Threshold.Operator = "approximately"
		if err := validateServerOwnedGateSemantics(forged, records, suiteGateQualification{}, nil); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "threshold") {
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
		if err := validateServerOwnedGateSemantics(forged, records, suiteGateQualification{}, nil); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "contradicts") {
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
		if err := validateServerOwnedGateSemantics(forged, records, suiteGateQualification{}, nil); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "does not match metric") {
			t.Fatalf("forged observed error=%v, want metric mismatch ErrInvalid", err)
		}
	})

	t.Run("E0 promotion pass", func(t *testing.T) {
		forged := diagnosticGateReport()
		forged.Summary.Verdict = "pass"
		if err := validateServerOwnedGateSemantics(forged, records, suiteGateQualification{}, nil); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "E0") {
			t.Fatalf("E0 pass error=%v, want E0 ErrInvalid", err)
		}
	})

	t.Run("unattested promotion gate", func(t *testing.T) {
		forged := diagnosticGateReport()
		observed := 1.0
		forged.Gates[4].Verdict = "pass"
		forged.Gates[4].Observed = &observed
		forged.Gates[4].Threshold = &GateThreshold{Operator: ">=", Value: 1, Unit: "boolean"}
		if err := validateServerOwnedGateSemantics(forged, records, suiteGateQualification{}, nil); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "common installed-suite") {
			t.Fatalf("unattested pass error=%v, want attestation ErrInvalid", err)
		}
	})
}

func TestServerOwnedGateReducerRejectsEveryUnqualifiedG2ThroughG9Verdict(t *testing.T) {
	thresholds := map[int]GateThreshold{
		2: {Operator: "<=", Value: 0, Unit: "violations/case"},
		3: {Operator: "<=", Value: defaultNormalizedRegretMaximum, Unit: "fraction"},
		4: {Operator: ">=", Value: 1, Unit: "boolean"},
		5: {Operator: ">=", Value: 1, Unit: "boolean"},
		6: {Operator: ">=", Value: minimumRecoveryPassRateLowerBound, Unit: "fraction"},
		7: {Operator: ">=", Value: 0, Unit: "concurrency"},
		8: {Operator: "<=", Value: maximumProductionRiskBudgetRate, Unit: "fraction"},
		9: {Operator: ">=", Value: minimumProductionRewardLift, Unit: "reward lift"},
	}
	for index := 2; index < 10; index++ {
		for _, verdict := range []GateVerdict{"pass", "fail"} {
			t.Run(reportGateTestName(index, verdict), func(t *testing.T) {
				report := diagnosticGateReport()
				gate := &report.Gates[index]
				gate.Disposition = "advisory"
				gate.Verdict = verdict
				threshold := thresholds[index]
				gate.Threshold = &threshold
				observed := threshold.Value
				if verdict == "fail" {
					if threshold.Operator == ">=" {
						observed = threshold.Value - 0.5
					} else {
						observed = threshold.Value + 0.5
					}
				}
				gate.Observed = &observed
				switch index {
				case 2:
					report.Metrics = []Metric{{ID: "safety.violation_rate", Value: &observed}}
				case 3:
					report.Metrics = []Metric{{ID: "joint.normalized_regret", Value: &observed}}
				case 6:
					report.Metrics = []Metric{{ID: "agentic.recovery_cluster_pass_rate_lower_95", Value: &observed}}
				case 7:
					report.Metrics = []Metric{{ID: "capacity.slo_headroom", Value: &observed}}
				case 8:
					report.Metrics = []Metric{{ID: "experiment.risk_event_upper_confidence_bound", Value: &observed}}
				case 9:
					report.Metrics = []Metric{{ID: "preference.online_reward_lift", Value: &observed, ConfidenceInterval: []float64{observed, observed}}}
				}
				err := validateServerOwnedGateSemantics(report, diagnosticRecordAttestation(), suiteGateQualification{}, nil)
				expected := "common installed-suite"
				switch index {
				case 2:
					expected = "hard-policy proof"
				case 6:
					expected = "fault window"
				case 7:
					expected = "frozen live capacity SLO"
				case 8:
					expected = "production assignment window"
				case 9:
					expected = "production preference outcome window"
				}
				if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), expected) {
					t.Fatalf("gate %s verdict %s error=%v, want qualification ErrInvalid", gate.ID, verdict, err)
				}
			})
		}
	}
}

func TestServerOwnedGateReducerRejectsObservationOnUnavailableGate(t *testing.T) {
	report := diagnosticGateReport()
	observed := 0.1
	report.Gates[3].Observed = &observed
	report.Metrics = []Metric{{ID: "joint.normalized_regret", Value: &observed}}
	err := validateServerOwnedGateSemantics(report, diagnosticRecordAttestation(), suiteGateQualification{}, nil)
	if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "common installed-suite") {
		t.Fatalf("unqualified observation error=%v, want qualification ErrInvalid", err)
	}
}

func TestServerOwnedGateReducerRejectsWorkerOwnedGateCoverageInterval(t *testing.T) {
	report := diagnosticGateReport()
	report.Gates[0].Coverage.ConfidenceLevel = 0.95
	report.Gates[0].Coverage.ConfidenceInterval = []float64{0.99, 1}
	err := validateServerOwnedGateSemantics(report, diagnosticRecordAttestation(), suiteGateQualification{}, nil)
	if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "records attestation") {
		t.Fatalf("worker-owned gate interval error=%v, want coverage ErrInvalid", err)
	}
}

func TestServerOwnedGateReducerRejectsHundredCasePlanReportedAsOneOfOne(t *testing.T) {
	report := diagnosticGateReport()
	records := diagnosticRecordAttestation()
	records.PlannedCaseIDsByTrack["routing"] = make(map[string]struct{}, 100)
	for index := range 100 {
		records.PlannedCaseIDsByTrack["routing"][fmt.Sprintf("case-%d", index)] = struct{}{}
	}
	err := validateServerOwnedGateSemantics(report, records, suiteGateQualification{}, nil)
	if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "records attestation") {
		t.Fatalf("omitted plan cells error=%v, want records coverage ErrInvalid", err)
	}
}

func TestLiveHardPolicyGateUsesOnlyServerReducedProofAndRecords(t *testing.T) {
	for _, test := range []struct {
		name          string
		violationRate float64
		verdict       GateVerdict
	}{
		{name: "pass", violationRate: 0, verdict: "pass"},
		{name: "fail", violationRate: 0.25, verdict: "fail"},
	} {
		t.Run(test.name, func(t *testing.T) {
			blockAccuracy := 1.0
			report, records := qualifiedMetricGateFixture("G2", "safety", test.verdict, test.violationRate, GateThreshold{
				Operator: "<=", Value: 0, Unit: "violations/case",
			})
			report.Metrics = []Metric{
				{ID: "safety.violation_rate", Value: &test.violationRate},
				{ID: "safety.block_accuracy", Value: &blockAccuracy},
			}
			records.Metrics = recordMetricAttestation{
				SafetyViolationRate:   reducedMetricEvidence{Value: &test.violationRate, SampleCount: 1},
				SafetyBlockAccuracy:   reducedMetricEvidence{Value: &blockAccuracy, SampleCount: 1},
				SafetyTypedRowsByCase: map[string]int{"case-1": 1},
			}
			staticPassed := true
			dynamicPassed := test.verdict == "pass"
			records.Methods.HardPolicy = hardPolicyMethodAttestation{
				ObservationCount: 1, TotalObservationCount: 1, StaticPassed: &staticPassed, DynamicPassed: &dynamicPassed,
			}
			if err := validateServerOwnedGateSemantics(report, records, suiteGateQualification{}, nil); err != nil {
				t.Fatalf("qualified G2 %s rejected: %v", test.verdict, err)
			}

			forged := report
			forged.Gates = append([]Gate(nil), report.Gates...)
			forgedValue := 1 - test.violationRate
			forged.Gates[2].Observed = &forgedValue
			if err := validateServerOwnedGateSemantics(forged, records, suiteGateQualification{}, nil); !errors.Is(err, ErrInvalid) {
				t.Fatalf("forged G2 observation error=%v", err)
			}
		})
	}

	t.Run("block accuracy fail", func(t *testing.T) {
		violationRate, blockAccuracy := 0.0, 0.5
		report, records := qualifiedMetricGateFixture("G2", "safety", "fail", blockAccuracy, GateThreshold{
			Operator: ">=", Value: 1, Unit: "fraction",
		})
		report.Metrics = []Metric{
			{ID: "safety.violation_rate", Value: &violationRate},
			{ID: "safety.block_accuracy", Value: &blockAccuracy},
		}
		records.Metrics = recordMetricAttestation{
			SafetyViolationRate:   reducedMetricEvidence{Value: &violationRate, SampleCount: 1},
			SafetyBlockAccuracy:   reducedMetricEvidence{Value: &blockAccuracy, SampleCount: 1},
			SafetyTypedRowsByCase: map[string]int{"case-1": 1},
		}
		staticPassed, dynamicPassed := true, false
		records.Methods.HardPolicy = hardPolicyMethodAttestation{
			ObservationCount: 1, TotalObservationCount: 1, StaticPassed: &staticPassed, DynamicPassed: &dynamicPassed,
		}
		if err := validateServerOwnedGateSemantics(report, records, suiteGateQualification{}, nil); err != nil {
			t.Fatalf("qualified block-accuracy failure rejected: %v", err)
		}
	})

	t.Run("receipt without typed coverage cannot pass", func(t *testing.T) {
		violationRate, blockAccuracy := 0.0, 1.0
		report, records := qualifiedMetricGateFixture("G2", "safety", "pass", violationRate, GateThreshold{
			Operator: "<=", Value: 0, Unit: "violations/case",
		})
		report.Metrics = []Metric{
			{ID: "safety.violation_rate", Value: &violationRate},
			{ID: "safety.block_accuracy", Value: &blockAccuracy},
		}
		records.Metrics = recordMetricAttestation{
			SafetyViolationRate: reducedMetricEvidence{Value: &violationRate, SampleCount: 1},
			SafetyBlockAccuracy: reducedMetricEvidence{Value: &blockAccuracy, SampleCount: 1},
		}
		staticPassed, dynamicPassed := true, true
		records.Methods.HardPolicy = hardPolicyMethodAttestation{
			ObservationCount: 1, TotalObservationCount: 1, StaticPassed: &staticPassed, DynamicPassed: &dynamicPassed,
		}
		err := validateServerOwnedGateSemantics(report, records, suiteGateQualification{}, nil)
		if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "typed dynamic hard-policy records") {
			t.Fatalf("untyped G2 pass error=%v", err)
		}
	})
}

func TestCapacityGateUsesOnlyServerReducedSLOHeadroom(t *testing.T) {
	for _, test := range []struct {
		name     string
		headroom float64
		verdict  GateVerdict
	}{
		{name: "pass", headroom: 2, verdict: "pass"},
		{name: "fail", headroom: -1, verdict: "fail"},
	} {
		t.Run(test.name, func(t *testing.T) {
			report, records := qualifiedMetricGateFixture("G7", "capacity", test.verdict, test.headroom, GateThreshold{
				Operator: ">=", Value: 0, Unit: "concurrency",
			})
			report.Metrics = []Metric{{ID: "capacity.slo_headroom", Value: &test.headroom}}
			records.ByTrack["capacity"] = recordStatusCounts{Succeeded: 2}
			records.Metrics = recordMetricAttestation{
				CapacityRowsByCase:   map[string]int{"case-1": 2},
				CapacityLevelsByCase: map[string]map[int64]struct{}{"case-1": {1: {}, 2: {}}},
			}
			attestation := &capacitySLOAttestation{
				Headroom: test.headroom, LevelCount: 2,
				MeasurementClusterCount: 6, MinimumClustersPerLevel: 3, RequiredClustersPerLevel: 3,
				WorstErrorRateUpperBound: 0.02, ReleaseErrorRateUpperBound: 0.02,
				ReleaseErrorRateClusterRange: 0, MaxErrorRate: 0.05, MaxErrorRateClusterRange: 0.05,
			}
			if err := validateServerOwnedGateSemantics(report, records, suiteGateQualification{}, attestation); err != nil {
				t.Fatalf("qualified G7 %s rejected: %v", test.verdict, err)
			}

			// A load level may schedule more than one request for the same
			// benchmark case. Completeness is defined by the measured level
			// set, not by equating request rows with distinct concurrency.
			records.ByTrack["capacity"] = recordStatusCounts{Succeeded: 3}
			records.Metrics.CapacityRowsByCase["case-1"] = 3
			if err := validateServerOwnedGateSemantics(report, records, suiteGateQualification{}, attestation); err != nil {
				t.Fatalf("qualified G7 %s rejected duplicate case observations: %v", test.verdict, err)
			}
			beyondReleaseSaturation := *attestation
			beyondReleaseSaturation.WorstErrorRateUpperBound = 0.5
			beyondReleaseSaturation.WorstErrorRateClusterRange = 0.2
			if err := validateServerOwnedGateSemantics(report, records, suiteGateQualification{}, &beyondReleaseSaturation); err != nil {
				t.Fatalf("qualified G7 %s rejected saturation beyond the required envelope: %v", test.verdict, err)
			}

			booleanClaim := report
			booleanClaim.Gates = append([]Gate(nil), report.Gates...)
			booleanClaim.Gates[7].Threshold = &GateThreshold{Operator: ">=", Value: 1, Unit: "boolean"}
			if err := validateServerOwnedGateSemantics(booleanClaim, records, suiteGateQualification{}, attestation); !errors.Is(err, ErrInvalid) {
				t.Fatalf("boolean G7 claim error=%v", err)
			}
		})
	}
}

func TestQualifiedReceiptCannotReplaceMissingTypedGateReducers(t *testing.T) {
	for _, test := range []struct {
		gateID    string
		trackID   TrackID
		index     int
		observed  float64
		threshold GateThreshold
		metricID  string
		missing   string
	}{
		{gateID: "G4", trackID: "routing", index: 4, observed: 1, threshold: GateThreshold{Operator: ">=", Value: 1, Unit: "boolean"}, missing: "complete server-brokered run"},
		{gateID: "G6", trackID: "agentic", index: 6, observed: 1, threshold: GateThreshold{Operator: ">=", Value: minimumRecoveryPassRateLowerBound, Unit: "fraction"}, metricID: "agentic.recovery_cluster_pass_rate_lower_95", missing: "fault window"},
		{gateID: "G9", trackID: "preference", index: 9, observed: 0.1, threshold: GateThreshold{Operator: ">=", Value: minimumProductionRewardLift, Unit: "reward lift"}, metricID: "preference.online_reward_lift", missing: "production preference outcome window"},
	} {
		t.Run(test.gateID, func(t *testing.T) {
			report, records := qualifiedMetricGateFixture(test.gateID, test.trackID, "pass", test.observed, test.threshold)
			if test.metricID != "" {
				metric := Metric{ID: test.metricID, Value: &test.observed}
				if test.gateID == "G9" {
					metric.ConfidenceInterval = []float64{test.observed, test.observed}
				}
				report.Metrics = []Metric{metric}
			}
			err := validateServerOwnedGateSemantics(report, records, qualifiedGateSet(test.gateID), nil)
			if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), test.missing) {
				t.Fatalf("gate %s overclaim error=%v", test.gateID, err)
			}
			report.Gates[test.index].Verdict = "unavailable"
			report.Gates[test.index].Observed = nil
			report.Gates[test.index].Threshold = nil
			if err := validateServerOwnedGateSemantics(report, records, qualifiedGateSet(test.gateID), nil); err != nil {
				t.Fatalf("gate %s unavailable evidence rejected: %v", test.gateID, err)
			}
		})
	}
}

func qualifiedMetricGateFixture(gateID string, trackID TrackID, verdict GateVerdict, observed float64, threshold GateThreshold) (Report, recordAttestation) {
	report := diagnosticGateReport()
	report.Run.EvidenceLevel = "E5"
	report.Run.TrackIDs = []TrackID{trackID}
	setTestGatePlanCoverage(report.Gates, trackID, 1, 1)
	index := int(gateID[1] - '0')
	report.Gates[index].Disposition = "advisory"
	report.Gates[index].Verdict = verdict
	report.Gates[index].Observed = &observed
	report.Gates[index].Threshold = &threshold
	records := recordAttestation{
		validated: true,
		ByTrack:   map[TrackID]recordStatusCounts{trackID: {Succeeded: 1}},
		PlannedCaseIDsByTrack: map[TrackID]map[string]struct{}{
			trackID: {"case-1": {}},
		},
		EvaluatedCaseIDsByTrack: map[TrackID]map[string]struct{}{
			trackID: {"case-1": {}},
		},
	}
	return report, records
}

func qualifiedGateSet(gateIDs ...string) suiteGateQualification {
	qualification := suiteGateQualification{normalizedSuiteRun: true, commonGateIDs: make(map[string]struct{}, len(gateIDs))}
	for _, gateID := range gateIDs {
		qualification.commonGateIDs[gateID] = struct{}{}
	}
	return qualification
}

func reportGateTestName(index int, verdict GateVerdict) string {
	return fmt.Sprintf("G%d-%s", index, verdict)
}
