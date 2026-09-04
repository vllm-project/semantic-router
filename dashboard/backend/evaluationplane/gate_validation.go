package evaluationplane

import (
	"fmt"
	"math"
)

const (
	defaultNormalizedRegretMaximum = 0.25
	defaultCapacitySuccessMinimum  = 0.95
)

var gateEvidenceLevels = []EvidenceLevel{"E0", "E0", "E3", "E4", "E4", "E5", "E5", "E5", "E5", "E5"}

var gateOwners = []string{
	"evaluation-platform", "evaluation-platform", "router-policy", "recipe-and-model-pool", "evaluation-workload",
	"router-and-serving-runtime", "agent-runtime", "serving-capacity", "release-operations", "online-learning",
}

// validateServerOwnedGateSemantics is the promotion trust boundary. The worker
// may serialize gate-shaped data, but it cannot choose a threshold or turn an
// unqualified observation into a promotion pass.
//
// G0/G1 are attested by the Go bundle validator itself. The current v1 bundle
// has no server-owned attestation for G2-G9 qualifications, so those gates may
// conservatively fail or remain unavailable, but may not pass. A future direct
// arm, paired-statistics, canary, or online-assignment seam must add its typed
// attestation here before it can produce a promotion pass.
func validateServerOwnedGateSemantics(report Report, records recordAttestation) error {
	metrics := make(map[string]Metric, len(report.Metrics))
	for _, metric := range report.Metrics {
		metrics[metric.ID] = metric
	}
	for index, gate := range report.Gates {
		if gate.EvidenceLevel != gateEvidenceLevels[index] || gate.Owner != gateOwners[index] {
			return fmt.Errorf("%w: gate %s evidence level or owner is not canonical", ErrInvalid, gate.ID)
		}
		if err := validateCanonicalGateThreshold(gate); err != nil {
			return err
		}
		if err := validateGateObservedMetric(gate, metrics); err != nil {
			return err
		}
		switch gate.ID {
		case "G0", "G1":
			if !records.validatesGateCoverage(gate) {
				return fmt.Errorf("%w: gate %s lacks the server-owned records attestation", ErrInvalid, gate.ID)
			}
			if gate.Verdict != "pass" {
				return fmt.Errorf("%w: gate %s contradicts the server-validated bundle", ErrInvalid, gate.ID)
			}
		default:
			if gate.Verdict == "pass" {
				return fmt.Errorf("%w: gate %s lacks a server-owned qualified evidence attestation", ErrInvalid, gate.ID)
			}
		}
	}
	if report.Run.EvidenceLevel == "E0" && report.Summary.Verdict == "pass" {
		return fmt.Errorf("%w: E0 diagnostic evidence cannot produce a promotion pass", ErrInvalid)
	}
	return nil
}

func validateCanonicalGateThreshold(gate Gate) error {
	if gate.Disposition == "not_applicable" {
		if gate.Observed != nil || gate.Threshold != nil {
			return fmt.Errorf("%w: not-applicable gate %s cannot publish an observation or threshold", ErrInvalid, gate.ID)
		}
		return nil
	}
	if gate.Observed != nil && !finiteFloat(*gate.Observed) {
		return fmt.Errorf("%w: gate %s observed value is not finite", ErrInvalid, gate.ID)
	}
	if gate.Threshold == nil {
		if gate.Verdict == "pass" {
			return fmt.Errorf("%w: passing gate %s requires a canonical observed threshold", ErrInvalid, gate.ID)
		}
		return nil
	}
	threshold := *gate.Threshold
	if gate.Observed == nil || !finiteFloat(threshold.Value) {
		return fmt.Errorf("%w: gate %s threshold requires finite observed and threshold values", ErrInvalid, gate.ID)
	}
	if !canonicalThresholdForGate(gate.ID, threshold) {
		return fmt.Errorf("%w: gate %s threshold is not part of the server-owned contract", ErrInvalid, gate.ID)
	}
	var met bool
	switch threshold.Operator {
	case ">=":
		met = *gate.Observed >= threshold.Value
	case "<=":
		met = *gate.Observed <= threshold.Value
	default:
		return fmt.Errorf("%w: gate %s threshold operator is unsupported", ErrInvalid, gate.ID)
	}
	expected := GateVerdict("fail")
	if met {
		expected = "pass"
	}
	if gate.Verdict != expected {
		return fmt.Errorf("%w: gate %s verdict contradicts its canonical threshold", ErrInvalid, gate.ID)
	}
	return nil
}

func canonicalThresholdForGate(gateID string, threshold GateThreshold) bool {
	booleanMinimum := threshold.Operator == ">=" && threshold.Value == 1 && threshold.Unit == "boolean"
	switch gateID {
	case "G0":
		return threshold.Operator == ">=" && threshold.Value == 1 && threshold.Unit == "fraction"
	case "G1", "G4", "G5", "G6", "G8", "G9":
		return booleanMinimum
	case "G2":
		return (threshold.Operator == "<=" && threshold.Value == 0 && threshold.Unit == "violations/case") ||
			(threshold.Operator == ">=" && threshold.Value == 1 && threshold.Unit == "fraction")
	case "G3":
		return threshold.Operator == "<=" && threshold.Value == defaultNormalizedRegretMaximum && threshold.Unit == "fraction"
	case "G7":
		return booleanMinimum ||
			(threshold.Operator == ">=" && threshold.Value == defaultCapacitySuccessMinimum && threshold.Unit == "fraction")
	default:
		return false
	}
}

func validateGateObservedMetric(gate Gate, metrics map[string]Metric) error {
	if gate.Observed == nil {
		return nil
	}
	metricID := ""
	switch gate.ID {
	case "G2":
		if gate.Threshold != nil && gate.Threshold.Unit == "violations/case" {
			metricID = "safety.violation_rate"
		} else if gate.Threshold != nil && gate.Threshold.Unit == "fraction" {
			metricID = "safety.block_accuracy"
		}
	case "G3":
		metricID = "joint.normalized_regret"
	case "G7":
		if gate.Threshold != nil && gate.Threshold.Unit == "fraction" {
			metricID = "capacity.success_rate"
		}
	}
	if metricID == "" {
		return nil
	}
	metric, ok := metrics[metricID]
	if !ok || metric.Value == nil || !finiteFloat(*metric.Value) || *metric.Value != *gate.Observed {
		return fmt.Errorf("%w: gate %s observed value does not match metric %s", ErrInvalid, gate.ID, metricID)
	}
	return nil
}

func finiteFloat(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
