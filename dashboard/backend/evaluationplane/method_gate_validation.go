package evaluationplane

import "fmt"

// These validators are the gate-facing half of the typed method reducers. A
// worker report can describe the result, but only the server-reduced method
// attestation below can authorize that description.
func validateHardPolicyGate(gate Gate, records recordAttestation) error {
	method := records.Methods.HardPolicy
	if method.StaticPassed == nil {
		return requireUnavailableGate(gate, "a server-brokered hard-policy proof and complete dynamic window")
	}
	if !*method.StaticPassed {
		if gate.Verdict != "fail" || gate.Observed != nil || gate.Threshold != nil {
			return fmt.Errorf("%w: gate %s contradicts the incomplete or invalid static policy proof", ErrInvalid, gate.ID)
		}
		return nil
	}
	if method.DynamicPassed == nil || !completeSafetyGateEvidence(records) {
		return requireUnavailableGate(gate, "complete typed dynamic hard-policy records")
	}
	return validateQualifiedSafetyGate(gate, records)
}

func validateRobustnessGate(gate Gate, records recordAttestation) error {
	method := records.Methods.Robustness
	if method.Passed == nil || !method.SourceQualified {
		return requireUnavailableGate(gate, "a complete server-brokered run of one exact pinned declared-shift relation set")
	}
	observed := 0.0
	verdict := GateVerdict("fail")
	if *method.Passed {
		observed = 1
		verdict = "pass"
	}
	return requireGateDecision(gate, verdict, observed, GateThreshold{
		Operator: ">=", Value: 1, Unit: "boolean",
	})
}

func validateRecoveryGate(gate Gate, records recordAttestation) error {
	method := records.Methods.Recovery
	if method.Passed == nil || method.ClusterPassRateLower95 == nil {
		return requireUnavailableGate(gate, "a complete server-brokered live exact-step fault window with the platform minimum independent clusters")
	}
	verdict := GateVerdict("fail")
	if *method.Passed {
		verdict = "pass"
	}
	return requireGateDecision(gate, verdict, *method.ClusterPassRateLower95, GateThreshold{
		Operator: ">=", Value: minimumRecoveryPassRateLowerBound, Unit: "fraction",
	})
}

func validateProductionControlsGate(gate Gate, records recordAttestation) error {
	method := records.Methods.Production
	if method.CandidateSafe == nil || method.RiskEventUpper95 == nil || method.RiskBudgetMaxRate == nil {
		return requireUnavailableGate(gate, "a complete sealed production assignment window and frozen operational controls")
	}
	verdict := GateVerdict("fail")
	if *method.CandidateSafe {
		verdict = "pass"
	}
	return requireGateDecision(gate, verdict, *method.RiskEventUpper95, GateThreshold{
		Operator: "<=", Value: *method.RiskBudgetMaxRate, Unit: "fraction",
	})
}

func validateProductionPreferenceGate(gate Gate, records recordAttestation) error {
	method := records.Methods.Production
	if method.PreferencePassed == nil {
		return requireUnavailableGate(gate, "a complete production preference outcome window")
	}
	threshold := GateThreshold{Operator: ">=", Value: method.MinimumRewardLift, Unit: "reward lift"}
	if !method.CausalEligible || method.RewardLiftLower95 == nil {
		if *method.PreferencePassed || gate.Verdict != "fail" || gate.Observed != nil || gate.Threshold == nil || *gate.Threshold != threshold {
			return fmt.Errorf("%w: gate %s makes a causal reward claim without eligible evidence", ErrInvalid, gate.ID)
		}
		return nil
	}
	verdict := GateVerdict("fail")
	if *method.PreferencePassed {
		verdict = "pass"
	}
	return requireGateDecision(gate, verdict, *method.RewardLiftLower95, threshold)
}
