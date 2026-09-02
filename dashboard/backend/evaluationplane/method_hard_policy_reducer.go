package evaluationplane

import (
	"fmt"
	"reflect"
)

type hardPolicyMethodAttestation struct {
	ObservationCount      int
	TotalObservationCount int
	PolicySnapshotDigest  string
	ConfigDigest          string
	TargetID              string
	BackendTopologyDigest string
	MixtureSnapshotDigest string
	StaticPassed          *bool
	DynamicPassed         *bool
}

func reduceHardPolicyMethod(records []executionRecordEvidence) (hardPolicyMethodAttestation, error) {
	var proof *hardPolicyStaticProofEvidence
	observationIDs := make(map[string]struct{})
	attackIDs := make(map[string]struct{})
	decisionIDs := make(map[string]struct{})
	observedBindings := make(map[string]struct{})
	dynamicPassed := true
	count := 0
	for _, record := range records {
		method := record.HardPolicy
		if method == nil {
			continue
		}
		if proof == nil {
			copyProof := method.Proof
			proof = &copyProof
		} else if !reflect.DeepEqual(*proof, method.Proof) {
			return hardPolicyMethodAttestation{}, fmt.Errorf("hard-policy rows mix static proofs")
		}
		if _, duplicate := observationIDs[method.ObservationID]; duplicate {
			return hardPolicyMethodAttestation{}, fmt.Errorf("hard-policy observation identities must be unique")
		}
		if _, duplicate := attackIDs[method.AttackID]; duplicate {
			return hardPolicyMethodAttestation{}, fmt.Errorf("hard-policy attack identities must be unique")
		}
		if _, duplicate := decisionIDs[method.DecisionReceiptID]; duplicate {
			return hardPolicyMethodAttestation{}, fmt.Errorf("hard-policy decision receipts must be unique")
		}
		binding := method.RuleID + "\x00" + method.EnforcementPoint
		if _, duplicate := observedBindings[binding]; duplicate {
			return hardPolicyMethodAttestation{}, fmt.Errorf("hard-policy observations must exactly cover proof bindings")
		}
		observationIDs[method.ObservationID] = struct{}{}
		attackIDs[method.AttackID] = struct{}{}
		decisionIDs[method.DecisionReceiptID] = struct{}{}
		observedBindings[binding] = struct{}{}
		dynamicPassed = dynamicPassed && method.Blocked == method.ShouldBlock && method.Violations == 0
		count++
	}
	if proof == nil {
		return hardPolicyMethodAttestation{}, nil
	}
	required := make(map[string]struct{}, len(proof.RequiredBindings))
	for _, binding := range proof.RequiredBindings {
		required[binding.RuleID+"\x00"+binding.EnforcementPoint] = struct{}{}
	}
	staticPassed := count == proof.LedgerTotalObservationCount && len(observedBindings) == len(required)
	if staticPassed {
		for binding := range required {
			if _, present := observedBindings[binding]; !present {
				staticPassed = false
				break
			}
		}
	}
	dynamicPassed = staticPassed && dynamicPassed
	return hardPolicyMethodAttestation{
		ObservationCount: count, TotalObservationCount: proof.LedgerTotalObservationCount,
		PolicySnapshotDigest: proof.PolicySnapshotDigest, ConfigDigest: proof.ConfigDigest,
		TargetID: proof.TargetID, BackendTopologyDigest: proof.BackendTopologyDigest,
		MixtureSnapshotDigest: proof.MixtureSnapshotDigest,
		StaticPassed:          &staticPassed, DynamicPassed: &dynamicPassed,
	}, nil
}
