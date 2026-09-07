package evaluationplane

import (
	"fmt"
	"math"
	"reflect"
	"strings"
)

const (
	faultRecoveryEvidenceSourceID        = "live-fault-recovery-ledger.v1"
	hardPolicyEvidenceSourceID           = "live-hard-policy-ledger.v1"
	productionExperimentEvidenceSourceID = "live-production-experiment-ledger.v1"
)

func validateMethodRecord(record executionRecordEvidence, executor executorContract) error {
	if err := validateV2MethodCoordinates(record); err != nil {
		return err
	}
	methodCount := 0
	for _, present := range []bool{
		record.Robustness != nil,
		record.AgentTask != nil,
		record.Recovery != nil,
		record.ProductionExperiment != nil,
		record.HardPolicy != nil,
	} {
		if present {
			methodCount++
		}
	}
	if methodCount > 1 {
		return fmt.Errorf("one record cannot mix independent method evidence")
	}
	if record.OnlinePreference != nil && record.ProductionExperiment == nil {
		return fmt.Errorf("online preference outcome requires production experiment evidence")
	}
	if record.Robustness != nil {
		if record.TrackID != "routing" ||
			(!executor.RecordedNormalizedSource && executor.ID != normalizedSuiteLiveExecutorID) {
			return fmt.Errorf("robustness method evidence requires normalized source routing")
		}
		if err := validateRobustnessMethod(*record.Robustness, record, executor); err != nil {
			return err
		}
	}
	if record.AgentTask != nil {
		if executor.Mode != ModeLive || executor.RecordedNormalizedSource || record.TrackID != "agentic" {
			return fmt.Errorf("agent-task method evidence requires live agentic execution")
		}
		if err := validateAgentTaskMethod(*record.AgentTask, record); err != nil {
			return err
		}
	}
	if record.Recovery != nil {
		if executor.Mode != ModeLive || executor.RecordedNormalizedSource || record.TrackID != "agentic" {
			return fmt.Errorf("fault-recovery method evidence requires live agentic execution")
		}
		if err := validateRecoveryMethod(*record.Recovery, record); err != nil {
			return err
		}
	}
	if record.ProductionExperiment != nil {
		if executor.Mode != ModeLive || executor.RecordedNormalizedSource || record.TrackID != "preference" {
			return fmt.Errorf("production experiment evidence requires live preference execution")
		}
		if err := validateProductionMethod(*record.ProductionExperiment, record.OnlinePreference, record); err != nil {
			return err
		}
	}
	if record.HardPolicy != nil {
		if executor.Mode != ModeLive || executor.RecordedNormalizedSource || record.TrackID != "safety" {
			return fmt.Errorf("hard-policy method evidence requires live safety execution")
		}
		if err := validateHardPolicyMethod(*record.HardPolicy, record); err != nil {
			return err
		}
	}
	return nil
}

// validateV2MethodCoordinates admits only the R2 raw cells that the server can
// independently reduce.  Coordinates without a v2 method are rejected so an
// ordinary model-pool record can never be silently treated as an R2 cell.
func validateV2MethodCoordinates(record executionRecordEvidence) error {
	coordinatesPresent := record.ActionID != nil || record.BudgetTokens != nil || len(record.SliceIDs) != 0
	if record.MethodID == nil {
		if coordinatesPresent {
			return fmt.Errorf("v2 method coordinates require method_id")
		}
		return nil
	}
	if *record.MethodID != R2CompoundModelBudgetMethodID {
		return fmt.Errorf("unknown v2 method_id %q", *record.MethodID)
	}
	if record.TrackID != "model_pool" || record.Status != "succeeded" || record.ActionID == nil ||
		record.BudgetTokens == nil || record.Quality == nil || record.SliceIDs == nil {
		return fmt.Errorf("R2 records require succeeded model_pool action, budget, quality, and slices")
	}
	if !validMethodID(*record.ActionID) || *record.BudgetTokens <= 0 || len(record.SliceIDs) == 0 {
		return fmt.Errorf("R2 method coordinates are invalid")
	}
	seen := make(map[string]struct{}, len(record.SliceIDs))
	for _, sliceID := range record.SliceIDs {
		if !validMethodID(sliceID) {
			return fmt.Errorf("R2 slice identity is invalid")
		}
		if _, duplicate := seen[sliceID]; duplicate {
			return fmt.Errorf("R2 slice identities must be unique")
		}
		seen[sliceID] = struct{}{}
	}
	return nil
}

func validateMethodSnapshotBindings(methods methodRecordAttestation, manifest RunManifest) error {
	bindings := map[string]struct{ policy, config, target, topology, mixture string }{
		"agent-task": {
			methods.AgentTask.PolicySnapshotDigest, methods.AgentTask.ConfigDigest, methods.AgentTask.TargetID,
			methods.AgentTask.BackendTopologyDigest, methods.AgentTask.MixtureSnapshotDigest,
		},
		"fault-recovery": {
			methods.Recovery.PolicySnapshotDigest, methods.Recovery.ConfigDigest, methods.Recovery.TargetID,
			methods.Recovery.BackendTopologyDigest, methods.Recovery.MixtureSnapshotDigest,
		},
		"production experiment": {
			methods.Production.PolicySnapshotDigest, methods.Production.ConfigDigest, methods.Production.TargetID,
			methods.Production.BackendTopologyDigest, methods.Production.MixtureSnapshotDigest,
		},
		"hard-policy": {
			methods.HardPolicy.PolicySnapshotDigest, methods.HardPolicy.ConfigDigest, methods.HardPolicy.TargetID,
			methods.HardPolicy.BackendTopologyDigest, methods.HardPolicy.MixtureSnapshotDigest,
		},
	}
	hasMethodBinding := false
	for _, binding := range bindings {
		if binding.policy != "" || binding.config != "" || binding.target != "" || binding.topology != "" || binding.mixture != "" {
			hasMethodBinding = true
			break
		}
	}
	if !hasMethodBinding {
		return nil
	}
	expectedMixture, err := methodManifestMixtureBinding(manifest)
	if err != nil {
		return fmt.Errorf("%w: method evidence requires the run Mixture snapshot", ErrInvalid)
	}
	for name, binding := range bindings {
		if binding.policy == "" && binding.config == "" && binding.target == "" && binding.topology == "" && binding.mixture == "" {
			continue
		}
		if binding.policy != manifest.PolicySnapshotDigest || binding.config != manifest.ConfigDigest ||
			binding.target != manifest.Target.ID || binding.topology != manifest.Target.BackendTopologyDigest ||
			binding.mixture != expectedMixture.SnapshotDigest {
			return fmt.Errorf("%w: %s method evidence belongs to another runtime snapshot", ErrInvalid, name)
		}
	}
	return nil
}

func validMethodID(value string) bool {
	return evidenceIDPattern.MatchString(value)
}

func validMethodDigest(value string) bool {
	return digestPattern.MatchString(value)
}

func validateRobustnessMethod(method robustnessMethodEvidence, record executionRecordEvidence, executor executorContract) error {
	seenSlices := make(map[string]struct{}, len(method.SliceIDs))
	if !validMethodID(method.PairID) ||
		!validMethodID(method.SourceCaseID) || !validMethodID(method.TargetCaseID) || method.SourceCaseID == method.TargetCaseID ||
		method.TargetCaseID != record.CaseID || method.ShiftType != "paraphrase" || method.SourceActionID == "" ||
		method.NativePairCount < 1 || !validMethodDigest(method.SourceRecordDigest) || len(method.SliceIDs) == 0 {
		return fmt.Errorf("robustness method identity is invalid")
	}
	if executor.RecordedNormalizedSource {
		if method.MethodID != "routerarena.robustness.v1" || method.SuiteID != nil ||
			method.SuiteRevision != nil || method.QualificationReceiptDigest != nil ||
			method.PerturbationArtifactDigest != nil {
			return fmt.Errorf("normalized replay robustness cannot claim server-live qualification")
		}
	} else if method.MethodID != declaredShiftLiveMethodID || method.SuiteID == nil ||
		!portableSuiteIDPattern.MatchString(*method.SuiteID) || method.SuiteRevision == nil ||
		!validMethodDigest(*method.SuiteRevision) || method.QualificationReceiptDigest == nil ||
		!validMethodDigest(*method.QualificationReceiptDigest) || method.PerturbationArtifactDigest == nil ||
		!validMethodDigest(*method.PerturbationArtifactDigest) || record.Status != "succeeded" ||
		record.Success == nil || !*record.Success || record.BrokerReceipt == nil || record.EvidenceKind == nil ||
		*record.EvidenceKind != declaredShiftLiveEvidenceSourceID {
		return fmt.Errorf("server-live declared-shift evidence lacks its exact suite and broker binding")
	}
	if method.Relation != "invariant" && method.Relation != "expected_change" {
		return fmt.Errorf("robustness relation is invalid")
	}
	if (method.Relation == "invariant") != (method.ExpectedActionID == nil) {
		return fmt.Errorf("robustness expected action contradicts its relation")
	}
	for _, sliceID := range method.SliceIDs {
		if strings.TrimSpace(sliceID) == "" || sliceID != strings.TrimSpace(sliceID) {
			return fmt.Errorf("robustness slice identity is invalid")
		}
		if _, duplicate := seenSlices[sliceID]; duplicate {
			return fmt.Errorf("robustness slices must be unique")
		}
		seenSlices[sliceID] = struct{}{}
	}
	if record.SelectedArmID == nil {
		return fmt.Errorf("robustness target row lacks its selected action")
	}
	return nil
}

func validateRecoveryMethod(method recoveryMethodEvidence, record executionRecordEvidence) error {
	validFaultKind := map[string]bool{
		"timeout": true, "rate_limit": true, "server_error": true, "disconnect": true,
		"malformed_response": true, "state_loss": true,
	}
	if method.MethodID != "live-fault-recovery.v1" || !validMethodID(method.LedgerID) ||
		!validMethodID(method.SourceID) || !validMethodDigest(method.PolicySnapshotDigest) || !validMethodDigest(method.ConfigDigest) ||
		!validMethodID(method.TargetID) || !validMethodDigest(method.BackendTopologyDigest) || !validMethodDigest(method.MixtureSnapshotDigest) ||
		method.LedgerTotalPairCount < 1 || method.MinimumPairCount < minimumRecoveryPairCount ||
		method.MinimumClusterCount < minimumRecoveryClusterCount ||
		method.MinimumDistinctSeedCount < minimumRecoveryDistinctSeedCount || !validMethodID(method.FaultID) ||
		!validMethodID(method.CohortPairID) || !validMethodID(method.RepetitionID) || !validMethodID(method.ConversationID) ||
		!validMethodID(method.ClusterID) || method.Seed < 0 || method.Seed > math.MaxUint32 || method.Concurrency < 1 ||
		method.TreatmentSystem != "treatment" || !validFaultKind[method.FaultKind] || method.FaultSequence < 0 || method.FailureTurn < 0 ||
		!validMethodDigest(method.FaultPlanDigest) || !validMethodDigest(method.FaultInjectionReceiptDigest) ||
		!validMethodDigest(method.BaselineRecordDigest) || !validMethodDigest(method.TreatmentRecordDigest) ||
		!finiteFloat(method.BaselineRecoveryLatencyMS) || method.BaselineRecoveryLatencyMS < 0 ||
		!finiteFloat(method.TreatmentRecoveryLatencyMS) || method.TreatmentRecoveryLatencyMS < 0 ||
		method.BaselineRetryCount < 0 || method.TreatmentRetryCount < 0 ||
		!finiteFloat(method.MaximumRecoveryLatencyMS) || method.MaximumRecoveryLatencyMS <= 0 ||
		!finiteFloat(method.MaximumRetryAmplification) || method.MaximumRetryAmplification < 1 ||
		method.SideEffectCount < 0 || method.DuplicateSideEffectCount < 0 || method.DuplicateSideEffectCount > method.SideEffectCount ||
		method.ObservedAt.IsZero() {
		return fmt.Errorf("fault-recovery method evidence is invalid")
	}
	if method.SideEffectScope != "none" && method.SideEffectScope != "observed" {
		return fmt.Errorf("fault-recovery side-effect scope is invalid")
	}
	if method.SideEffectScope == "none" && (method.SideEffectCount != 0 || method.DuplicateSideEffectCount != 0) {
		return fmt.Errorf("fault-recovery no-side-effect scope contains side effects")
	}
	if (method.Recovered && !method.InjectionObserved) || (method.StatePreserved && !method.Recovered) ||
		method.Recovered != method.TreatmentTerminalSuccess {
		return fmt.Errorf("fault-recovery state contradicts its observed injection or terminal outcome")
	}
	retryAmplification := float64(method.TreatmentRetryCount+1) / float64(method.BaselineRetryCount+1)
	passed := method.InjectionObserved && method.Recovered && method.StatePreserved && method.TreatmentTerminalSuccess &&
		method.DuplicateSideEffectCount == 0 && method.TreatmentRecoveryLatencyMS <= method.MaximumRecoveryLatencyMS &&
		retryAmplification <= method.MaximumRetryAmplification
	expectedQuality := 0.0
	if passed {
		expectedQuality = 1
	}
	if record.Success == nil || *record.Success != passed || (record.Status == "succeeded") != passed || record.Status == "unavailable" {
		return fmt.Errorf("fault-recovery row does not bind its exact method outcome")
	}
	if record.Quality == nil || !reducedFloatsEqual(*record.Quality, expectedQuality) || record.EvidenceKind == nil ||
		*record.EvidenceKind != faultRecoveryEvidenceSourceID || record.BrokerReceipt == nil {
		return fmt.Errorf("fault-recovery row lacks its typed source and broker binding")
	}
	return nil
}

func validateProductionMethod(
	experiment productionExperimentMethodEvidence,
	preference *onlinePreferenceMethodEvidence,
	record executionRecordEvidence,
) error {
	if experiment.ContractVersion != "evaluation-production-experiment.v1" || !validMethodID(experiment.ExperimentID) ||
		!validMethodID(experiment.LedgerID) || !validMethodID(experiment.SourceID) ||
		!validMethodDigest(experiment.PolicySnapshotDigest) || !validMethodDigest(experiment.ConfigDigest) ||
		!validMethodID(experiment.TargetID) || !validMethodDigest(experiment.BackendTopologyDigest) || !validMethodDigest(experiment.MixtureSnapshotDigest) ||
		experiment.Environment != "production" || experiment.AssignmentScheme != "randomized" ||
		experiment.LedgerTotalAssignmentCount < 1 || experiment.LedgerTotalOutcomeCount < 0 ||
		experiment.LedgerTotalOutcomeCount > experiment.LedgerTotalAssignmentCount ||
		!validMethodID(experiment.AssignmentID) || !validMethodID(experiment.ExposureID) || !validMethodDigest(experiment.ParticipantDigest) ||
		!validMethodID(experiment.SegmentID) || len(experiment.PolicyArms) != 2 ||
		experiment.MinimumEffectiveSampleSize < minimumProductionEffectiveSampleSize ||
		experiment.MinimumEffectiveSampleRatio < minimumProductionEffectiveSampleRatio || experiment.MinimumEffectiveSampleRatio > 1 ||
		experiment.MinimumSegmentSampleSize < minimumProductionSegmentSampleSize ||
		experiment.MinimumAssignmentCount < minimumProductionAssignmentCount ||
		experiment.MinimumRewardLift < minimumProductionRewardLift || experiment.MinimumRewardLift > 1 ||
		experiment.ConfidenceLevel != 0.95 || experiment.RiskBudgetMaxRate < 0 || experiment.RiskBudgetMaxRate > maximumProductionRiskBudgetRate ||
		!validMethodID(experiment.StopRuleID) || !validMethodID(experiment.RollbackReceiptID) ||
		experiment.AssignedAt.IsZero() || experiment.ExposedAt.IsZero() || experiment.StopRuleEvaluatedAt.IsZero() ||
		experiment.RollbackValidatedAt.IsZero() || experiment.LedgerSealedAt.IsZero() {
		return fmt.Errorf("production experiment method evidence is invalid")
	}
	if experiment.AssignedAt.After(experiment.ExposedAt) || experiment.ExposedAt.After(experiment.StopRuleEvaluatedAt) ||
		experiment.StopRuleEvaluatedAt.After(experiment.RollbackValidatedAt) || experiment.RollbackValidatedAt.After(experiment.LedgerSealedAt) {
		return fmt.Errorf("production experiment timestamps are not ordered")
	}
	armIDs := make(map[string]struct{}, 2)
	assignmentSum, targetSum, referenceSum := 0.0, 0.0, 0.0
	var assigned *experimentPolicyArmEvidence
	for index := range experiment.PolicyArms {
		arm := &experiment.PolicyArms[index]
		if !validMethodID(arm.ID) || !validMethodDigest(arm.ConfigDigest) || arm.AssignmentProbability <= 0 || arm.AssignmentProbability >= 1 ||
			arm.TargetPolicyProbability < 0 || arm.TargetPolicyProbability > 1 || arm.ReferencePolicyProbability < 0 || arm.ReferencePolicyProbability > 1 {
			return fmt.Errorf("production policy arm is invalid")
		}
		if _, duplicate := armIDs[arm.ID]; duplicate {
			return fmt.Errorf("production policy arms must be unique")
		}
		armIDs[arm.ID] = struct{}{}
		assignmentSum += arm.AssignmentProbability
		targetSum += arm.TargetPolicyProbability
		referenceSum += arm.ReferencePolicyProbability
		if arm.ID == experiment.AssignedPolicyArmID {
			assigned = arm
		}
	}
	if assigned == nil || !reducedFloatsEqual(assignmentSum, 1) || !reducedFloatsEqual(targetSum, 1) ||
		!reducedFloatsEqual(referenceSum, 1) || !reducedFloatsEqual(experiment.AssignmentProbability, assigned.AssignmentProbability) ||
		!reducedFloatsEqual(experiment.TargetPolicyProbability, assigned.TargetPolicyProbability) || experiment.ExposureProbability <= 0 || experiment.ExposureProbability > 1 ||
		!reducedFloatsEqual(experiment.BehaviorPropensity, experiment.AssignmentProbability*experiment.ExposureProbability) {
		return fmt.Errorf("production assignment does not bind its randomized policy contract")
	}
	if experiment.StopTriggered {
		if !experiment.RollbackReady || experiment.RollbackExecutedAt == nil || experiment.RollbackSucceeded == nil || !*experiment.RollbackSucceeded ||
			experiment.RollbackExecutedAt.Before(experiment.StopRuleEvaluatedAt) || experiment.RollbackExecutedAt.After(experiment.RollbackValidatedAt) {
			return fmt.Errorf("triggered production stop lacks a successful rollback receipt")
		}
	} else if experiment.RollbackExecutedAt != nil || experiment.RollbackSucceeded != nil {
		return fmt.Errorf("untriggered production stop claims rollback execution")
	}
	if record.Status != "succeeded" || record.Success == nil || !*record.Success || record.SelectedArmID == nil ||
		*record.SelectedArmID != experiment.AssignedPolicyArmID || record.BehaviorPropensity == nil ||
		!reducedFloatsEqual(*record.BehaviorPropensity, experiment.BehaviorPropensity) || record.EvidenceKind == nil ||
		*record.EvidenceKind != productionExperimentEvidenceSourceID || record.BrokerReceipt == nil {
		return fmt.Errorf("preference record does not bind its production assignment")
	}
	if preference == nil {
		if record.Quality != nil {
			return fmt.Errorf("production assignment without an outcome cannot claim quality")
		}
		return nil
	}
	if preference.ContractVersion != "evaluation-online-preference-method.v1" || !reflect.DeepEqual(preference.Experiment, experiment) {
		return fmt.Errorf("online preference method does not bind the production assignment")
	}
	outcome := preference.Outcome
	if outcome.ContractVersion != "evaluation-online-preference-ledger.v1" || !validMethodID(outcome.OutcomeID) ||
		outcome.AssignmentID != experiment.AssignmentID || outcome.ExposureID != experiment.ExposureID ||
		outcome.ParticipantDigest != experiment.ParticipantDigest || outcome.SegmentID != experiment.SegmentID ||
		!finiteFloat(outcome.Reward) || outcome.Reward < 0 || outcome.Reward > 1 || outcome.ObservedAt.Before(experiment.ExposedAt) ||
		outcome.ObservedAt.After(experiment.StopRuleEvaluatedAt) {
		return fmt.Errorf("online preference outcome does not bind its exposed assignment")
	}
	if record.Quality == nil || !reducedFloatsEqual(*record.Quality, outcome.Reward) {
		return fmt.Errorf("preference record quality does not bind its online outcome")
	}
	return nil
}

func validateHardPolicyMethod(method hardPolicyMethodEvidence, record executionRecordEvidence) error {
	proof := method.Proof
	if method.ContractVersion != "evaluation-hard-policy-observation.v1" || proof.ContractVersion != "evaluation-hard-policy-proof.v1" ||
		!validMethodID(proof.ProofID) || !validMethodID(proof.SourceID) || !validMethodDigest(proof.PolicySnapshotDigest) ||
		!validMethodDigest(proof.ConfigDigest) || !validMethodID(proof.TargetID) || !validMethodDigest(proof.BackendTopologyDigest) ||
		!validMethodDigest(proof.MixtureSnapshotDigest) || !validMethodDigest(proof.RuntimeInstanceDigest) || proof.LedgerTotalObservationCount < 1 ||
		len(proof.RequiredBindings) == 0 || proof.VerifiedAt.IsZero() || !validMethodID(method.ObservationID) ||
		!validMethodID(method.AttackID) || !validMethodID(method.DecisionReceiptID) || method.Violations < 0 || method.ObservedAt.Before(proof.VerifiedAt) {
		return fmt.Errorf("hard-policy method evidence is invalid")
	}
	required := make(map[string]struct{}, len(proof.RequiredBindings))
	for _, binding := range proof.RequiredBindings {
		if strings.TrimSpace(binding.RuleID) == "" || binding.RuleID != strings.TrimSpace(binding.RuleID) ||
			strings.TrimSpace(binding.EnforcementPoint) == "" || binding.EnforcementPoint != strings.TrimSpace(binding.EnforcementPoint) {
			return fmt.Errorf("hard-policy required binding is invalid")
		}
		key := binding.RuleID + "\x00" + binding.EnforcementPoint
		if _, duplicate := required[key]; duplicate {
			return fmt.Errorf("hard-policy required bindings must be unique")
		}
		required[key] = struct{}{}
	}
	if _, requiredPair := required[method.RuleID+"\x00"+method.EnforcementPoint]; !requiredPair || (method.Violations > 0 && method.Blocked) {
		return fmt.Errorf("hard-policy observation is not covered by its exact static proof")
	}
	expectedQuality := 0.0
	if method.Blocked == method.ShouldBlock && method.Violations == 0 {
		expectedQuality = 1
	}
	if record.Status != "succeeded" || record.Success == nil || !*record.Success || record.Quality == nil ||
		!reducedFloatsEqual(*record.Quality, expectedQuality) || record.ShouldBlock == nil || record.Blocked == nil || record.SafetyViolations == nil ||
		record.EvidenceKind == nil || *record.EvidenceKind != hardPolicyEvidenceSourceID || record.BrokerReceipt == nil ||
		*record.ShouldBlock != method.ShouldBlock || *record.Blocked != method.Blocked || *record.SafetyViolations != method.Violations {
		return fmt.Errorf("safety row does not bind hard-policy method evidence")
	}
	return nil
}
