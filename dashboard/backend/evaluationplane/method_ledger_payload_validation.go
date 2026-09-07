package evaluationplane

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"reflect"
	"time"
)

const maximumMethodLedgerFreshness = 24 * time.Hour

type faultRecoveryLedgerPayload struct {
	ContractVersion           string                   `json:"contract_version"`
	LedgerID                  string                   `json:"ledger_id"`
	SourceID                  string                   `json:"source_id"`
	Environment               string                   `json:"environment"`
	PolicySnapshotDigest      string                   `json:"policy_snapshot_digest"`
	ConfigDigest              string                   `json:"config_digest"`
	TargetID                  string                   `json:"target_id"`
	BackendTopologyDigest     string                   `json:"backend_topology_digest"`
	Mixture                   methodMixtureBinding     `json:"mixture"`
	MinimumPairCount          int                      `json:"minimum_pair_count"`
	MinimumClusterCount       int                      `json:"minimum_cluster_count"`
	MinimumDistinctSeedCount  int                      `json:"minimum_distinct_seed_count"`
	MaximumRecoveryLatencyMS  float64                  `json:"maximum_recovery_latency_ms"`
	MaximumRetryAmplification float64                  `json:"maximum_retry_amplification"`
	WindowStartedAt           time.Time                `json:"window_started_at"`
	WindowEndedAt             time.Time                `json:"window_ended_at"`
	SealedAt                  time.Time                `json:"sealed_at"`
	Pairs                     []recoveryMethodEvidence `json:"pairs"`
}

type hardPolicyLedgerPayload struct {
	ContractVersion       string                        `json:"contract_version"`
	LedgerID              string                        `json:"ledger_id"`
	SourceID              string                        `json:"source_id"`
	Environment           string                        `json:"environment"`
	PolicySnapshotDigest  string                        `json:"policy_snapshot_digest"`
	ConfigDigest          string                        `json:"config_digest"`
	TargetID              string                        `json:"target_id"`
	BackendTopologyDigest string                        `json:"backend_topology_digest"`
	Mixture               methodMixtureBinding          `json:"mixture"`
	Proof                 hardPolicyStaticProofEvidence `json:"proof"`
	WindowStartedAt       time.Time                     `json:"window_started_at"`
	WindowEndedAt         time.Time                     `json:"window_ended_at"`
	SealedAt              time.Time                     `json:"sealed_at"`
	Observations          []hardPolicyMethodEvidence    `json:"observations"`
}

type productionExperimentLedgerPayload struct {
	ContractVersion             string                               `json:"contract_version"`
	ExperimentID                string                               `json:"experiment_id"`
	LedgerID                    string                               `json:"ledger_id"`
	SourceID                    string                               `json:"source_id"`
	PolicySnapshotDigest        string                               `json:"policy_snapshot_digest"`
	ConfigDigest                string                               `json:"config_digest"`
	TargetID                    string                               `json:"target_id"`
	BackendTopologyDigest       string                               `json:"backend_topology_digest"`
	Mixture                     methodMixtureBinding                 `json:"mixture"`
	Environment                 string                               `json:"environment"`
	AssignmentScheme            string                               `json:"assignment_scheme"`
	RiskBudgetMaxRate           float64                              `json:"risk_budget_max_rate"`
	StopRuleID                  string                               `json:"stop_rule_id"`
	StopRuleEvaluatedAt         time.Time                            `json:"stop_rule_evaluated_at"`
	StopTriggered               bool                                 `json:"stop_triggered"`
	RollbackReceiptID           string                               `json:"rollback_receipt_id"`
	RollbackValidatedAt         time.Time                            `json:"rollback_validated_at"`
	RollbackReady               bool                                 `json:"rollback_ready"`
	RollbackExecutedAt          *time.Time                           `json:"rollback_executed_at,omitempty"`
	RollbackSucceeded           *bool                                `json:"rollback_succeeded,omitempty"`
	MinimumEffectiveSampleSize  float64                              `json:"minimum_effective_sample_size"`
	MinimumEffectiveSampleRatio float64                              `json:"minimum_effective_sample_ratio"`
	MinimumSegmentSampleSize    int                                  `json:"minimum_segment_sample_size"`
	MinimumAssignmentCount      int                                  `json:"minimum_assignment_count"`
	MinimumRewardLift           float64                              `json:"minimum_reward_lift"`
	ConfidenceLevel             float64                              `json:"confidence_level"`
	WindowStartedAt             time.Time                            `json:"window_started_at"`
	WindowEndedAt               time.Time                            `json:"window_ended_at"`
	SealedAt                    time.Time                            `json:"sealed_at"`
	Assignments                 []productionExperimentMethodEvidence `json:"assignments"`
	PreferenceOutcomes          []onlinePreferenceOutcomeEvidence    `json:"preference_outcomes"`
}

func isMethodLedgerOperation(operation string) bool {
	switch operation {
	case workerBrokerAgentTaskLedger, workerBrokerFaultRecoveryLedger, workerBrokerHardPolicyLedger, workerBrokerProductionExperimentLedger:
		return true
	default:
		return false
	}
}

func validateMethodLedgerBrokerBinding(
	entry executionAttestationEntry,
	records []executionRecordEvidence,
	manifest RunManifest,
) error {
	if !entry.UpstreamAttempted || !entry.Success || entry.responsePayload == nil || len(records) == 0 ||
		entry.FetchedAt == nil || entry.LedgerSealedAt == nil {
		return fmt.Errorf("method ledger fetch was not a successful retained response")
	}
	if entry.AttemptID != "ledger-fetch" {
		return fmt.Errorf("method ledger fetch attempt identity is invalid")
	}
	for _, record := range records {
		if record.BrokerReceipt == nil || *record.BrokerReceipt != entry.BrokerReceipt {
			return fmt.Errorf("method ledger records do not share their exact broker receipt")
		}
	}
	encoded, err := json.Marshal(entry.responsePayload)
	if err != nil {
		return fmt.Errorf("encode retained method ledger: %w", err)
	}
	switch entry.Operation {
	case workerBrokerAgentTaskLedger:
		if entry.TrackID != "agentic" || entry.CaseID != "agent-task-ledger" {
			return fmt.Errorf("agent-task ledger fetch identity is invalid")
		}
		var payload agentTaskLedgerPayload
		if err := decodeExactJSON(encoded, &payload); err != nil {
			return fmt.Errorf("decode retained agent-task ledger: %w", err)
		}
		if !payload.SealedAt.Equal(*entry.LedgerSealedAt) {
			return fmt.Errorf("agent-task ledger seal differs from its broker receipt")
		}
		if err := validateMethodLedgerFreshness(*entry.LedgerSealedAt, *entry.FetchedAt); err != nil {
			return err
		}
		return validateAgentTaskLedgerPayload(payload, records, manifest)
	case workerBrokerFaultRecoveryLedger:
		if entry.TrackID != "agentic" || entry.CaseID != "fault-recovery-ledger" {
			return fmt.Errorf("fault-recovery ledger fetch identity is invalid")
		}
		var payload faultRecoveryLedgerPayload
		if err := decodeExactJSON(encoded, &payload); err != nil {
			return fmt.Errorf("decode retained fault-recovery ledger: %w", err)
		}
		if !payload.SealedAt.Equal(*entry.LedgerSealedAt) {
			return fmt.Errorf("fault-recovery ledger seal differs from its broker receipt")
		}
		if err := validateMethodLedgerFreshness(*entry.LedgerSealedAt, *entry.FetchedAt); err != nil {
			return err
		}
		return validateFaultRecoveryLedgerPayload(payload, records, manifest)
	case workerBrokerHardPolicyLedger:
		if entry.TrackID != "safety" || entry.CaseID != "hard-policy-ledger" {
			return fmt.Errorf("hard-policy ledger fetch identity is invalid")
		}
		var payload hardPolicyLedgerPayload
		if err := decodeExactJSON(encoded, &payload); err != nil {
			return fmt.Errorf("decode retained hard-policy ledger: %w", err)
		}
		if !payload.SealedAt.Equal(*entry.LedgerSealedAt) {
			return fmt.Errorf("hard-policy ledger seal differs from its broker receipt")
		}
		if err := validateMethodLedgerFreshness(*entry.LedgerSealedAt, *entry.FetchedAt); err != nil {
			return err
		}
		return validateHardPolicyLedgerPayload(payload, records, manifest)
	case workerBrokerProductionExperimentLedger:
		if entry.TrackID != "preference" || entry.CaseID != "production-ledger" {
			return fmt.Errorf("production ledger fetch identity is invalid")
		}
		var payload productionExperimentLedgerPayload
		if err := decodeExactJSON(encoded, &payload); err != nil {
			return fmt.Errorf("decode retained production ledger: %w", err)
		}
		if !payload.SealedAt.Equal(*entry.LedgerSealedAt) {
			return fmt.Errorf("production ledger seal differs from its broker receipt")
		}
		if err := validateMethodLedgerFreshness(*entry.LedgerSealedAt, *entry.FetchedAt); err != nil {
			return err
		}
		return validateProductionLedgerPayload(payload, records, manifest)
	default:
		return fmt.Errorf("operation is not a method ledger")
	}
}

func validateFaultRecoveryLedgerPayload(
	payload faultRecoveryLedgerPayload,
	records []executionRecordEvidence,
	manifest RunManifest,
) error {
	expectedMixture, err := methodManifestMixtureBinding(manifest)
	if err != nil {
		return err
	}
	if payload.ContractVersion != "evaluation-fault-recovery-ledger.v1" || payload.Environment != "production" ||
		!validMethodID(payload.LedgerID) || !validMethodID(payload.SourceID) || payload.PolicySnapshotDigest != manifest.PolicySnapshotDigest ||
		payload.ConfigDigest != manifest.ConfigDigest || !validMethodID(payload.TargetID) || payload.TargetID != manifest.Target.ID ||
		!validMethodDigest(payload.BackendTopologyDigest) || payload.BackendTopologyDigest != manifest.Target.BackendTopologyDigest ||
		!validMethodMixtureBinding(payload.Mixture) || !reflect.DeepEqual(payload.Mixture, expectedMixture) ||
		payload.MinimumPairCount < minimumRecoveryPairCount ||
		payload.MinimumClusterCount < minimumRecoveryClusterCount ||
		payload.MinimumDistinctSeedCount < minimumRecoveryDistinctSeedCount || !finiteFloat(payload.MaximumRecoveryLatencyMS) ||
		payload.MaximumRecoveryLatencyMS <= 0 || !finiteFloat(payload.MaximumRetryAmplification) || payload.MaximumRetryAmplification < 1 ||
		!validSealedMethodWindow(payload.WindowStartedAt, payload.WindowEndedAt, payload.SealedAt) || len(payload.Pairs) != len(records) {
		return fmt.Errorf("fault-recovery ledger envelope violates its sealed contract")
	}
	byFault := make(map[string]recoveryMethodEvidence, len(payload.Pairs))
	repetitions := make(map[string]struct{}, len(payload.Pairs))
	injectionReceipts := make(map[string]struct{}, len(payload.Pairs))
	for _, pair := range payload.Pairs {
		if pair.LedgerID != payload.LedgerID || pair.SourceID != payload.SourceID || pair.PolicySnapshotDigest != payload.PolicySnapshotDigest ||
			pair.ConfigDigest != payload.ConfigDigest || pair.TargetID != payload.TargetID ||
			pair.BackendTopologyDigest != payload.BackendTopologyDigest || pair.MixtureSnapshotDigest != payload.Mixture.SnapshotDigest ||
			pair.LedgerTotalPairCount != len(payload.Pairs) ||
			pair.MinimumPairCount != payload.MinimumPairCount || pair.MinimumClusterCount != payload.MinimumClusterCount ||
			pair.MinimumDistinctSeedCount != payload.MinimumDistinctSeedCount ||
			!reducedFloatsEqual(pair.MaximumRecoveryLatencyMS, payload.MaximumRecoveryLatencyMS) ||
			!reducedFloatsEqual(pair.MaximumRetryAmplification, payload.MaximumRetryAmplification) ||
			pair.ObservedAt.Before(payload.WindowStartedAt) || pair.ObservedAt.After(payload.WindowEndedAt) {
			return fmt.Errorf("fault-recovery pair does not bind the sealed ledger")
		}
		if _, duplicate := byFault[pair.FaultID]; duplicate {
			return fmt.Errorf("fault-recovery ledger contains a duplicate fault")
		}
		repetition := pair.CohortPairID + "\x00" + pair.RepetitionID
		if _, duplicate := repetitions[repetition]; duplicate {
			return fmt.Errorf("fault-recovery ledger contains a duplicate cohort repetition")
		}
		if _, duplicate := injectionReceipts[pair.FaultInjectionReceiptDigest]; duplicate {
			return fmt.Errorf("fault-recovery ledger contains a reused injection receipt")
		}
		byFault[pair.FaultID] = pair
		repetitions[repetition] = struct{}{}
		injectionReceipts[pair.FaultInjectionReceiptDigest] = struct{}{}
	}
	for _, record := range records {
		if record.Recovery == nil || record.TrackID != "agentic" || record.BrokerReceipt == nil {
			return fmt.Errorf("fault-recovery ledger is bound to a non-recovery record")
		}
		pair, present := byFault[record.Recovery.FaultID]
		if !present || !canonicalMethodValuesEqual(pair, *record.Recovery) ||
			record.CaseID != methodLedgerCaseID("fault-recovery", payload.LedgerID, pair.FaultID) ||
			record.ID != "agentic-"+record.CaseID || record.AttemptID != "agentic-"+record.CaseID {
			return fmt.Errorf("fault-recovery ledger membership differs from its emitted records")
		}
		delete(byFault, pair.FaultID)
	}
	if len(byFault) != 0 {
		return fmt.Errorf("fault-recovery ledger contains an unreported pair")
	}
	return nil
}

func validateHardPolicyLedgerPayload(
	payload hardPolicyLedgerPayload,
	records []executionRecordEvidence,
	manifest RunManifest,
) error {
	expectedMixture, err := methodManifestMixtureBinding(manifest)
	if err != nil {
		return err
	}
	if payload.ContractVersion != "evaluation-hard-policy-ledger.v1" || payload.Environment != "production" ||
		!validMethodID(payload.LedgerID) || !validMethodID(payload.SourceID) || payload.PolicySnapshotDigest != manifest.PolicySnapshotDigest ||
		payload.ConfigDigest != manifest.ConfigDigest || !validMethodID(payload.TargetID) || payload.TargetID != manifest.Target.ID ||
		!validMethodDigest(payload.BackendTopologyDigest) || payload.BackendTopologyDigest != manifest.Target.BackendTopologyDigest ||
		!validMethodMixtureBinding(payload.Mixture) || !reflect.DeepEqual(payload.Mixture, expectedMixture) || payload.Proof.SourceID != payload.SourceID ||
		payload.Proof.PolicySnapshotDigest != payload.PolicySnapshotDigest || payload.Proof.ConfigDigest != payload.ConfigDigest ||
		payload.Proof.TargetID != payload.TargetID || payload.Proof.BackendTopologyDigest != payload.BackendTopologyDigest ||
		payload.Proof.MixtureSnapshotDigest != payload.Mixture.SnapshotDigest ||
		payload.Proof.LedgerTotalObservationCount != len(payload.Observations) ||
		!validSealedMethodWindow(payload.WindowStartedAt, payload.WindowEndedAt, payload.SealedAt) || len(payload.Observations) != len(records) {
		return fmt.Errorf("hard-policy ledger envelope violates its sealed contract")
	}
	byObservation := make(map[string]hardPolicyMethodEvidence, len(payload.Observations))
	attackIDs := make(map[string]struct{}, len(payload.Observations))
	decisionReceipts := make(map[string]struct{}, len(payload.Observations))
	requiredBindings := make(map[string]struct{}, len(payload.Proof.RequiredBindings))
	for _, binding := range payload.Proof.RequiredBindings {
		requiredBindings[binding.RuleID+"\x00"+binding.EnforcementPoint] = struct{}{}
	}
	observedBindings := make(map[string]struct{}, len(payload.Observations))
	for _, observation := range payload.Observations {
		if !canonicalMethodValuesEqual(payload.Proof, observation.Proof) || observation.ObservedAt.Before(payload.WindowStartedAt) ||
			observation.ObservedAt.After(payload.WindowEndedAt) {
			return fmt.Errorf("hard-policy observation does not bind the sealed ledger")
		}
		if _, duplicate := byObservation[observation.ObservationID]; duplicate {
			return fmt.Errorf("hard-policy ledger contains a duplicate observation")
		}
		if _, duplicate := attackIDs[observation.AttackID]; duplicate {
			return fmt.Errorf("hard-policy ledger contains a duplicate attack")
		}
		if _, duplicate := decisionReceipts[observation.DecisionReceiptID]; duplicate {
			return fmt.Errorf("hard-policy ledger contains a reused decision receipt")
		}
		byObservation[observation.ObservationID] = observation
		attackIDs[observation.AttackID] = struct{}{}
		decisionReceipts[observation.DecisionReceiptID] = struct{}{}
		observedBindings[observation.RuleID+"\x00"+observation.EnforcementPoint] = struct{}{}
	}
	if len(observedBindings) != len(requiredBindings) {
		return fmt.Errorf("hard-policy observations do not exactly cover proof bindings")
	}
	for binding := range requiredBindings {
		if _, observed := observedBindings[binding]; !observed {
			return fmt.Errorf("hard-policy observations do not exactly cover proof bindings")
		}
	}
	for _, record := range records {
		if record.HardPolicy == nil || record.TrackID != "safety" || record.BrokerReceipt == nil {
			return fmt.Errorf("hard-policy ledger is bound to a non-safety record")
		}
		observation, present := byObservation[record.HardPolicy.ObservationID]
		if !present || !canonicalMethodValuesEqual(observation, *record.HardPolicy) ||
			record.CaseID != methodLedgerCaseID("hard-policy", payload.LedgerID, observation.ObservationID) ||
			record.ID != "safety-"+record.CaseID || record.AttemptID != "safety-"+record.CaseID {
			return fmt.Errorf("hard-policy ledger membership differs from its emitted records")
		}
		delete(byObservation, observation.ObservationID)
	}
	if len(byObservation) != 0 {
		return fmt.Errorf("hard-policy ledger contains an unreported observation")
	}
	return nil
}

func validateProductionLedgerPayload(
	payload productionExperimentLedgerPayload,
	records []executionRecordEvidence,
	manifest RunManifest,
) error {
	expectedMixture, err := methodManifestMixtureBinding(manifest)
	if err != nil {
		return err
	}
	if err := validateProductionLedgerEnvelope(payload, records, manifest, expectedMixture); err != nil {
		return err
	}
	byAssignment := make(map[string]productionExperimentMethodEvidence, len(payload.Assignments))
	exposures := make(map[string]struct{}, len(payload.Assignments))
	participants := make(map[string]struct{}, len(payload.Assignments))
	for _, assignment := range payload.Assignments {
		if assignment.ExperimentID != payload.ExperimentID || assignment.LedgerID != payload.LedgerID ||
			assignment.LedgerTotalAssignmentCount != len(payload.Assignments) || assignment.LedgerTotalOutcomeCount != len(payload.PreferenceOutcomes) ||
			assignment.SourceID != payload.SourceID || assignment.PolicySnapshotDigest != payload.PolicySnapshotDigest || assignment.ConfigDigest != payload.ConfigDigest ||
			assignment.TargetID != payload.TargetID || assignment.BackendTopologyDigest != payload.BackendTopologyDigest ||
			assignment.MixtureSnapshotDigest != payload.Mixture.SnapshotDigest ||
			assignment.Environment != payload.Environment || assignment.AssignmentScheme != payload.AssignmentScheme ||
			!reducedFloatsEqual(assignment.RiskBudgetMaxRate, payload.RiskBudgetMaxRate) || assignment.StopRuleID != payload.StopRuleID ||
			!assignment.StopRuleEvaluatedAt.Equal(payload.StopRuleEvaluatedAt) || assignment.StopTriggered != payload.StopTriggered ||
			assignment.RollbackReceiptID != payload.RollbackReceiptID || !assignment.RollbackValidatedAt.Equal(payload.RollbackValidatedAt) ||
			assignment.RollbackReady != payload.RollbackReady || !optionalTimesEqual(assignment.RollbackExecutedAt, payload.RollbackExecutedAt) ||
			!optionalBoolsEqual(assignment.RollbackSucceeded, payload.RollbackSucceeded) || assignment.MinimumAssignmentCount != payload.MinimumAssignmentCount ||
			!reducedFloatsEqual(assignment.MinimumEffectiveSampleSize, payload.MinimumEffectiveSampleSize) ||
			!reducedFloatsEqual(assignment.MinimumEffectiveSampleRatio, payload.MinimumEffectiveSampleRatio) ||
			assignment.MinimumSegmentSampleSize != payload.MinimumSegmentSampleSize || !reducedFloatsEqual(assignment.MinimumRewardLift, payload.MinimumRewardLift) ||
			!reducedFloatsEqual(assignment.ConfidenceLevel, payload.ConfidenceLevel) || !assignment.LedgerSealedAt.Equal(payload.SealedAt) ||
			assignment.AssignedAt.Before(payload.WindowStartedAt) || assignment.ExposedAt.After(payload.WindowEndedAt) {
			return fmt.Errorf("production assignment does not bind the sealed ledger")
		}
		if assignment.SelectedModelID != nil && !manifestHasModelArm(manifest, *assignment.SelectedModelID) {
			return fmt.Errorf("production assignment selected an undeclared model")
		}
		if _, duplicate := byAssignment[assignment.AssignmentID]; duplicate {
			return fmt.Errorf("production ledger contains a duplicate assignment")
		}
		if _, duplicate := exposures[assignment.ExposureID]; duplicate {
			return fmt.Errorf("production ledger contains a duplicate exposure")
		}
		if _, duplicate := participants[assignment.ParticipantDigest]; duplicate {
			return fmt.Errorf("production ledger contains a duplicate participant")
		}
		byAssignment[assignment.AssignmentID] = assignment
		exposures[assignment.ExposureID] = struct{}{}
		participants[assignment.ParticipantDigest] = struct{}{}
	}
	outcomes := make(map[string]onlinePreferenceOutcomeEvidence, len(payload.PreferenceOutcomes))
	outcomeIDs := make(map[string]struct{}, len(payload.PreferenceOutcomes))
	for _, outcome := range payload.PreferenceOutcomes {
		assignment, present := byAssignment[outcome.AssignmentID]
		if !present || outcome.ExposureID != assignment.ExposureID || outcome.ParticipantDigest != assignment.ParticipantDigest ||
			outcome.SegmentID != assignment.SegmentID || outcome.ObservedAt.Before(assignment.ExposedAt) || outcome.ObservedAt.After(payload.WindowEndedAt) {
			return fmt.Errorf("production preference outcome does not bind the sealed window")
		}
		if _, duplicate := outcomes[outcome.AssignmentID]; duplicate {
			return fmt.Errorf("production ledger contains duplicate outcomes for an assignment")
		}
		if _, duplicate := outcomeIDs[outcome.OutcomeID]; duplicate {
			return fmt.Errorf("production ledger contains a duplicate outcome identity")
		}
		outcomes[outcome.AssignmentID] = outcome
		outcomeIDs[outcome.OutcomeID] = struct{}{}
	}
	for _, record := range records {
		if record.ProductionExperiment == nil || record.TrackID != "preference" || record.BrokerReceipt == nil {
			return fmt.Errorf("production ledger is bound to a non-experiment record")
		}
		assignment, present := byAssignment[record.ProductionExperiment.AssignmentID]
		if !present || !canonicalMethodValuesEqual(assignment, *record.ProductionExperiment) ||
			record.CaseID != methodLedgerCaseID("experiment", payload.LedgerID, assignment.AssignmentID) ||
			record.ID != "preference-"+record.CaseID || record.AttemptID != "preference-"+record.CaseID {
			return fmt.Errorf("production ledger membership differs from its emitted records")
		}
		outcome, hasOutcome := outcomes[assignment.AssignmentID]
		if hasOutcome != (record.OnlinePreference != nil) ||
			(hasOutcome && !canonicalMethodValuesEqual(outcome, record.OnlinePreference.Outcome)) {
			return fmt.Errorf("production outcome membership differs from its emitted records")
		}
		delete(byAssignment, assignment.AssignmentID)
		delete(outcomes, assignment.AssignmentID)
	}
	if len(byAssignment) != 0 || len(outcomes) != 0 {
		return fmt.Errorf("production ledger contains unreported assignments or outcomes")
	}
	return nil
}

func validateProductionLedgerEnvelope(
	payload productionExperimentLedgerPayload,
	records []executionRecordEvidence,
	manifest RunManifest,
	expectedMixture methodMixtureBinding,
) error {
	if payload.ContractVersion != "evaluation-production-experiment-ledger.v1" || payload.Environment != "production" ||
		payload.AssignmentScheme != "randomized" || !validMethodID(payload.ExperimentID) || !validMethodID(payload.LedgerID) ||
		!validMethodID(payload.SourceID) || payload.PolicySnapshotDigest != manifest.PolicySnapshotDigest || payload.ConfigDigest != manifest.ConfigDigest ||
		!validMethodID(payload.TargetID) || payload.TargetID != manifest.Target.ID ||
		!validMethodDigest(payload.BackendTopologyDigest) || payload.BackendTopologyDigest != manifest.Target.BackendTopologyDigest ||
		!validMethodMixtureBinding(payload.Mixture) || !reflect.DeepEqual(payload.Mixture, expectedMixture) ||
		payload.MinimumAssignmentCount < minimumProductionAssignmentCount || payload.MinimumEffectiveSampleSize < minimumProductionEffectiveSampleSize ||
		payload.MinimumEffectiveSampleRatio < minimumProductionEffectiveSampleRatio || payload.MinimumSegmentSampleSize < minimumProductionSegmentSampleSize ||
		payload.MinimumRewardLift < minimumProductionRewardLift || payload.RiskBudgetMaxRate < 0 || payload.RiskBudgetMaxRate > maximumProductionRiskBudgetRate ||
		payload.ConfidenceLevel != 0.95 || !validSealedMethodWindow(payload.WindowStartedAt, payload.WindowEndedAt, payload.SealedAt) ||
		payload.StopRuleEvaluatedAt.Before(payload.WindowEndedAt) || payload.RollbackValidatedAt.Before(payload.StopRuleEvaluatedAt) ||
		payload.RollbackValidatedAt.After(payload.SealedAt) ||
		len(payload.Assignments) != len(records) || len(payload.PreferenceOutcomes) > len(payload.Assignments) {
		return fmt.Errorf("production ledger envelope violates its sealed contract")
	}
	return nil
}

func validSealedMethodWindow(startedAt, endedAt, sealedAt time.Time) bool {
	return !startedAt.IsZero() && startedAt.Before(endedAt) && !endedAt.After(sealedAt)
}

func validateMethodLedgerFreshness(sealedAt, fetchedAt time.Time) error {
	if sealedAt.IsZero() || fetchedAt.IsZero() || sealedAt.After(fetchedAt) {
		return fmt.Errorf("method ledger seal is in the future relative to its broker fetch")
	}
	if fetchedAt.Sub(sealedAt) > maximumMethodLedgerFreshness {
		return fmt.Errorf("method ledger exceeds the maximum %s freshness window", maximumMethodLedgerFreshness)
	}
	return nil
}

func canonicalMethodValuesEqual(left, right any) bool {
	leftDigest, leftErr := canonicalValueDigest(left)
	rightDigest, rightErr := canonicalValueDigest(right)
	return leftErr == nil && rightErr == nil && leftDigest == rightDigest
}

func methodLedgerCaseID(prefix, ledgerID, rowID string) string {
	digest := sha256.Sum256([]byte(ledgerID + "\x00" + rowID))
	return fmt.Sprintf("%s-%x", prefix, digest[:12])
}

func manifestHasModelArm(manifest RunManifest, modelID string) bool {
	if manifest.Target.Mixture == nil {
		return false
	}
	for _, arm := range manifest.Target.Mixture.ModelArms {
		if arm.ID == modelID {
			return true
		}
	}
	return false
}
