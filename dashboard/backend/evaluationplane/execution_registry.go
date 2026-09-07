package evaluationplane

import (
	"fmt"
)

func (r *Registry) executionContracts() *executionContractRegistry {
	executors := make(map[string]executorContract, len(r.executors))
	for id, contract := range r.executors {
		contract.TrackIDs = append([]TrackID(nil), contract.TrackIDs...)
		executors[id] = contract
	}
	targets := make(map[string]executionTargetContract, len(r.targets))
	for id, target := range r.targets {
		targets[id] = executionTargetContract{Definition: copyTargetDefinition(target)}
	}
	return &executionContractRegistry{executors: executors, targets: targets}
}

func (registry *executionContractRegistry) resolve(manifest RunManifest) (resolvedExecutionContract, error) {
	if registry == nil {
		return resolvedExecutionContract{}, fmt.Errorf("%w: evaluation execution contract registry is unavailable", ErrInvalid)
	}
	executorID, ok := manifestExecutorIdentity(manifest)
	if !ok {
		return resolvedExecutionContract{}, fmt.Errorf("%w: evaluation manifest executor identity is invalid", ErrInvalid)
	}
	executor, ok := registry.executors[executorID]
	if !ok || executor.Mode != manifest.Mode {
		return resolvedExecutionContract{}, fmt.Errorf("%w: evaluation manifest executor is not registered for mode %q", ErrInvalid, manifest.Mode)
	}
	target, ok := registry.targets[manifest.Target.ID]
	if !ok || target.Definition.Public.Kind != manifest.Target.Kind || target.Definition.Contract.ExecutionProfile != executor.TargetProfile ||
		!executorTargetMatches(executorID, manifest.Mode, target.Definition.Public) || !manifestMatchesTargetDefinition(manifest.Target, target.Definition) {
		return resolvedExecutionContract{}, fmt.Errorf("%w: evaluation manifest target does not accept its executor", ErrInvalid)
	}
	if manifest.ConfigDigest != target.Definition.ConfigDigest {
		return resolvedExecutionContract{}, fmt.Errorf("%w: evaluation manifest config digest does not match target %q", ErrInvalid, manifest.Target.ID)
	}
	for _, trackID := range manifest.TrackIDs {
		if !containsTrack(target.Definition.Public.TrackIDs, trackID) || !containsTrack(executor.TrackIDs, trackID) {
			return resolvedExecutionContract{}, fmt.Errorf("%w: evaluation manifest target cannot execute track %q", ErrInvalid, trackID)
		}
	}
	if err := validateManifestTargetProfile(manifest, target.Definition.Contract.ExecutionProfile); err != nil {
		return resolvedExecutionContract{}, err
	}
	return resolvedExecutionContract{Executor: executor, Target: target}, nil
}

func validateManifestTargetProfile(manifest RunManifest, profile targetExecutionProfile) error {
	target := manifest.Target
	switch profile {
	case targetProfileRecorded:
		if target.RouterAPIURL != "" || target.EnvoyURL != "" || target.RouterAPIKey != nil || target.EnvoyAPIKey != nil ||
			target.AgentTaskLedger != nil || target.FaultRecoveryLedger != nil || target.HardPolicyLedger != nil || target.ProductionExperimentLedger != nil || target.BackendTopologyDigest != "" || target.Mixture != nil {
			return fmt.Errorf("%w: recorded-source target contains runtime connectivity", ErrInvalid)
		}
	case targetProfileRuntime:
		if target.EnvoyURL == "" || target.Mixture == nil ||
			!digestPattern.MatchString(target.BackendTopologyDigest) {
			return fmt.Errorf("%w: brokered-runtime target is incomplete", ErrInvalid)
		}
	default:
		return fmt.Errorf("%w: evaluation target execution profile is invalid", ErrInvalid)
	}
	return nil
}

func policySnapshotDigestForTarget(target targetDefinition, suiteRevisions map[string]string) string {
	switch target.Contract.PolicySnapshot {
	case policySnapshotFixture:
		return fixturePolicySnapshotDigest
	case policySnapshotNormalized:
		digest, err := canonicalValueDigest(map[string]any{
			"kind": "normalized-replay-policy", "suite_revisions": suiteRevisions,
		})
		if err == nil {
			return digest
		}
	case policySnapshotRuntime:
		if target.Mixture != nil {
			return target.Mixture.RecipeDigest
		}
	}
	return ""
}
