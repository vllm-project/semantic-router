package evaluationplane

import "fmt"

func validateTargetRegistration(target targetDefinition, executors map[string]executorContract) error {
	public := target.Public
	if !portableIDPattern.MatchString(public.ID) || !portableIDPattern.MatchString(public.Kind) ||
		!digestPattern.MatchString(target.ConfigDigest) ||
		(len(public.TrackIDs) != 0 && !canonicalTrackOrder(public.TrackIDs)) ||
		len(public.Modes) == 0 || len(public.Modes) != len(public.AcceptedExecutors) ||
		(target.Contract.ExecutionProfile != targetProfileRecorded && target.Contract.ExecutionProfile != targetProfileRuntime) ||
		(target.Contract.PolicySnapshot != policySnapshotFixture && target.Contract.PolicySnapshot != policySnapshotNormalized && target.Contract.PolicySnapshot != policySnapshotRuntime) {
		return fmt.Errorf("invalid evaluation target registration %q", public.ID)
	}
	if err := validateTargetCapabilities(target); err != nil {
		return err
	}
	if target.Contract.ExecutionProfile == targetProfileRuntime && target.Contract.PolicySnapshot != policySnapshotRuntime ||
		target.Contract.ExecutionProfile == targetProfileRecorded && target.Contract.PolicySnapshot == policySnapshotRuntime {
		return fmt.Errorf("inconsistent evaluation target registration %q", public.ID)
	}
	for _, mode := range public.Modes {
		accepted, present := public.AcceptedExecutors[mode]
		if !present || len(accepted) == 0 {
			return fmt.Errorf("evaluation target %q has no executor contract for mode %q", public.ID, mode)
		}
		seen := make(map[string]struct{}, len(accepted))
		for _, executorID := range accepted {
			executor, registered := executors[executorID]
			if _, duplicate := seen[executorID]; duplicate || !registered || executor.Mode != mode ||
				executor.TargetProfile != target.Contract.ExecutionProfile {
				return fmt.Errorf("evaluation target %q declares an incompatible executor %q", public.ID, executorID)
			}
			seen[executorID] = struct{}{}
		}
	}
	return nil
}

func (r *Registry) registerTarget(target targetDefinition) error {
	target.Public.TrackIDs = availableTargetTracks(target)
	if err := validateTargetRegistration(target, r.executors); err != nil {
		return err
	}
	if _, duplicate := r.targets[target.Public.ID]; duplicate {
		return fmt.Errorf("duplicate evaluation target %q", target.Public.ID)
	}
	r.targets[target.Public.ID] = copyTargetDefinition(target)
	r.targetOrder = append(r.targetOrder, target.Public.ID)
	return nil
}
