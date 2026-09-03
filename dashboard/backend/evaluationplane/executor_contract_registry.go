package evaluationplane

import "fmt"

const (
	fixtureReplayExecutorID = "fixture-replay.v1"
	momReplayExecutorID     = "mom-cohort-replay.v1"
	liveRuntimeExecutorID   = "live-runtime.v1"
)

func builtinExecutorContracts() []executorContract {
	return []executorContract{
		{
			ID: fixtureReplayExecutorID, Mode: ModeReplay, SuiteClass: executorSuiteFixture,
			TargetProfile: targetProfileRecorded, LineageProfile: lineageFixture,
			RequiresFixtureRef: true,
			TrackIDs:           append([]TrackID(nil), allTrackIDs...),
		},
		{
			ID: liveRuntimeExecutorID, Mode: ModeLive, SuiteClass: executorSuiteRuntime,
			TargetProfile: targetProfileRuntime, LineageProfile: lineageRuntime,
			TrackIDs:           []TrackID{"routing", "model_pool", "joint", "agentic", "multimodal", "preference", "safety", "capacity"},
			CaseBudgetPerSuite: true,
		},
		{
			ID: momReplayExecutorID, Mode: ModeReplay, SuiteClass: executorSuiteMoMCohort,
			TargetProfile: targetProfileRuntime, LineageProfile: lineageRuntime,
			RequiresFixtureRef: true, EvidenceLevelCeiling: "E0",
			TrackIDs: []TrackID{"routing", "model_pool", "joint"},
		},
		{
			ID: normalizedSuiteExecutorID, Mode: ModeReplay, SuiteClass: executorSuiteNormalized,
			TargetProfile: targetProfileRecorded, LineageProfile: lineageNormalized,
			NormalizedSuite: true, RecordedNormalizedSource: true, CaseBudgetPerSuite: true,
			TrackIDs: append([]TrackID(nil), allTrackIDs...),
		},
		{
			ID: normalizedSuiteLiveExecutorID, Mode: ModeLive, SuiteClass: executorSuiteNormalized,
			TargetProfile: targetProfileRuntime, LineageProfile: lineageRuntime,
			NormalizedSuite: true, CaseBudgetPerSuite: true, EvidenceLevelCeiling: "E4",
			TrackIDs: []TrackID{"routing", "model_pool", "joint", "multimodal", "capacity"},
		},
	}
}

func builtinExecutorContract(id string) (executorContract, bool) {
	for _, contract := range builtinExecutorContracts() {
		if contract.ID == id {
			return contract, true
		}
	}
	return executorContract{}, false
}

func executorIsMoMCohortReplay(contract executorContract) bool {
	return contract.Mode == ModeReplay && contract.SuiteClass == executorSuiteMoMCohort &&
		contract.TargetProfile == targetProfileRuntime
}

func manifestUsesMoMCohortReplay(manifest RunManifest) bool {
	executorID, ok := manifestExecutorIdentity(manifest)
	if !ok {
		return false
	}
	executor, ok := builtinExecutorContract(executorID)
	return ok && executorIsMoMCohortReplay(executor) && manifest.Mode == ModeReplay &&
		manifest.Target.Mixture != nil && validateManifestTargetProfile(manifest, targetProfileRuntime) == nil
}

func validateExecutorContract(contract executorContract) error {
	if !portableIDPattern.MatchString(contract.ID) ||
		(contract.Mode != ModeReplay && contract.Mode != ModeLive) ||
		!portableIDPattern.MatchString(string(contract.SuiteClass)) ||
		(contract.TargetProfile != targetProfileRecorded && contract.TargetProfile != targetProfileRuntime) ||
		(contract.LineageProfile != lineageFixture && contract.LineageProfile != lineageNormalized && contract.LineageProfile != lineageRuntime) || len(contract.TrackIDs) == 0 || !canonicalTrackOrder(contract.TrackIDs) ||
		(contract.EvidenceLevelCeiling != "" && evidenceLevelRank(contract.EvidenceLevelCeiling) < 0) {
		return fmt.Errorf("invalid evaluation executor contract %q", contract.ID)
	}
	if contract.Mode == ModeReplay && contract.TargetProfile != targetProfileRecorded && contract.SuiteClass != executorSuiteMoMCohort ||
		contract.Mode == ModeLive && contract.TargetProfile != targetProfileRuntime ||
		contract.RequiresFixtureRef != (contract.LineageProfile == lineageFixture || contract.SuiteClass == executorSuiteMoMCohort) ||
		contract.RecordedNormalizedSource != (contract.LineageProfile == lineageNormalized) ||
		contract.RecordedNormalizedSource && !contract.NormalizedSuite {
		return fmt.Errorf("inconsistent evaluation executor contract %q", contract.ID)
	}
	return nil
}

func (r *Registry) registerExecutor(contract executorContract) error {
	if err := validateExecutorContract(contract); err != nil {
		return err
	}
	if _, duplicate := r.executors[contract.ID]; duplicate {
		return fmt.Errorf("duplicate evaluation executor contract %q", contract.ID)
	}
	r.executors[contract.ID] = contract
	return nil
}

func (r *Registry) executor(id string) (executorContract, bool) {
	contract, ok := r.executors[id]
	return contract, ok
}
