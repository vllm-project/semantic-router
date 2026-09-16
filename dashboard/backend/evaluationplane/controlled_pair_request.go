package evaluationplane

import (
	"fmt"
	"time"
)

func newControlledPairManifest(
	actor Actor,
	request CreateControlledPairRequest,
	baselineSource controlledPairSource,
	candidateSource controlledPairSource,
	baselineRun Run,
	candidateRun Run,
	baselineManifest RunManifest,
	candidateManifest RunManifest,
) (controlledPairManifest, error) {
	baselineRun.ControlledPair = &ControlledPairRunMembership{
		PairID: request.ClientRequestID, Role: controlledPairRoleBaseline,
	}
	candidateRun.ControlledPair = &ControlledPairRunMembership{
		PairID: request.ClientRequestID, Role: controlledPairRoleCandidate,
	}
	cohortDigest, treatmentDigest, err := controlledPairCohortTreatmentDigests(baselineSource, candidateSource)
	if err != nil {
		return controlledPairManifest{}, err
	}
	pair := controlledPairManifest{
		SchemaVersion: SchemaVersion, ContractVersion: controlledPairProtocolVersion,
		PairID: request.ClientRequestID, ClientRequestID: request.ClientRequestID,
		Protocol: controlledPairInterleaveABBA, OwnerPrincipalDigest: actor.principalDigest,
		BaselineSourceRunID:  request.BaselineSourceRunID,
		CandidateSourceRunID: request.CandidateSourceRunID,
		BaselineRunID:        request.BaselineRunID, CandidateRunID: request.CandidateRunID,
		BaselineRole: controlledPairRoleBaseline, CandidateRole: controlledPairRoleCandidate,
		BaselineSourceManifestSemanticDigest:  baselineSource.manifest.ManifestDigest,
		CandidateSourceManifestSemanticDigest: candidateSource.manifest.ManifestDigest,
		BaselineSourceManifestArtifactDigest:  baselineSource.manifestArtifactDigest,
		CandidateSourceManifestArtifactDigest: candidateSource.manifestArtifactDigest,
		BaselineSourceAnchorDigest:            baselineSource.anchorDigest,
		CandidateSourceAnchorDigest:           candidateSource.anchorDigest,
		BaselineSourceAttestationDigest:       baselineSource.attestationDigest,
		CandidateSourceAttestationDigest:      candidateSource.attestationDigest,
		BaselineMemberManifestDigest:          baselineManifest.ManifestDigest,
		CandidateMemberManifestDigest:         candidateManifest.ManifestDigest,
		CohortDigest:                          cohortDigest, TreatmentDigest: treatmentDigest,
		State: controlledPairStatePending, BaselineRun: baselineRun, CandidateRun: candidateRun,
		CreatedAt: baselineRun.CreatedAt,
	}
	if err := validateControlledPairManifest(pair); err != nil {
		return controlledPairManifest{}, err
	}
	return pair, nil
}

func controlledPairCohortTreatmentDigests(
	baselineSource controlledPairSource,
	candidateSource controlledPairSource,
) (string, string, error) {
	cohortDigest, err := canonicalValueDigest(map[string]any{
		"suite_ids":                baselineSource.manifest.SuiteIDs,
		"suite_revisions":          baselineSource.manifest.SuiteRevisions,
		"suite_executors":          baselineSource.manifest.SuiteExecutors,
		"track_ids":                baselineSource.manifest.TrackIDs,
		"sample_limit":             baselineSource.manifest.SampleLimit,
		"concurrency":              baselineSource.manifest.Concurrency,
		"seed":                     baselineSource.manifest.Seed,
		"workload_snapshot_digest": baselineSource.report.Provenance.WorkloadSnapshotDigest,
		"benchmark_revisions":      baselineSource.report.Provenance.BenchmarkRevisions,
	})
	if err != nil {
		return "", "", fmt.Errorf("seal controlled pair cohort: %w", err)
	}
	treatmentDigest, err := canonicalValueDigest(map[string]any{
		"change_profile":   baselineSource.run.ChangeProfile,
		"baseline_target":  baselineSource.manifest.Target,
		"candidate_target": candidateSource.manifest.Target,
		"baseline_policy":  baselineSource.manifest.PolicySnapshotDigest,
		"candidate_policy": candidateSource.manifest.PolicySnapshotDigest,
	})
	if err != nil {
		return "", "", fmt.Errorf("seal controlled pair treatment: %w", err)
	}
	return cohortDigest, treatmentDigest, nil
}

func validateControlledPairRequest(request CreateControlledPairRequest) error {
	ids := []string{
		request.ClientRequestID, request.BaselineSourceRunID, request.CandidateSourceRunID,
		request.BaselineRunID, request.CandidateRunID,
	}
	seen := make(map[string]bool, len(ids))
	for _, id := range ids {
		if !validClientRequestID(id) {
			return fmt.Errorf("%w: controlled pair identities must be canonical UUIDs", ErrInvalid)
		}
		if seen[id] {
			return fmt.Errorf("%w: controlled pair identities must be distinct", ErrInvalid)
		}
		seen[id] = true
	}
	return nil
}

func cloneControlledPairRun(
	source controlledPairSource,
	runID, baselineRunID, role string,
	createdAt time.Time,
) (Run, RunManifest, error) {
	run := source.run
	run.ID, run.ClientRequestID = runID, runID
	run.Name = "Controlled pair " + role
	run.Description = "Server-owned " + controlledPairInterleaveABBA + " execution"
	run.Status, run.BaselineRunID = StatusPending, baselineRunID
	run.Progress = RunProgress{Total: len(run.TrackIDs), Message: "Run created"}
	run.CreatedAt, run.StartedAt, run.CompletedAt, run.Error = createdAt, nil, nil, ""

	manifest := source.manifest
	manifest.RunID, manifest.Name, manifest.Description = runID, run.Name, run.Description
	manifest.BaselineRunID, manifest.CreatedAt = baselineRunID, createdAt
	manifest.ManifestDigest = ""
	digest, err := manifestSemanticDigest(manifest)
	if err != nil {
		return Run{}, RunManifest{}, fmt.Errorf("%w: seal controlled pair manifest: %w", ErrInvalid, err)
	}
	manifest.ManifestDigest = digest
	return run, manifest, nil
}

func (s *Service) persistControlledPairRunsAs(
	actor Actor,
	pair controlledPairManifest,
	baselineManifest RunManifest,
	candidateManifest RunManifest,
) (controlledPairManifest, error) {
	return s.store.createControlledPairBundlesAs(actor, pair, baselineManifest, candidateManifest)
}
