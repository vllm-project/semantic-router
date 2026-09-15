package evaluationplane

import "fmt"

func reduceCampaignPairedLiveEvidence(
	baseline campaignRunEvidence,
	candidate campaignRunEvidence,
	cohort campaignPairedCohort,
	sessionID string,
) (*CampaignPairedLiveEvidence, error) {
	statistics := campaignPairedStatistics(cohort, baseline.report.Run.TrackIDs, candidate.report.Run.Seed)
	armReliability := campaignArmReliabilityStatistics(cohort.poolCases, candidate.report.Run.Seed)
	promotionStatistics := campaignG3PromotionStatistics(cohort, candidate.report.Run.Seed)
	attestedBaseline, attestedCandidate := baseline.attestation, candidate.attestation
	if baseline.report.Run.Mixture == nil || candidate.report.Run.Mixture == nil ||
		!sameRunMixtureIdentity(baseline.report.Run.Mixture, candidate.report.Run.Mixture) {
		return nil, fmt.Errorf("%w: paired-live evidence does not identify one logical Mixture", ErrInvalid)
	}
	evidence := &CampaignPairedLiveEvidence{
		SchemaVersion: SchemaVersion, ContractVersion: CampaignPairedLiveContractVersion,
		ControlledPairSessionID: sessionID, ControlledPairProtocol: controlledPairInterleaveABBA,
		BaselineRunID: baseline.report.Run.ID, CandidateRunID: candidate.report.Run.ID,
		CandidateSubjectDigest: candidate.anchor.CandidateSubjectDigest,
		BaselineTargetID:       baseline.report.Run.TargetID,
		CandidateTargetID:      candidate.report.Run.TargetID,
		MixtureID:              candidate.report.Run.Mixture.ID,
		RecipeName:             candidate.report.Run.Mixture.RecipeName,
		TrackIDs:               append([]TrackID(nil), baseline.report.Run.TrackIDs...),
		WorkloadSnapshotDigest: baseline.report.Provenance.WorkloadSnapshotDigest,
		BenchmarkRevisions:     copyCampaignRevisionMap(baseline.report.Provenance.BenchmarkRevisions),
		Seed:                   candidate.report.Run.Seed, BootstrapSamples: campaignPairedBootstrapSamples,
		ConfidenceLevel:                     campaignPairedConfidenceLevel,
		PromotionPolicy:                     frozenCampaignG3PromotionPolicy,
		PromotionStatistics:                 promotionStatistics,
		BaselineManifestDigest:              baseline.anchor.ManifestSemanticDigest,
		CandidateManifestDigest:             candidate.anchor.ManifestSemanticDigest,
		BaselineExecutionAttestationDigest:  baseline.anchor.ExecutionAttestationDigest,
		CandidateExecutionAttestationDigest: candidate.anchor.ExecutionAttestationDigest,
		BaselinePolicySnapshotDigest:        attestedBaseline.PolicySnapshotDigest,
		CandidatePolicySnapshotDigest:       attestedCandidate.PolicySnapshotDigest,
		BaselineBindingSnapshotDigest:       baseline.report.Provenance.BindingSnapshotDigest,
		CandidateBindingSnapshotDigest:      candidate.report.Provenance.BindingSnapshotDigest,
		BaselinePoolSnapshotDigest:          baseline.report.Provenance.PoolSnapshotDigest,
		CandidatePoolSnapshotDigest:         candidate.report.Provenance.PoolSnapshotDigest,
		BaselineEnvironmentSnapshotDigest:   baseline.report.Provenance.EnvironmentSnapshotDigest,
		CandidateEnvironmentSnapshotDigest:  candidate.report.Provenance.EnvironmentSnapshotDigest,
		BaselineBackendTopologyDigest:       attestedBaseline.BackendTopologyDigest,
		CandidateBackendTopologyDigest:      attestedCandidate.BackendTopologyDigest,
		BaselineCodeRevision:                baseline.report.Provenance.CodeRevision,
		CandidateCodeRevision:               candidate.report.Provenance.CodeRevision,
		Statistics:                          statistics,
		ModelPoolArmReliability:             armReliability,
	}
	var err error
	evidence.Digest, err = campaignPairedLiveEvidenceDigest(*evidence)
	if err != nil {
		return nil, err
	}
	return evidence, nil
}

func campaignPairedLiveEvidenceDigest(evidence CampaignPairedLiveEvidence) (string, error) {
	evidence.Digest = ""
	digest, err := canonicalValueDigest(evidence)
	if err != nil {
		return "", fmt.Errorf("digest campaign paired-live evidence: %w", err)
	}
	return digest, nil
}

func copyCampaignRevisionMap(values map[string]string) map[string]string {
	copied := make(map[string]string, len(values))
	for key, value := range values {
		copied[key] = value
	}
	return copied
}
