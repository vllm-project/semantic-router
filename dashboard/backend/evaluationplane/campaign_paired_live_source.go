package evaluationplane

import "fmt"

func validateCampaignPairedLiveSources(
	baseline campaignRunEvidence,
	candidate campaignRunEvidence,
) error {
	if err := validateControlledPairedReportCohort(baseline, candidate); err != nil {
		return err
	}
	if baseline.attestation == nil || candidate.attestation == nil ||
		evidenceLevelRank(baseline.report.Run.EvidenceLevel) < evidenceLevelRank("E3") ||
		evidenceLevelRank(candidate.report.Run.EvidenceLevel) < evidenceLevelRank("E3") ||
		baseline.report.Run.TargetID == candidate.report.Run.TargetID ||
		baseline.report.Run.TargetID != baseline.attestation.TargetID ||
		candidate.report.Run.TargetID != candidate.attestation.TargetID ||
		baseline.report.Provenance.TargetID != baseline.report.Run.TargetID ||
		candidate.report.Provenance.TargetID != candidate.report.Run.TargetID ||
		!sameRunMixtureIdentity(baseline.report.Run.Mixture, candidate.report.Run.Mixture) {
		return fmt.Errorf("%w: paired live evidence is not bound to two exact deployment targets over one Mixture", ErrInvalid)
	}
	for _, trackID := range baseline.report.Run.TrackIDs {
		if !campaignTrackHasExecutionContract(trackID) {
			return fmt.Errorf("%w: paired live track %q lacks a server execution attestation contract", ErrInvalid, trackID)
		}
	}
	return nil
}

func campaignTrackHasExecutionContract(trackID TrackID) bool {
	return trackID == "routing" || trackID == "model_pool" || trackID == "joint" ||
		trackID == "multimodal" || trackID == "capacity"
}
