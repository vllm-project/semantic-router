package evaluationplane

import (
	"fmt"
	"reflect"
)

func (s *Service) loadControlledPairSource(id string) (controlledPairSource, error) {
	run, err := s.store.GetRun(id)
	if err != nil {
		return controlledPairSource{}, err
	}
	if run.Status != StatusCompleted || run.Mode != ModeLive {
		return controlledPairSource{}, fmt.Errorf("%w: source must be a completed live run", ErrInvalid)
	}
	manifest, manifestBytes, err := s.readDurableManifest(id)
	if err != nil {
		return controlledPairSource{}, err
	}
	report, err := s.decodedReport(id)
	if err != nil {
		return controlledPairSource{}, err
	}
	anchor, err := s.store.readReportAnchor(id)
	manifestArtifactDigest, _ := digestAndSize(manifestBytes)
	if err != nil || anchor.ManifestSemanticDigest != manifest.ManifestDigest ||
		anchor.ManifestArtifactDigest != manifestArtifactDigest || anchor.ExecutionAttestationDigest == "" {
		return controlledPairSource{}, fmt.Errorf("%w: source lacks sealed server-owned live provenance", ErrInvalid)
	}
	attestation, err := s.store.readExecutionAttestationForManifest(id, manifest)
	if err != nil || attestation.Digest != anchor.ExecutionAttestationDigest ||
		attestation.ManifestDigest != manifest.ManifestDigest {
		return controlledPairSource{}, fmt.Errorf("%w: source execution attestation is unavailable", ErrInvalid)
	}
	anchorDigest, err := s.store.reportAnchorDigest(id)
	if err != nil {
		return controlledPairSource{}, fmt.Errorf("%w: source report anchor is unavailable", ErrInvalid)
	}
	return controlledPairSource{
		run: run, manifest: manifest, report: report,
		manifestArtifactDigest: manifestArtifactDigest, anchorDigest: anchorDigest,
		attestationDigest: attestation.Digest,
	}, nil
}

func (s *Service) validateControlledPairSources(baseline, candidate controlledPairSource) error {
	registry, err := s.registrySnapshot()
	if err != nil {
		return err
	}
	return validateControlledPairSourcesAgainstRegistry(baseline, candidate, s.codeRevision, registry)
}

func validateControlledPairSourcesAgainstRegistry(
	baseline controlledPairSource,
	candidate controlledPairSource,
	codeRevision string,
	registry *Registry,
) error {
	left, right := baseline.manifest, candidate.manifest
	if left.CodeRevision != codeRevision || right.CodeRevision != codeRevision {
		return fmt.Errorf("%w: both controlled pair sources must match the active evaluation worker revision", ErrConflict)
	}
	if err := validateControlledPairRegistryTargets(registry, left, right); err != nil {
		return err
	}
	if len(left.SuiteIDs) != 1 || len(right.SuiteIDs) != 1 || left.SuiteIDs[0] != right.SuiteIDs[0] {
		return fmt.Errorf("%w: controlled pair requires one exact shared campaign suite", ErrInvalid)
	}
	suite, exists := registry.suite(left.SuiteIDs[0])
	if !exists || suite.CampaignProtocol == nil ||
		left.SampleLimit < suite.CampaignProtocol.MinimumCases ||
		right.SampleLimit < suite.CampaignProtocol.MinimumCases {
		return fmt.Errorf("%w: controlled pair suite does not satisfy its declared campaign cohort protocol", ErrInvalid)
	}
	if !reflect.DeepEqual(left.SuiteRevisions, right.SuiteRevisions) ||
		!reflect.DeepEqual(left.SuiteRevisions, suiteRevisionSnapshot(registry, left.SuiteIDs)) ||
		!reflect.DeepEqual(left.SuiteExecutors, right.SuiteExecutors) ||
		!reflect.DeepEqual(left.SuiteExecutors, suiteExecutorSnapshot(registry, left.SuiteIDs, ModeLive)) ||
		!reflect.DeepEqual(left.TrackIDs, right.TrackIDs) || !reflect.DeepEqual(left.TrackIDs, suite.TrackIDs) ||
		left.SampleLimit != right.SampleLimit || left.Concurrency != right.Concurrency || left.Seed != right.Seed ||
		left.ChangeProfile != right.ChangeProfile || !reflect.DeepEqual(left.CapacitySLO, right.CapacitySLO) ||
		!reflect.DeepEqual(left.CapacityLoadProtocol, right.CapacityLoadProtocol) ||
		baseline.report.Provenance.WorkloadSnapshotDigest == "" ||
		baseline.report.Provenance.WorkloadSnapshotDigest != candidate.report.Provenance.WorkloadSnapshotDigest ||
		!reflect.DeepEqual(baseline.report.Provenance.BenchmarkRevisions, candidate.report.Provenance.BenchmarkRevisions) {
		return fmt.Errorf("%w: controlled pair sources do not share one immutable suite revision and workload", ErrInvalid)
	}
	if left.Target.ID == right.Target.ID {
		return fmt.Errorf("%w: controlled pair requires two distinct deployment-scoped targets", ErrConflict)
	}
	if left.Target.Mixture == nil || right.Target.Mixture == nil || left.Target.Mixture.ID != right.Target.Mixture.ID ||
		left.Target.Mixture.RecipeName != right.Target.Mixture.RecipeName {
		return fmt.Errorf("%w: controlled pair sources do not identify one Mixture-of-Models subject", ErrInvalid)
	}
	if err := validateControlledPairAddressability(left, right); err != nil {
		return err
	}
	return validateControlledPairTreatment(baseline.report, candidate.report, left, right)
}

func validateControlledPairAddressability(baseline, candidate RunManifest) error {
	for _, trackID := range baseline.TrackIDs {
		if !campaignTrackHasExecutionContract(trackID) {
			return fmt.Errorf("%w: controlled pair track %q has no paired broker protocol", ErrInvalid, trackID)
		}
	}
	if containsTrack(baseline.TrackIDs, "routing") {
		distinct, err := serverOriginsDistinct(baseline.Target.RouterAPIURL, candidate.Target.RouterAPIURL)
		if err != nil {
			return fmt.Errorf("%w: controlled pair Router origins: %w", ErrInvalid, err)
		}
		if !distinct {
			return fmt.Errorf(
				"%w: routing variants are not simultaneously addressable at distinct server-owned Router origins", ErrConflict,
			)
		}
	}
	for _, trackID := range baseline.TrackIDs {
		if trackID == "model_pool" || trackID == "joint" || trackID == "multimodal" || trackID == "capacity" {
			distinct, err := serverOriginsDistinct(baseline.Target.EnvoyURL, candidate.Target.EnvoyURL)
			if err != nil {
				return fmt.Errorf("%w: controlled pair Envoy origins: %w", ErrInvalid, err)
			}
			if !distinct {
				return fmt.Errorf(
					"%w: live variants are not simultaneously addressable at distinct server-owned Envoy origins", ErrConflict,
				)
			}
			break
		}
	}
	return nil
}

func validateControlledPairTreatment(
	baseline Report,
	candidate Report,
	baselineManifest RunManifest,
	candidateManifest RunManifest,
) error {
	if baseline.Run.ChangeProfile == "schema_adapter" {
		return fmt.Errorf(
			"%w: schema_adapter pairing requires two simultaneously installed worker revisions and is unavailable", ErrConflict,
		)
	}
	if baseline.Run.ChangeProfile != "runtime_capacity" && baseline.Run.ChangeProfile != "model_pool" {
		candidate.Provenance.EnvironmentSnapshotDigest = baseline.Provenance.EnvironmentSnapshotDigest
	}
	if err := validateTreatmentFactors(baseline, candidate); err != nil {
		return fmt.Errorf("%w: controlled pair treatment: %w", ErrInvalid, err)
	}
	allowed := comparisonTreatment(baseline.Run.ChangeProfile)
	if !allowed.environment && baselineManifest.Target.BackendTopologyDigest != candidateManifest.Target.BackendTopologyDigest {
		return fmt.Errorf("%w: controlled pair backend topology changed outside the declared treatment", ErrInvalid)
	}
	if baseline.Run.ChangeProfile == "runtime_capacity" &&
		baselineManifest.Target.BackendTopologyDigest == candidateManifest.Target.BackendTopologyDigest {
		return fmt.Errorf("%w: runtime_capacity requires the server-owned topology factor to change", ErrInvalid)
	}
	return nil
}
