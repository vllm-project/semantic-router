package evaluationplane

import (
	"errors"
	"fmt"
	"time"
)

func (s *Service) validateCreateBaseline(
	actor Actor,
	request CreateRunRequest,
	target targetDefinition,
) error {
	if request.BaselineRunID == "" {
		return nil
	}
	baseline, err := s.store.baselineSnapshotForCreateAs(
		actor, request.BaselineRunID, request.ClientRequestID,
	)
	if err != nil {
		return err
	}
	if baseline.Status != StatusCompleted {
		return fmt.Errorf("%w: baseline run must be completed", ErrInvalid)
	}
	if err := validateComparableRunRequest(request, baseline); err != nil {
		return err
	}
	return s.validateComparableTargetSnapshot(
		request.ChangeProfile,
		target,
		baseline.ID,
	)
}

func (s *Service) newPendingRunManifest(
	registry *Registry,
	request CreateRunRequest,
	target targetDefinition,
	evidenceLevel EvidenceLevel,
) (Run, RunManifest, error) {
	// Python datetime serialization is microsecond-precise. Freeze the shared
	// timestamp at that precision so Go/Python evidence compares byte-stably.
	now := time.Now().UTC().Truncate(time.Microsecond)
	run := Run{
		SchemaVersion: SchemaVersion,
		ID:            request.ClientRequestID, ClientRequestID: request.ClientRequestID,
		Name: request.Name, Description: request.Description,
		Status: StatusPending, Mode: request.Mode, EvidenceLevel: evidenceLevel,
		TrackEvidenceLevels: initialTrackEvidenceLevels(request.TrackIDs, evidenceLevel),
		TargetID:            request.TargetID, ChangeProfile: request.ChangeProfile,
		Mixture:  copyCatalogMixture(target.Public.Mixture),
		SuiteIDs: request.SuiteIDs, TrackIDs: request.TrackIDs,
		SampleLimit: request.SampleLimit, Concurrency: request.Concurrency, Seed: request.Seed,
		CapacitySLO:          copyCapacitySLO(request.CapacitySLO),
		CapacityLoadProtocol: copyCapacityLoadProtocol(request.CapacityLoadProtocol),
		BaselineRunID:        request.BaselineRunID,
		Progress:             RunProgress{Total: len(request.TrackIDs), Message: "Run created"},
		CreatedAt:            now,
	}
	suiteRevisions := suiteRevisionSnapshot(registry, run.SuiteIDs)
	manifest := RunManifest{
		SchemaVersion: SchemaVersion, RunID: run.ID, Name: run.Name, Description: run.Description, Mode: run.Mode,
		Target: ManifestTarget{
			SchemaVersion: SchemaVersion, ID: target.Public.ID, Kind: target.Public.Kind,
			RouterAPIURL: target.RouterAPIURL, EnvoyURL: target.EnvoyURL,
			RouterAPIKey: copySecretRef(target.RouterAPIKey), EnvoyAPIKey: copySecretRef(target.EnvoyAPIKey),
			AgentTaskLedger:            copyServiceEndpoint(target.AgentTaskLedger),
			FaultRecoveryLedger:        copyServiceEndpoint(target.FaultRecoveryLedger),
			HardPolicyLedger:           copyServiceEndpoint(target.HardPolicyLedger),
			ProductionExperimentLedger: copyServiceEndpoint(target.ProductionExperimentLedger),
			Mixture:                    copyManifestMixture(target.Mixture),
			BackendTopologyDigest:      target.BackendTopologyDigest,
		},
		ChangeProfile:       run.ChangeProfile,
		GateContractVersion: GateContractVersion,
		SuiteIDs:            run.SuiteIDs,
		SuiteRevisions:      suiteRevisions,
		SuiteExecutors:      suiteExecutorSnapshot(registry, run.SuiteIDs, run.Mode),
		TrackIDs:            run.TrackIDs, SampleLimit: run.SampleLimit,
		Concurrency: run.Concurrency, Seed: run.Seed, BaselineRunID: run.BaselineRunID,
		CapacitySLO:          copyCapacitySLO(run.CapacitySLO),
		CapacityLoadProtocol: copyCapacityLoadProtocol(run.CapacityLoadProtocol),
		CreatedAt:            now, CodeRevision: s.codeRevision, ConfigDigest: target.ConfigDigest,
		PolicySnapshotDigest: policySnapshotDigestForTarget(target, suiteRevisions),
		RedactionPolicy:      "evaluation-default-v1",
	}
	manifestDigest, err := manifestSemanticDigest(manifest)
	if err != nil {
		return Run{}, RunManifest{}, fmt.Errorf(
			"%w: compute immutable evaluation manifest identity: %w",
			ErrInvalid,
			err,
		)
	}
	manifest.ManifestDigest = manifestDigest
	return run, manifest, nil
}

func initialTrackEvidenceLevels(trackIDs []TrackID, level EvidenceLevel) map[TrackID]EvidenceLevel {
	levels := make(map[TrackID]EvidenceLevel, len(trackIDs))
	for _, trackID := range trackIDs {
		levels[trackID] = level
	}
	return levels
}

// persistPendingRunAs publishes a complete bundle at the client request UUID.
// The directory publication is the sole idempotency boundary, so identity and
// durable state cannot diverge across separate persistence structures.
func (s *Service) persistPendingRunAs(
	actor Actor,
	request CreateRunRequest,
	run Run,
	manifest RunManifest,
) (Run, error) {
	if _, err := s.store.CreateBundleAs(actor, run, manifest); err != nil {
		if errors.Is(err, errRunPublicationDurabilityUncertain) {
			return Run{}, err
		}
		if !errors.Is(err, ErrConflict) {
			return Run{}, err
		}
		existing, getErr := s.resolveExistingCreate(actor, request, request.ClientRequestID)
		if getErr != nil {
			return Run{}, errors.Join(err, getErr)
		}
		return existing, nil
	}
	return run, nil
}
