package evaluationplane

import (
	"fmt"
	"reflect"
)

func (s *Service) resolveExistingCreate(
	actor Actor,
	request CreateRunRequest,
	runID string,
) (Run, error) {
	// Keep authorization, identity validation, and the durability retry under
	// the same lifecycle/evidence lock order used by deletion. An ID cannot be
	// deleted and rebound to another owner between those decisions.
	s.store.lifecycle.mu.Lock()
	defer s.store.lifecycle.mu.Unlock()
	s.store.lifecycle.evidenceMu.Lock()
	defer s.store.lifecycle.evidenceMu.Unlock()
	s.store.runIndex.coordinator.Lock()
	defer s.store.runIndex.coordinator.Unlock()
	run, err := s.store.getRunForCreateRetry(runID)
	if err != nil {
		return Run{}, err
	}
	if err := s.store.auditExistingCreateUnlocked(actor, run); err != nil {
		return Run{}, err
	}
	if err := s.validateManifestForCreateRetry(run); err != nil {
		return Run{}, fmt.Errorf("%w: existing client_request_id bundle is invalid", ErrConflict)
	}
	if run.ID != request.ClientRequestID ||
		run.ClientRequestID != request.ClientRequestID || !createRequestMatchesRun(request, run) {
		return Run{}, fmt.Errorf("%w: client_request_id was already used for a different evaluation run", ErrConflict)
	}
	// A previous create can observe a complete bundle after rename while the
	// runs parent fsync reports failure. Never turn that visible-but-uncertain
	// state into an idempotent success until this retry closes the same boundary.
	if err := s.store.resolveRunPublicationDurabilityUnlocked(actor, run); err != nil {
		return Run{}, err
	}
	return run, nil
}

// validateManifestForCreateRetry validates only the immutable files already
// visible in a pending publication. General report/execution readers must use
// readDurableManifest, which rejects the undecided namespace projection.
func (s *Service) validateManifestForCreateRetry(run Run) error {
	path, err := s.store.ManifestPath(run.ID)
	if err != nil {
		return err
	}
	manifest, _, err := readRunManifestStrict(path)
	if err != nil {
		return err
	}
	if manifest.RunID != run.ID {
		return fmt.Errorf("run manifest identity mismatch")
	}
	return validateRunManifestFrozenFields(run, manifest)
}

func createRequestMatchesRun(request CreateRunRequest, run Run) bool {
	return request.Name == run.Name && request.Description == run.Description &&
		request.Mode == run.Mode && request.TargetID == run.TargetID &&
		request.ChangeProfile == run.ChangeProfile && request.SampleLimit == run.SampleLimit &&
		request.Concurrency == run.Concurrency && request.Seed == run.Seed &&
		reflect.DeepEqual(request.CapacitySLO, run.CapacitySLO) &&
		reflect.DeepEqual(request.CapacityLoadProtocol, run.CapacityLoadProtocol) &&
		request.BaselineRunID == run.BaselineRunID &&
		reflect.DeepEqual(request.SuiteIDs, run.SuiteIDs) && reflect.DeepEqual(request.TrackIDs, run.TrackIDs)
}
