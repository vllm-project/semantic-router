package evaluationplane

import (
	"context"
	"fmt"
	"time"
)

type CreateControlledPairRequest struct {
	ClientRequestID      string `json:"client_request_id"`
	BaselineSourceRunID  string `json:"baseline_source_run_id"`
	CandidateSourceRunID string `json:"candidate_source_run_id"`
	BaselineRunID        string `json:"baseline_run_id"`
	CandidateRunID       string `json:"candidate_run_id"`
}

type ControlledPairExecution struct {
	SchemaVersion        string                     `json:"schema_version"`
	ContractVersion      string                     `json:"contract_version"`
	ID                   string                     `json:"id"`
	Protocol             string                     `json:"protocol"`
	BaselineSourceRunID  string                     `json:"baseline_source_run_id"`
	CandidateSourceRunID string                     `json:"candidate_source_run_id"`
	BaselineRun          Run                        `json:"baseline_run"`
	CandidateRun         Run                        `json:"candidate_run"`
	State                string                     `json:"state"`
	Capabilities         ControlledPairCapabilities `json:"capabilities"`
}

type ControlledPairCapabilities struct {
	CanCancel bool `json:"can_cancel"`
	CanDelete bool `json:"can_delete"`
}

type controlledPairSource struct {
	run                    Run
	manifest               RunManifest
	report                 Report
	manifestArtifactDigest string
	anchorDigest           string
	attestationDigest      string
}

type controlledPairLaunchPreparation struct {
	pair                 controlledPairManifest
	pairExists           bool
	alreadyStarted       bool
	baseline             controlledPairSource
	candidate            controlledPairSource
	baselineCredentials  workerBrokerCredentials
	candidateCredentials workerBrokerCredentials
}

// CreateControlledPairExecutionAs clones two completed server-owned live target
// snapshots into fresh workers and starts them behind one AB/BA coordinator.
// The request contains only durable run identities; endpoint origins and
// credentials always come from the sealed source manifests.
func (s *Service) CreateControlledPairExecutionAs(
	ctx context.Context,
	actor Actor,
	request CreateControlledPairRequest,
) (ControlledPairExecution, error) {
	releaseOperation, operationErr := s.beginOperation()
	if operationErr != nil {
		return ControlledPairExecution{}, operationErr
	}
	defer releaseOperation()
	if err := ctx.Err(); err != nil {
		return ControlledPairExecution{}, err
	}
	if err := validateActor(actor); err != nil {
		return ControlledPairExecution{}, err
	}
	if err := validateControlledPairRequest(request); err != nil {
		return ControlledPairExecution{}, err
	}
	prelaunchContext, finishPrelaunch, prelaunchErr := s.beginControlledPairPrelaunch(ctx)
	if prelaunchErr != nil {
		return ControlledPairExecution{}, prelaunchErr
	}
	defer finishPrelaunch()
	var launchDone chan struct{}
	for {
		launchOwner, done := s.acquireControlledPairLaunch(request.ClientRequestID)
		if launchOwner {
			launchDone = done
			break
		}
		select {
		case <-ctx.Done():
			return ControlledPairExecution{}, ctx.Err()
		case <-prelaunchContext.Done():
			return ControlledPairExecution{}, prelaunchContext.Err()
		case <-done:
		}
	}
	defer s.releaseControlledPairLaunch(request.ClientRequestID, launchDone)
	if err := controlledPairPrelaunchErr(ctx, prelaunchContext); err != nil {
		return ControlledPairExecution{}, err
	}
	preparation, err := s.prepareControlledPairLaunch(actor, request, ctx, prelaunchContext)
	if err != nil {
		return ControlledPairExecution{}, err
	}
	if preparation.alreadyStarted {
		return s.getControlledPairExecutionAs(actor, preparation.pair.PairID)
	}
	return s.materializeControlledPairLaunch(ctx, prelaunchContext, actor, request, preparation)
}

func (s *Service) prepareControlledPairLaunch(
	actor Actor,
	request CreateControlledPairRequest,
	ctx context.Context,
	prelaunchContext context.Context,
) (controlledPairLaunchPreparation, error) {
	s.store.lifecycle.mu.Lock()
	pair, pairExists, operationErr := s.store.prepareControlledPairRequestUnlocked(actor, request)
	s.store.lifecycle.mu.Unlock()
	if operationErr != nil {
		return controlledPairLaunchPreparation{}, operationErr
	}
	if err := controlledPairPrelaunchErr(ctx, prelaunchContext); err != nil {
		return controlledPairLaunchPreparation{}, err
	}
	preparation := controlledPairLaunchPreparation{pair: pair, pairExists: pairExists}
	if pairExists {
		switch pair.State {
		case controlledPairStateTerminal:
			preparation.alreadyStarted = true
			return preparation, nil
		case controlledPairStateRunning:
			active := s.activity.countActiveRuns(pair.BaselineRunID, pair.CandidateRunID)
			switch active {
			case 2:
				preparation.alreadyStarted = true
				return preparation, nil
			case 1:
				return controlledPairLaunchPreparation{}, fmt.Errorf(
					"%w: controlled pair has partial worker ownership", ErrConflict,
				)
			}
		case controlledPairStateDeleted, controlledPairStateDeleting:
			return controlledPairLaunchPreparation{}, fmt.Errorf("%w: controlled pair request identity is retired", ErrConflict)
		case controlledPairStatePending:
		default:
			return controlledPairLaunchPreparation{}, fmt.Errorf("%w: controlled pair recovery did not reach a launchable state", ErrConflict)
		}
	}
	preparation.baseline, preparation.candidate, operationErr = s.loadControlledPairLaunchSources(request)
	if operationErr != nil {
		return controlledPairLaunchPreparation{}, operationErr
	}
	if err := controlledPairPrelaunchErr(ctx, prelaunchContext); err != nil {
		return controlledPairLaunchPreparation{}, err
	}
	freezer, ok := s.process.(controlledPairCredentialFreezer)
	if !ok {
		return controlledPairLaunchPreparation{}, fmt.Errorf(
			"%w: controlled pairing is unavailable because the process backend cannot freeze two target credentials",
			ErrConflict,
		)
	}
	preparation.baselineCredentials, operationErr = freezer.freezeControlledPairCredentials(prelaunchContext, preparation.baseline.manifest)
	if operationErr != nil {
		return controlledPairLaunchPreparation{}, fmt.Errorf("%w: baseline target capability is unavailable: %w", ErrConflict, operationErr)
	}
	preparation.candidateCredentials, operationErr = freezer.freezeControlledPairCredentials(prelaunchContext, preparation.candidate.manifest)
	if operationErr != nil {
		return controlledPairLaunchPreparation{}, fmt.Errorf("%w: candidate target capability is unavailable: %w", ErrConflict, operationErr)
	}
	if err := controlledPairPrelaunchErr(ctx, prelaunchContext); err != nil {
		return controlledPairLaunchPreparation{}, err
	}
	if ledgerErr := s.requireCompleteRunLedger(); ledgerErr != nil {
		return controlledPairLaunchPreparation{}, ledgerErr
	}
	if err := controlledPairPrelaunchErr(ctx, prelaunchContext); err != nil {
		return controlledPairLaunchPreparation{}, err
	}
	return preparation, nil
}

// loadControlledPairLaunchSources holds the evidence read lease only while it
// verifies the two sealed source bundles. Later lifecycle publication takes
// the write side and revalidates the exact source digests before committing,
// so retaining a read lease across publication would add no safety and would
// deadlock the non-reentrant root evidence lock.
func (s *Service) loadControlledPairLaunchSources(
	request CreateControlledPairRequest,
) (controlledPairSource, controlledPairSource, error) {
	release, err := s.acquireEvidenceRead()
	if err != nil {
		return controlledPairSource{}, controlledPairSource{}, err
	}
	defer release()
	baseline, err := s.loadControlledPairSource(request.BaselineSourceRunID)
	if err != nil {
		return controlledPairSource{}, controlledPairSource{}, fmt.Errorf(
			"baseline controlled-pair source: %w", err,
		)
	}
	candidate, err := s.loadControlledPairSource(request.CandidateSourceRunID)
	if err != nil {
		return controlledPairSource{}, controlledPairSource{}, fmt.Errorf(
			"candidate controlled-pair source: %w", err,
		)
	}
	if err := s.validateControlledPairSources(baseline, candidate); err != nil {
		return controlledPairSource{}, controlledPairSource{}, err
	}
	return baseline, candidate, nil
}

func (s *Service) materializeControlledPairLaunch(
	ctx context.Context,
	prelaunchContext context.Context,
	actor Actor,
	request CreateControlledPairRequest,
	preparation controlledPairLaunchPreparation,
) (ControlledPairExecution, error) {
	pair := preparation.pair
	var baselineManifest, candidateManifest RunManifest
	if !preparation.pairExists {
		baselineCreatedAt := time.Now().UTC().Truncate(time.Microsecond)
		candidateCreatedAt := baselineCreatedAt.Add(time.Microsecond)
		baselineRun, manifest, err := cloneControlledPairRun(
			preparation.baseline, request.BaselineRunID, "", controlledPairRoleBaseline, baselineCreatedAt,
		)
		if err != nil {
			return ControlledPairExecution{}, err
		}
		baselineManifest = manifest
		candidateRun, manifest, err := cloneControlledPairRun(
			preparation.candidate, request.CandidateRunID, request.BaselineRunID, controlledPairRoleCandidate, candidateCreatedAt,
		)
		if err != nil {
			return ControlledPairExecution{}, err
		}
		candidateManifest = manifest
		pair, err = newControlledPairManifest(
			actor, request, preparation.baseline, preparation.candidate,
			baselineRun, candidateRun, baselineManifest, candidateManifest,
		)
		if err != nil {
			return ControlledPairExecution{}, err
		}
	}
	if err := controlledPairPrelaunchErr(ctx, prelaunchContext); err != nil {
		return ControlledPairExecution{}, err
	}
	releaseSlots, reserveErr := s.reserveControlledPairWorkerSlots(ctx)
	if reserveErr != nil {
		return ControlledPairExecution{}, reserveErr
	}
	slotsTransferred := false
	defer func() {
		if !slotsTransferred {
			releaseSlots()
		}
	}()
	if !preparation.pairExists {
		if err := controlledPairPrelaunchErr(ctx, prelaunchContext); err != nil {
			return ControlledPairExecution{}, err
		}
		var err error
		pair, err = s.persistControlledPairRunsAs(actor, pair, baselineManifest, candidateManifest)
		if err != nil {
			return ControlledPairExecution{}, err
		}
	}
	if err := controlledPairPrelaunchErr(ctx, prelaunchContext); err != nil {
		return ControlledPairExecution{}, err
	}
	if pair.State == controlledPairStateTerminal {
		return s.getControlledPairExecutionAs(actor, pair.PairID)
	}
	baselineManifest, _, manifestErr := s.readDurableManifest(pair.BaselineRunID)
	if manifestErr != nil {
		return ControlledPairExecution{}, manifestErr
	}
	candidateManifest, _, manifestErr = s.readDurableManifest(pair.CandidateRunID)
	if manifestErr != nil {
		return ControlledPairExecution{}, manifestErr
	}
	if err := controlledPairPrelaunchErr(ctx, prelaunchContext); err != nil {
		return ControlledPairExecution{}, err
	}

	coordinator := newControlledPairCoordinator(
		request.ClientRequestID, candidateManifest.Seed, baselineManifest, candidateManifest,
	)
	baselineContext := &controlledPairRunContext{
		role: controlledPairRoleBaseline, coordinator: coordinator, credentials: preparation.baselineCredentials,
	}
	candidateContext := &controlledPairRunContext{
		role: controlledPairRoleCandidate, coordinator: coordinator, credentials: preparation.candidateCredentials,
	}
	_, _, launched, err := s.startControlledPairRunsAs(
		ctx, actor, pair.PairID, baselineContext, candidateContext,
	)
	if err != nil {
		if launched {
			slotsTransferred = true
		} else {
			coordinator.abort(err)
		}
		return ControlledPairExecution{}, err
	}
	if launched {
		slotsTransferred = true
	}
	return s.getControlledPairExecutionAs(actor, pair.PairID)
}
