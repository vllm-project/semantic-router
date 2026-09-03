package evaluationplane

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
)

func (s *Service) execute(
	ctx context.Context,
	runID, manifestPath string,
	controlledPair *controlledPairRunContext,
) {
	defer s.workers.Done()
	defer s.activity.releaseWorkerSlots(1)
	if err := s.recordWorkerEvent(runID, WorkerEvent{Type: "progress", Message: "Evaluation worker started"}); err != nil {
		s.finalizeRun(runID, err)
		return
	}
	registry, err := s.registrySnapshot()
	if err != nil {
		s.finalizeRun(runID, fmt.Errorf("resolve evaluation execution contracts: %w", err))
		return
	}
	result, err := s.process.Run(ctx, ProcessSpec{
		ManifestPath: manifestPath, StorePath: s.store.Root(), SuiteStorePath: s.registrySource.suiteStorePath,
		executionContracts:  registry.executionContracts(),
		controlledPair:      controlledPair,
		evidencePublication: s.store.withEvidenceSerialization,
	}, func(event WorkerEvent) error {
		return s.recordWorkerEvent(runID, event)
	})
	defer result.discardStagedEvidence()
	if err == nil {
		err = ctx.Err()
	}
	if err == nil {
		err = s.beginSealing(runID)
	}
	if err == nil {
		err = result.publishStagedEvidence()
	}
	if err == nil {
		err = s.attestAndAnchorExecution(runID, result.ExecutionTranscript)
	}
	if err != nil && controlledPair != nil {
		controlledPair.coordinator.abort(err)
	}
	s.finalizeRun(runID, err)
}

func (s *Service) beginSealing(runID string) error {
	paired, err := s.store.acquireControlledPairMutationBarrier(runID)
	if err != nil {
		return err
	}
	defer s.store.releaseControlledPairMutationBarrier(paired)

	s.mu.Lock()
	defer s.mu.Unlock()
	run, err := s.store.commitRunSealingWithinLifecycle(runID)
	if err != nil {
		if run.Status == StatusCancelled {
			return context.Canceled
		}
		return err
	}
	_, err = s.appendEventLocked(Event{
		RunID: runID, Type: "progress", Timestamp: time.Now().UTC(),
		Message: run.Progress.Message, Progress: &run.Progress,
	})
	return err
}

func (s *Service) attestAndAnchorExecution(runID string, transcript *brokerExecutionTranscript) error {
	return s.store.withEvidencePublication(func() error {
		attestationDigest, err := s.persistExecutionAttestationDuringPublication(runID, transcript)
		if err != nil {
			return fmt.Errorf("attest evaluation execution: %w", err)
		}
		validationErr := s.validateAndAnchorReportDuringPublication(runID)
		if validationErr == nil {
			return nil
		}
		// An unanchored attestation is not durable evidence. Roll it back in the
		// same publication transaction; an already-published anchor is left
		// intact so restart recovery can validate the completed seal.
		if attestationDigest != "" {
			if rollbackErr := s.rollbackUnanchoredExecutionAttestation(runID); rollbackErr != nil {
				return fmt.Errorf("validate evaluation worker report: %w; %w", validationErr, rollbackErr)
			}
		}
		return fmt.Errorf("validate evaluation worker report: %w", validationErr)
	})
}

func (s *Service) rollbackUnanchoredExecutionAttestation(runID string) error {
	return s.store.rollbackUnanchoredExecutionAttestation(runID)
}

func (s *Service) recordWorkerEvent(runID string, workerEvent WorkerEvent) error {
	workerEvent, err := sanitizeWorkerEvent(workerEvent)
	if err != nil {
		return fmt.Errorf("reject evaluation worker event: %w", err)
	}
	paired, err := s.store.acquireControlledPairMutationBarrier(runID)
	if err != nil {
		return err
	}
	defer s.store.releaseControlledPairMutationBarrier(paired)

	s.mu.Lock()
	defer s.mu.Unlock()
	run, err := s.store.GetRun(runID)
	if err != nil {
		return err
	}
	if run.Status != StatusRunning {
		return context.Canceled
	}
	if s.workerEvents[runID] >= maxWorkerEventsPerRun {
		return fmt.Errorf("worker exceeded the per-run event limit")
	}
	s.workerEvents[runID]++
	if validationErr := validateWorkerEventForRun(workerEvent, run); validationErr != nil {
		return validationErr
	}
	// Terminal state is server-owned. A worker terminal marker counts against
	// the protocol budget but is not persisted until the process actually exits.
	if workerEvent.Type == "completed" || workerEvent.Type == "failed" || workerEvent.Type == "cancelled" {
		return nil
	}
	if workerEvent.Progress != nil {
		progress := *workerEvent.Progress
		progress.Message = workerEvent.Message
		if updateErr := s.store.commitWorkerProgress(runID, progress); updateErr != nil {
			return updateErr
		}
		workerEvent.Progress = &progress
	}
	_, err = s.appendEventLocked(Event{
		RunID: runID, Type: workerEvent.Type, Timestamp: time.Now().UTC(),
		Message: workerEvent.Message, TrackID: workerEvent.TrackID,
		Progress: workerEvent.Progress, Payload: workerEvent.Payload,
	})
	return err
}

func validateWorkerEventForRun(event WorkerEvent, run Run) error {
	if event.Type == "track" {
		if event.TrackID == "" || event.Progress == nil {
			return fmt.Errorf("track worker event requires track_id and progress")
		}
	} else if event.TrackID != "" {
		return fmt.Errorf("worker event type %q cannot identify a track", event.Type)
	}
	if event.TrackID != "" && !containsTrack(run.TrackIDs, event.TrackID) {
		return fmt.Errorf("worker emitted unknown run track %q", event.TrackID)
	}
	if event.Progress == nil {
		return nil
	}
	progress := event.Progress
	if math.IsNaN(progress.Percent) || math.IsInf(progress.Percent, 0) ||
		progress.Percent < 0 || progress.Percent > 100 ||
		progress.Total != len(run.TrackIDs) || progress.Completed < 0 || progress.Completed > progress.Total ||
		progress.Message != strings.TrimSpace(progress.Message) || len(progress.Message) > maxWorkerMessageBytes {
		return fmt.Errorf("worker progress does not match the immutable run contract")
	}
	if progress.CurrentTrackID != "" && !containsTrack(run.TrackIDs, progress.CurrentTrackID) {
		return fmt.Errorf("worker progress identified unknown run track %q", progress.CurrentTrackID)
	}
	if event.Type == "track" && progress.CurrentTrackID != event.TrackID {
		return fmt.Errorf("track worker progress does not match track_id")
	}
	if event.Type == "completed" && (progress.Percent != 100 || progress.Completed != progress.Total || progress.CurrentTrackID != "") {
		return fmt.Errorf("completed worker progress is not terminal")
	}
	return nil
}

func (s *Service) finalizeRun(runID string, processErr error) {
	retryDelay := 10 * time.Millisecond
	var terminalCause error
	for {
		paired, barrierErr := s.store.acquireControlledPairMutationBarrier(runID)
		if barrierErr != nil {
			s.mu.Lock()
			s.recordLifecycleErrorLocked(fmt.Errorf("acquire terminal lifecycle barrier: %w", barrierErr))
			s.cleanupWorkerLocked(runID)
			s.mu.Unlock()
			return
		}
		s.mu.Lock()
		current, readErr := s.store.GetRun(runID)
		if readErr != nil {
			s.recordLifecycleErrorLocked(fmt.Errorf("read run before terminal transition: %w", readErr))
			s.cleanupWorkerLocked(runID)
			s.mu.Unlock()
			s.store.releaseControlledPairMutationBarrier(paired)
			return
		}
		var transitionErr error
		var terminalEvent Event
		terminalFailed := current.Status == StatusFailed
		if terminalStatus(current.Status) {
			terminalEvent, transitionErr = s.store.commitTerminalRunWithinLifecycle(current)
		} else if current.Status == StatusRunning || current.Status == StatusSealing {
			var terminal Run
			terminal, terminalCause = s.buildTerminalRun(current, processErr)
			terminalFailed = terminal.Status == StatusFailed
			terminalEvent, transitionErr = s.store.commitTerminalRunWithinLifecycle(terminal)
		} else {
			transitionErr = fmt.Errorf("%w: run cannot complete from %s", ErrConflict, current.Status)
		}
		if transitionErr == nil {
			transitionErr = s.store.refreshControlledPairTerminalState(runID)
		}
		if transitionErr == nil {
			s.cleanupWorkerLocked(runID)
			s.broadcastEventLocked(terminalEvent)
			s.mu.Unlock()
			s.store.releaseControlledPairMutationBarrier(paired)
			if terminalFailed && terminalCause != nil {
				// Public run state stays deliberately generic; detailed worker,
				// broker, sealing, and attestation failures belong only here.
				s.diagnosticLogger.Printf(
					"evaluationplane: execution failed run_id=%q error=%q", runID, terminalCause,
				)
			}
			return
		}
		s.mu.Unlock()
		s.store.releaseControlledPairMutationBarrier(paired)
		s.diagnosticLogger.Printf("evaluationplane: terminal lifecycle persistence retry run_id=%q error=%q", runID, transitionErr)

		timer := time.NewTimer(retryDelay)
		select {
		case <-timer.C:
			if retryDelay < time.Second {
				retryDelay *= 2
				if retryDelay > time.Second {
					retryDelay = time.Second
				}
			}
		case <-s.shutdown:
			if !timer.Stop() {
				<-timer.C
			}
			s.mu.Lock()
			s.recordLifecycleErrorLocked(fmt.Errorf("persist terminal lifecycle for run %s: %w", runID, transitionErr))
			s.cleanupWorkerLocked(runID)
			s.mu.Unlock()
			return
		}
	}
}

func (s *Service) buildTerminalRun(run Run, processErr error) (Run, error) {
	now := time.Now().UTC()
	if processErr == nil && run.Status != StatusSealing {
		processErr = fmt.Errorf("evaluation evidence was not in the sealing phase")
	}
	if processErr == nil {
		data, reportErr := s.store.ReadReport(run.ID)
		if reportErr != nil {
			processErr = fmt.Errorf("read sealed evaluation report: %w", reportErr)
		} else if report, decodeErr := decodeReportStrict(run.ID, data); decodeErr != nil {
			processErr = fmt.Errorf("read sealed evaluation report completion: %w", decodeErr)
		} else if report.Run.CompletedAt == nil {
			processErr = fmt.Errorf("sealed evaluation report omits completed_at")
		} else {
			now = report.Run.CompletedAt.UTC()
		}
	}
	run.CompletedAt = &now
	message := "Evaluation completed"
	if processErr == nil {
		run.Status = StatusCompleted
		run.Progress = RunProgress{Percent: 100, Completed: run.Progress.Total, Total: run.Progress.Total, Message: message}
	} else if errors.Is(processErr, context.Canceled) && run.Status == StatusRunning {
		run.Status = StatusCancelled
		message = "Evaluation cancelled"
		run.Progress.Message = message
	} else if errors.Is(processErr, context.DeadlineExceeded) {
		run.Status = StatusFailed
		message = "Evaluation worker timed out"
		run.Error = "Evaluation worker exceeded its server-owned time limit"
		run.Progress.Message = message
	} else {
		run.Status = StatusFailed
		message = "Evaluation worker failed"
		run.Error = "Evaluation worker failed; inspect protected server diagnostics"
		run.Progress.Message = message
	}
	return run, processErr
}

func (s *Service) cleanupWorkerLocked(runID string) {
	if cancel, ok := s.active[runID]; ok {
		cancel()
	}
	delete(s.active, runID)
	delete(s.workerEvents, runID)
	s.activity.release(runID)
}

func (s *Service) recordLifecycleErrorLocked(err error) {
	if err != nil {
		s.lifecycleErr = errors.Join(s.lifecycleErr, err)
	}
}

func (s *Service) RecoverInterruptedRuns() error {
	release, operationErr := s.beginOperation()
	if operationErr != nil {
		return operationErr
	}
	defer release()
	s.store.lifecycle.mu.Lock()
	defer s.store.lifecycle.mu.Unlock()
	runs, err := s.store.listRunsWithinLifecycle()
	if err != nil {
		return err
	}
	for index := range runs {
		run := runs[index]
		if run.Status == StatusRunning || run.Status == StatusSealing {
			if s.activity.contains(run.ID) {
				continue
			}
			if _, recoverErr := s.recoverInterruptedRun(run); recoverErr != nil {
				return recoverErr
			}
		}
	}
	return nil
}

// recoverInterruptedRun distinguishes the publication crash window from a
// genuinely interrupted worker. A completed status is reconstructed only from
// the exact server-sealed report, private receipt, anchor, and evidence set.
func (s *Service) recoverInterruptedRun(run Run) (Run, error) {
	report, reportErr := s.validateInterruptedRunSeal(run)
	if run.Status == StatusSealing && reportErr == nil {
		completedAt := report.Run.CompletedAt.UTC()
		run.Status = StatusCompleted
		run.CompletedAt = &completedAt
		run.Error = ""
		run.Progress = RunProgress{
			Percent: 100, Completed: len(run.TrackIDs), Total: len(run.TrackIDs), Message: "Evaluation completed",
		}
	} else {
		now := time.Now().UTC()
		run.Status = StatusFailed
		run.CompletedAt = &now
		run.Error = "Dashboard restarted while the evaluation worker was running"
		run.Progress.Message = "Run interrupted by Dashboard restart"
	}
	if _, err := s.store.commitTerminalRunWithinLifecycle(run); err != nil {
		return Run{}, err
	}
	if err := s.store.refreshControlledPairTerminalState(run.ID); err != nil {
		return Run{}, err
	}
	return run, nil
}

func (s *Service) validateInterruptedRunSeal(run Run) (Report, error) {
	data, err := s.store.ReadReport(run.ID)
	if err != nil {
		return Report{}, err
	}
	report, err := decodeReportStrict(run.ID, data)
	if err != nil {
		return Report{}, err
	}
	if report.Run.CompletedAt == nil {
		return Report{}, fmt.Errorf("%w: sealed evaluation report omits completed_at", ErrInvalid)
	}
	manifest, _, err := s.readDurableManifest(run.ID)
	if err != nil {
		return Report{}, err
	}
	completed := run
	completedAt := report.Run.CompletedAt.UTC()
	completed.Status = StatusCompleted
	completed.CompletedAt = &completedAt
	completed.Error = ""
	completed.Progress = RunProgress{
		Percent: 100, Completed: len(run.TrackIDs), Total: len(run.TrackIDs), Message: "Evaluation completed",
	}
	if err := validateStoredRun(completed.ID, completed); err != nil {
		return Report{}, err
	}
	if err := validateReportFrozenFields(completed, manifest, report); err != nil {
		return Report{}, err
	}
	if err := s.verifyReportAnchor(run.ID, data, report.AttestationRevision); err != nil {
		return Report{}, err
	}
	if err := s.rejectConfiguredSecretBytes(data); err != nil {
		return Report{}, err
	}
	return report, nil
}

func (s *Service) EventsAfterAs(actor Actor, runID, afterID string) ([]Event, error) {
	release, operationErr := s.beginOperation()
	if operationErr != nil {
		return nil, operationErr
	}
	defer release()
	releaseEvidence, authorizationErr := s.acquireAuthorizedEvidenceRead(actor, runID)
	if authorizationErr != nil {
		return nil, authorizationErr
	}
	defer releaseEvidence()
	var after uint64
	if afterID != "" {
		parsed, err := strconv.ParseUint(afterID, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("%w: Last-Event-ID must be numeric", ErrInvalid)
		}
		after = parsed
	}
	return s.store.EventsAfter(runID, after)
}

func (s *Service) SubscribeAs(actor Actor, runID string) (<-chan Event, func(), error) {
	release, operationErr := s.beginOperation()
	if operationErr != nil {
		return nil, nil, operationErr
	}
	defer release()
	s.store.lifecycle.mu.Lock()
	defer s.store.lifecycle.mu.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, nil, fmt.Errorf("%w: evaluation service is closed", ErrConflict)
	}
	if _, err := s.store.runForActorUnlocked(actor, runID); err != nil {
		return nil, nil, err
	}
	return s.activity.eventSubscribers.subscribe(s, runID, actor.principalDigest, func() error {
		_, err := s.store.GetRun(runID)
		return err
	})
}

func (s *Service) appendEventLocked(event Event) (Event, error) {
	persisted, err := s.store.AppendEvent(event)
	if err != nil {
		return Event{}, err
	}
	s.broadcastEventLocked(persisted)
	return persisted, nil
}

func (s *Service) broadcastEventLocked(event Event) {
	s.activity.eventSubscribers.broadcast(event)
}
