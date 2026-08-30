package evaluationplane

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"time"
)

func (s *Service) execute(ctx context.Context, runID, manifestPath string) {
	defer s.workers.Done()
	defer func() { <-s.semaphore }()
	if err := s.recordWorkerEvent(runID, WorkerEvent{Type: "progress", Message: "Evaluation worker started"}); err != nil {
		s.finalizeRun(runID, err)
		return
	}
	err := s.process.Run(ctx, ProcessSpec{ManifestPath: manifestPath, StorePath: s.store.Root()}, func(event WorkerEvent) error {
		return s.recordWorkerEvent(runID, event)
	})
	if err == nil {
		if reportErr := s.validateAndAnchorReport(runID); reportErr != nil {
			err = fmt.Errorf("validate evaluation worker report: %w", reportErr)
		}
	}
	s.finalizeRun(runID, err)
}

func (s *Service) recordWorkerEvent(runID string, workerEvent WorkerEvent) error {
	workerEvent, err := sanitizeWorkerEvent(workerEvent)
	if err != nil {
		return fmt.Errorf("reject evaluation worker event: %w", err)
	}
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
	// Terminal state is server-owned. A worker terminal marker counts against
	// the protocol budget but is not persisted until the process actually exits.
	if workerEvent.Type == "completed" || workerEvent.Type == "failed" || workerEvent.Type == "cancelled" {
		return nil
	}
	if workerEvent.TrackID != "" && !containsTrack(run.TrackIDs, workerEvent.TrackID) {
		return fmt.Errorf("worker emitted unknown run track %q", workerEvent.TrackID)
	}
	if workerEvent.Progress != nil {
		if workerEvent.Progress.CurrentTrackID != "" && !containsTrack(run.TrackIDs, workerEvent.Progress.CurrentTrackID) {
			return fmt.Errorf("worker progress identified unknown run track %q", workerEvent.Progress.CurrentTrackID)
		}
		progress := normalizedProgress(*workerEvent.Progress, run.Progress.Total, workerEvent.Message)
		run.Progress = progress
		if updateErr := s.store.UpdateRun(run); updateErr != nil {
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

func normalizedProgress(progress RunProgress, total int, message string) RunProgress {
	if progress.Percent < 0 {
		progress.Percent = 0
	}
	if progress.Percent > 100 {
		progress.Percent = 100
	}
	progress.Total = total
	if progress.Completed < 0 {
		progress.Completed = 0
	}
	if progress.Completed > progress.Total {
		progress.Completed = progress.Total
	}
	progress.Message = message
	return progress
}

func (s *Service) finalizeRun(runID string, processErr error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.active, runID)
	delete(s.workerEvents, runID)
	run, err := s.store.GetRun(runID)
	if err != nil || run.Status != StatusRunning {
		return
	}
	now := time.Now().UTC()
	run.CompletedAt = &now
	eventType := "completed"
	message := "Evaluation completed"
	if processErr == nil {
		run.Status = StatusCompleted
		run.Progress = RunProgress{Percent: 100, Completed: run.Progress.Total, Total: run.Progress.Total, Message: message}
	} else if errors.Is(processErr, context.Canceled) {
		run.Status = StatusCancelled
		eventType = "cancelled"
		message = "Evaluation cancelled"
		run.Progress.Message = message
	} else if errors.Is(processErr, context.DeadlineExceeded) {
		run.Status = StatusFailed
		eventType = "failed"
		message = "Evaluation worker timed out"
		run.Error = "Evaluation worker exceeded its server-owned time limit"
		run.Progress.Message = message
	} else {
		run.Status = StatusFailed
		eventType = "failed"
		message = "Evaluation worker failed"
		run.Error = "Evaluation worker failed; inspect protected server diagnostics"
		run.Progress.Message = message
	}
	if err := s.store.UpdateRun(run); err != nil {
		return
	}
	_, _ = s.appendEventLocked(Event{RunID: runID, Type: eventType, Timestamp: now, Message: message, Progress: &run.Progress})
}

func (s *Service) RecoverInterruptedRuns() error {
	runs, err := s.store.ListRuns()
	if err != nil {
		return err
	}
	for _, run := range runs {
		if run.Status != StatusRunning {
			continue
		}
		now := time.Now().UTC()
		run.Status = StatusFailed
		run.CompletedAt = &now
		run.Error = "Dashboard restarted while the evaluation worker was running"
		run.Progress.Message = "Run interrupted by Dashboard restart"
		if err := s.store.UpdateRun(run); err != nil {
			return err
		}
		if _, err := s.appendEvent(Event{RunID: run.ID, Type: "failed", Timestamp: now, Message: run.Progress.Message, Progress: &run.Progress}); err != nil {
			return err
		}
	}
	return nil
}

func (s *Service) EventsAfter(runID, afterID string) ([]Event, error) {
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

func (s *Service) Subscribe(runID string) (<-chan Event, func(), error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, nil, fmt.Errorf("%w: evaluation service is closed", ErrConflict)
	}
	// Keep the run lookup under the service lock so DeleteRun cannot remove the
	// bundle between validation and subscriber registration.
	if _, err := s.store.GetRun(runID); err != nil {
		return nil, nil, err
	}
	if len(s.subscribers[runID]) >= maxSubscribersPerRun || s.subscriberCount >= maxSubscribersGlobal {
		return nil, nil, fmt.Errorf("%w: evaluation event subscriber capacity is exhausted", ErrConflict)
	}
	channel := make(chan Event, 256)
	if s.subscribers[runID] == nil {
		s.subscribers[runID] = make(map[chan Event]struct{})
	}
	s.subscribers[runID][channel] = struct{}{}
	s.subscriberCount++
	unsubscribe := func() {
		s.mu.Lock()
		if subscribers := s.subscribers[runID]; subscribers != nil {
			if _, subscribed := subscribers[channel]; subscribed {
				delete(subscribers, channel)
				s.subscriberCount--
			}
			if len(subscribers) == 0 {
				delete(s.subscribers, runID)
			}
		}
		s.mu.Unlock()
	}
	return channel, unsubscribe, nil
}

func (s *Service) appendEvent(event Event) (Event, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.appendEventLocked(event)
}

func (s *Service) appendEventLocked(event Event) (Event, error) {
	persisted, err := s.store.AppendEvent(event)
	if err != nil {
		return Event{}, err
	}
	for subscriber := range s.subscribers[event.RunID] {
		select {
		case subscriber <- persisted:
		default:
			close(subscriber)
			delete(s.subscribers[event.RunID], subscriber)
			s.subscriberCount--
		}
	}
	return persisted, nil
}
