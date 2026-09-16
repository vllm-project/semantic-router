package evaluationplane

import (
	"fmt"
	"time"
)

func (s *Service) GetRunAs(actor Actor, id string) (Run, error) {
	release, err := s.beginOperation()
	if err != nil {
		return Run{}, err
	}
	defer release()
	s.store.lifecycle.mu.Lock()
	defer s.store.lifecycle.mu.Unlock()
	return s.store.runForActorUnlocked(actor, id)
}

func (s *Service) CancelRunAs(actor Actor, id string) (Run, error) {
	release, err := s.beginOperation()
	if err != nil {
		return Run{}, err
	}
	defer release()
	s.store.lifecycle.mu.Lock()
	defer s.store.lifecycle.mu.Unlock()
	if err := s.store.authorizeRunActionUnlocked(actor, id, "cancel"); err != nil {
		return Run{}, err
	}
	return s.cancelRunInternal(id)
}

func (s *Service) cancelRunInternal(id string) (Run, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	run, err := s.store.GetRun(id)
	if err != nil {
		return Run{}, err
	}
	if terminalStatus(run.Status) {
		terminalEvent, eventErr := s.store.commitTerminalRun(run)
		if eventErr != nil {
			return Run{}, eventErr
		}
		if run.Status == StatusCancelled {
			s.activity.requestCancel(id)
		}
		s.broadcastEventLocked(terminalEvent)
		return run, nil
	}
	if run.Status != StatusRunning {
		return Run{}, fmt.Errorf("%w: run cannot be cancelled from %s", ErrConflict, run.Status)
	}
	now := time.Now().UTC()
	run.Status = StatusCancelled
	run.CompletedAt = &now
	run.Progress.Message = "Run cancelled"
	terminalEvent, err := s.store.commitTerminalRun(run)
	if err != nil {
		return Run{}, err
	}
	durable, err := s.store.GetRun(id)
	if err != nil {
		return Run{}, err
	}
	if durable.Status == StatusCancelled {
		s.activity.requestCancel(id)
	}
	s.broadcastEventLocked(terminalEvent)
	return durable, nil
}

func (s *Service) DeleteRunAs(actor Actor, id string) error {
	release, err := s.beginOperation()
	if err != nil {
		return err
	}
	defer release()
	s.store.lifecycle.mu.Lock()
	defer s.store.lifecycle.mu.Unlock()
	s.mu.Lock()
	if resumed, err := s.store.resumeRunDeletionAsUnlocked(actor, id); resumed || err != nil {
		if err == nil {
			s.cleanupDeletedRunSubscribersLocked(id)
		}
		s.mu.Unlock()
		return err
	}
	s.mu.Unlock()
	if err := s.store.authorizeRunActionUnlocked(actor, id, "delete"); err != nil {
		return err
	}
	return s.deleteRunInternal(actor, id)
}

func (s *Service) deleteRunInternal(actor Actor, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	run, err := s.store.GetRun(id)
	if err != nil {
		return err
	}
	if s.activity.contains(id) {
		return fmt.Errorf("%w: evaluation worker is still exiting", ErrConflict)
	}
	if run.Status == StatusRunning || run.Status == StatusSealing {
		return fmt.Errorf("%w: evaluation execution is still active", ErrConflict)
	}
	if err := s.store.deleteRunAuthorizedUnlocked(actor, id); err != nil {
		return err
	}
	s.cleanupDeletedRunSubscribersLocked(id)
	return nil
}

func (s *Service) cleanupDeletedRunSubscribersLocked(id string) {
	s.activity.eventSubscribers.closeRun(id)
}

func terminalStatus(status RunStatus) bool {
	return status == StatusCompleted || status == StatusFailed || status == StatusCancelled
}
