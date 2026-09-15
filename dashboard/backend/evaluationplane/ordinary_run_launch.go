package evaluationplane

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"time"
)

// resumeOrdinaryRunLaunchLocked turns the durable Running snapshot into one
// live worker. Running is the launch intent: status and its start event commit
// before the process-local activity claim, so a retry can close either
// durability uncertainty without launching the worker twice.
//
// The caller holds the root lifecycle mutex and Service.mu.
func (s *Service) resumeOrdinaryRunLaunchLocked(ctx context.Context, run Run) (Run, error) {
	if run.ControlledPair != nil {
		return Run{}, fmt.Errorf("%w: controlled pair members must be started through their pair", ErrConflict)
	}
	if err := ctx.Err(); err != nil {
		return Run{}, err
	}
	if run.Status == StatusRunning {
		runDir, err := s.store.checkedRunDir(run.ID)
		if err != nil {
			return Run{}, err
		}
		if err := s.store.syncRunStatusDirectory(runDir, "evaluation run start retry"); err != nil {
			return Run{}, err
		}
		if s.activity.contains(run.ID) {
			return run, nil
		}
	} else if run.Status != StatusPending {
		return Run{}, fmt.Errorf("%w: run cannot be started from %s", ErrConflict, run.Status)
	}
	if err := s.validateRunStart(run); err != nil {
		return Run{}, err
	}
	manifestPath, manifestErr := s.store.ManifestPath(run.ID)
	if manifestErr != nil {
		return Run{}, manifestErr
	}
	releaseSlot, reserved := s.activity.reserveWorkerSlots(1)
	if !reserved {
		return Run{}, fmt.Errorf("%w: evaluation worker capacity is full", ErrConflict)
	}
	if err := ctx.Err(); err != nil {
		releaseSlot()
		return Run{}, err
	}
	if run.Status == StatusPending {
		now := time.Now().UTC()
		run.Status = StatusRunning
		run.StartedAt = &now
		run.Error = ""
		run.Progress.Message = "Evaluation worker starting"
		if err := ctx.Err(); err != nil {
			releaseSlot()
			return Run{}, err
		}
		if err := s.store.commitOrdinaryRunStart(run); err != nil {
			releaseSlot()
			return Run{}, err
		}
	}
	workerContext, cancel := context.WithTimeout(context.Background(), s.workerTimeout)
	startEvent, err := s.store.ensureOrdinaryRunStartEvent(run)
	if err != nil {
		cancel()
		releaseSlot()
		return Run{}, err
	}
	if !s.activity.claim([]string{run.ID}, []context.CancelFunc{cancel}) {
		cancel()
		releaseSlot()
		if s.activity.contains(run.ID) {
			return run, nil
		}
		return Run{}, fmt.Errorf("%w: evaluation worker launch ownership is uncertain", ErrConflict)
	}
	s.broadcastEventLocked(startEvent)
	s.active[run.ID] = cancel
	s.workerEvents[run.ID] = 0
	s.workers.Add(1)
	go s.execute(workerContext, run.ID, manifestPath, nil)
	return run, nil
}

// ensureOrdinaryRunStartEvent closes the second ordinary-run launch commit
// cut. It accepts only the immutable initial snapshot, or that snapshot plus
// the exact event derived from the durable Running intent. A visible append
// whose sync failed is therefore retried with fsync, never a duplicate append.
func (s *Store) ensureOrdinaryRunStartEvent(run Run) (Event, error) {
	if run.Status != StatusRunning || run.StartedAt == nil || run.ControlledPair != nil {
		return Event{}, fmt.Errorf("%w: ordinary run launch intent is invalid", ErrInvalid)
	}
	expected := Event{
		ID: "2", RunID: run.ID, Type: "progress", Timestamp: run.StartedAt.UTC(),
		Message: run.Progress.Message, Progress: &run.Progress,
	}
	if err := validateStoredEvent(expected); err != nil {
		return Event{}, err
	}

	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	runDir, err := s.checkedRunDir(run.ID)
	if err != nil {
		return Event{}, err
	}
	current, err := s.getRunUnlocked(run.ID)
	if err != nil {
		return Event{}, err
	}
	if current.Status != StatusRunning || current.StartedAt == nil ||
		!current.StartedAt.Equal(*run.StartedAt) || current.Progress != run.Progress {
		return Event{}, fmt.Errorf("%w: ordinary run launch intent changed", ErrConflict)
	}
	eventsPath := filepath.Join(runDir, eventsFileName)
	events, err := readOrdinaryRunLaunchEvents(eventsPath, run.ID)
	if err != nil {
		return Event{}, err
	}
	if len(events) == 1 {
		encoded, encodeErr := json.Marshal(expected)
		if encodeErr != nil {
			return Event{}, fmt.Errorf("encode evaluation start event: %w", encodeErr)
		}
		if err := s.eventPersistence.Append(eventsPath, append(encoded, '\n')); err != nil {
			return Event{}, err
		}
	} else if len(events) == 2 && ordinaryRunStartEventMatches(events[1], expected) {
		if err := s.eventPersistence.Sync(eventsPath, "evaluation run start event retry"); err != nil {
			return Event{}, fmt.Errorf("evaluation run start event durability is uncertain: %w", err)
		}
	} else {
		return Event{}, fmt.Errorf("%w: ordinary run start event history is not resumable", ErrConflict)
	}
	s.runIndex.setEventSequence(run.ID, 2)
	return expected, nil
}

func readOrdinaryRunLaunchEvents(path, runID string) ([]Event, error) {
	file, err := openBundleFile(path, os.O_RDONLY)
	if err != nil {
		return nil, fmt.Errorf("open evaluation event log: %w", err)
	}
	defer func() { _ = file.Close() }()
	info, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat evaluation event log: %w", err)
	}
	if info.Size() > maxEventLogBytes {
		return nil, fmt.Errorf("%w: evaluation event log exceeds its per-run byte limit", ErrInvalid)
	}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 4*1024), maxWorkerEventLineBytes)
	events := make([]Event, 0, 2)
	for scanner.Scan() {
		event, decodeErr := decodeStoredEvent(scanner.Bytes())
		if decodeErr != nil {
			return nil, decodeErr
		}
		sequence := len(events) + 1
		if event.RunID != runID || event.ID != strconv.Itoa(sequence) {
			return nil, fmt.Errorf("%w: evaluation event history is not a strictly monotonic run-local sequence", ErrInvalid)
		}
		events = append(events, event)
		if len(events) > 2 {
			return events, nil
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan evaluation event log: %w", err)
	}
	if len(events) == 0 || events[0].Type != "snapshot" || events[0].ID != "1" {
		return nil, fmt.Errorf("%w: ordinary run initial event is invalid", ErrInvalid)
	}
	return events, nil
}

func ordinaryRunStartEventMatches(stored, expected Event) bool {
	return stored.ID == expected.ID && stored.RunID == expected.RunID &&
		stored.Type == expected.Type && stored.Timestamp.Equal(expected.Timestamp) &&
		stored.Message == expected.Message && stored.TrackID == expected.TrackID &&
		reflect.DeepEqual(stored.Progress, expected.Progress) &&
		reflect.DeepEqual(stored.Payload, expected.Payload)
}
