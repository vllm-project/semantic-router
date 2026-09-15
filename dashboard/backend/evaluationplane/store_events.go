package evaluationplane

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
)

func (s *Store) AppendEvent(event Event) (Event, error) {
	if err := validateResourceID(event.RunID); err != nil {
		return Event{}, err
	}
	if terminalWorkerEventType(event.Type) {
		return Event{}, fmt.Errorf("%w: terminal events are derived from durable run status", ErrInvalid)
	}
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	runDir, runDirErr := s.checkedRunDir(event.RunID)
	if runDirErr != nil {
		return Event{}, runDirErr
	}
	run, runErr := s.getRunUnlocked(event.RunID)
	if runErr != nil {
		return Event{}, runErr
	}
	if terminalStatus(run.Status) {
		return Event{}, fmt.Errorf("%w: control events cannot follow terminal run status", ErrConflict)
	}
	sequence, loaded := s.runIndex.eventSequence(event.RunID)
	if !loaded {
		var sequenceErr error
		sequence, sequenceErr = lastEventSequence(filepath.Join(runDir, eventsFileName), event.RunID)
		if sequenceErr != nil {
			return Event{}, sequenceErr
		}
	}
	sequence++
	if sequence >= maxEventsPerRun {
		return Event{}, fmt.Errorf("%w: evaluation event log reached its per-run limit", ErrInvalid)
	}
	event.ID = strconv.FormatUint(sequence, 10)
	if validationErr := validateStoredEvent(event); validationErr != nil {
		return Event{}, validationErr
	}
	encoded, encodeErr := json.Marshal(event)
	if encodeErr != nil {
		return Event{}, fmt.Errorf("encode evaluation event: %w", encodeErr)
	}
	if err := s.eventPersistence.Append(
		filepath.Join(runDir, eventsFileName), append(encoded, '\n'),
	); err != nil {
		return Event{}, err
	}
	s.runIndex.setEventSequence(event.RunID, sequence)
	return event, nil
}

func (s *Store) EventsAfter(id string, after uint64) ([]Event, error) {
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	runDir, runDirErr := s.checkedRunDir(id)
	if runDirErr != nil {
		return nil, runDirErr
	}
	file, openErr := openBundleFile(filepath.Join(runDir, eventsFileName), os.O_RDONLY)
	if openErr != nil {
		return nil, fmt.Errorf("open evaluation event log: %w", openErr)
	}
	defer func() { _ = file.Close() }()
	visibleLimit, limited, limitErr := s.controlledPairEventLimit(id, runDir)
	if limitErr != nil {
		return nil, limitErr
	}
	if !limited {
		info, statErr := file.Stat()
		if statErr != nil {
			return nil, fmt.Errorf("stat evaluation event log: %w", statErr)
		}
		if info.Size() > maxEventLogBytes {
			return nil, fmt.Errorf("%w: evaluation event log exceeds its per-run byte limit", ErrInvalid)
		}
	}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 4*1024), maxWorkerEventLineBytes)
	var events []Event
	var scanned uint64
	for !limited || scanned < visibleLimit {
		if !scanner.Scan() {
			break
		}
		scanned++
		if scanned > maxEventsPerRun {
			return nil, fmt.Errorf("%w: evaluation event log exceeds its per-run event limit", ErrInvalid)
		}
		event, decodeErr := decodeStoredEvent(scanner.Bytes())
		if decodeErr != nil {
			return nil, decodeErr
		}
		if terminalWorkerEventType(event.Type) {
			return nil, fmt.Errorf("%w: control event log contains a derived terminal event", ErrInvalid)
		}
		sequence, sequenceErr := strconv.ParseUint(event.ID, 10, 64)
		if sequenceErr != nil {
			return nil, fmt.Errorf("decode evaluation event id: %w", sequenceErr)
		}
		if event.RunID != id || sequence != scanned || event.ID != strconv.FormatUint(scanned, 10) {
			return nil, fmt.Errorf("evaluation event history is not a strictly monotonic run-local sequence")
		}
		if sequence > after {
			events = append(events, event)
		}
	}
	if scanErr := scanner.Err(); scanErr != nil {
		return nil, fmt.Errorf("scan evaluation event log: %w", scanErr)
	}
	run, runErr := s.getRunUnlocked(id)
	if runErr != nil {
		return nil, runErr
	}
	if terminalStatus(run.Status) {
		terminal, terminalErr := terminalEventForRun(run, scanned+1)
		if terminalErr != nil {
			return nil, terminalErr
		}
		if scanned+1 > after {
			events = append(events, terminal)
		}
	}
	return events, nil
}
