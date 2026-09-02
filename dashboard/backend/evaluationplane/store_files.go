package evaluationplane

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unicode/utf8"
)

func writeJSONAtomic(path string, value any) error {
	encoded, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("encode evaluation bundle: %w", err)
	}
	encoded = append(encoded, '\n')
	dir := filepath.Dir(path)
	temp, err := os.CreateTemp(dir, ".tmp-evaluation-*")
	if err != nil {
		return fmt.Errorf("stage evaluation bundle: %w", err)
	}
	tempName := temp.Name()
	defer func() { _ = os.Remove(tempName) }()
	if err := temp.Chmod(0o600); err != nil {
		_ = temp.Close()
		return fmt.Errorf("protect staged evaluation bundle: %w", err)
	}
	if _, err := temp.Write(encoded); err != nil {
		_ = temp.Close()
		return fmt.Errorf("write staged evaluation bundle: %w", err)
	}
	if err := temp.Sync(); err != nil {
		_ = temp.Close()
		return fmt.Errorf("sync staged evaluation bundle: %w", err)
	}
	if err := temp.Close(); err != nil {
		return fmt.Errorf("close staged evaluation bundle: %w", err)
	}
	if err := os.Rename(tempName, path); err != nil {
		return fmt.Errorf("publish evaluation bundle: %w", err)
	}
	if err := syncEvaluationDirectory(dir, "evaluation bundle"); err != nil {
		return err
	}
	return nil
}

func readJSON(path string, destination any) error {
	data, err := readBundleFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("%w: evaluation bundle file", ErrNotFound)
		}
		return fmt.Errorf("read evaluation bundle: %w", err)
	}
	if err := rejectDuplicateJSONKeys(data); err != nil {
		return fmt.Errorf("decode evaluation bundle: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return fmt.Errorf("decode evaluation bundle: %w", err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return err
	}
	return nil
}

func openBundleFile(path string, flags int) (*os.File, error) {
	before, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !before.Mode().IsRegular() || before.Mode()&os.ModeSymlink != 0 {
		return nil, fmt.Errorf("evaluation bundle file is not a regular file")
	}
	if before.Mode().Perm() != 0o600 {
		return nil, fmt.Errorf("evaluation bundle file must have mode 0600")
	}
	file, err := os.OpenFile(path, flags, 0)
	if err != nil {
		return nil, err
	}
	after, err := file.Stat()
	if err != nil || !after.Mode().IsRegular() || !os.SameFile(before, after) {
		_ = file.Close()
		return nil, fmt.Errorf("evaluation bundle file changed while opening")
	}
	return file, nil
}

func readBundleFile(path string) ([]byte, error) {
	return readEvidenceBytes(path, maxStructuredArtifactBytes)
}

func requirePrivateDirectory(path string) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("evaluation store path is not a directory")
	}
	if info.Mode().Perm() != 0o700 {
		return fmt.Errorf("evaluation store directory %s must have mode 0700", filepath.Base(path))
	}
	return nil
}

func ensureJSONEOF(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("evaluation bundle contains trailing JSON")
		}
		return fmt.Errorf("decode evaluation bundle: %w", err)
	}
	return nil
}

func lastEventSequence(path, runID string) (uint64, error) {
	file, err := openBundleFile(path, os.O_RDONLY)
	if err != nil {
		return 0, fmt.Errorf("open evaluation event log: %w", err)
	}
	defer func() { _ = file.Close() }()
	info, err := file.Stat()
	if err != nil {
		return 0, fmt.Errorf("stat evaluation event log: %w", err)
	}
	if info.Size() > maxEventLogBytes {
		return 0, fmt.Errorf("%w: evaluation event log exceeds its per-run byte limit", ErrInvalid)
	}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 4*1024), maxWorkerEventLineBytes)
	var last uint64
	var count uint64
	for scanner.Scan() {
		count++
		if count > maxEventsPerRun {
			return 0, fmt.Errorf("%w: evaluation event log exceeds its per-run event limit", ErrInvalid)
		}
		event, err := decodeStoredEvent(scanner.Bytes())
		if err != nil {
			return 0, err
		}
		if terminalWorkerEventType(event.Type) {
			return 0, fmt.Errorf("%w: control event log contains a derived terminal event", ErrInvalid)
		}
		sequence, err := strconv.ParseUint(event.ID, 10, 64)
		if err != nil {
			return 0, fmt.Errorf("decode evaluation event id: %w", err)
		}
		if event.RunID != runID || sequence != count || event.ID != strconv.FormatUint(count, 10) {
			return 0, fmt.Errorf("evaluation event history is not a strictly monotonic run-local sequence")
		}
		last = sequence
	}
	if err := scanner.Err(); err != nil {
		return 0, fmt.Errorf("scan evaluation event log: %w", err)
	}
	return last, nil
}

func decodeStoredEvent(line []byte) (Event, error) {
	if len(line) == 0 || len(line) > maxWorkerEventLineBytes || !utf8.Valid(line) {
		return Event{}, fmt.Errorf("%w: evaluation event exceeds the durable envelope", ErrInvalid)
	}
	if err := rejectDuplicateJSONKeys(line); err != nil {
		return Event{}, fmt.Errorf("decode evaluation event: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(line))
	decoder.DisallowUnknownFields()
	var event Event
	if err := decoder.Decode(&event); err != nil {
		return Event{}, fmt.Errorf("decode evaluation event: %w", err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return Event{}, err
	}
	if err := validateStoredEvent(event); err != nil {
		return Event{}, err
	}
	return event, nil
}

func validateStoredEvent(event Event) error {
	sequence, err := strconv.ParseUint(event.ID, 10, 64)
	if err != nil || sequence == 0 || event.ID != strconv.FormatUint(sequence, 10) ||
		!validClientRequestID(event.RunID) || !allowedWorkerEventType(event.Type) || event.Timestamp.IsZero() {
		return fmt.Errorf("%w: evaluation event identity is invalid", ErrInvalid)
	}
	if event.Message != strings.TrimSpace(event.Message) || event.Message == "" || len(event.Message) > maxWorkerMessageBytes {
		return fmt.Errorf("%w: evaluation event message is invalid", ErrInvalid)
	}
	if event.TrackID != "" && !containsTrack(allTrackIDs, event.TrackID) {
		return fmt.Errorf("%w: evaluation event track is invalid", ErrInvalid)
	}
	if event.Progress != nil {
		progress := event.Progress
		if math.IsNaN(progress.Percent) || math.IsInf(progress.Percent, 0) ||
			progress.Percent < 0 || progress.Percent > 100 || progress.Completed < 0 ||
			progress.Total < 0 || progress.Total > len(allTrackIDs) || progress.Completed > progress.Total ||
			(progress.CurrentTrackID != "" && !containsTrack(allTrackIDs, progress.CurrentTrackID)) ||
			progress.Message != strings.TrimSpace(progress.Message) || len(progress.Message) > maxWorkerMessageBytes {
			return fmt.Errorf("%w: evaluation event progress is invalid", ErrInvalid)
		}
	}
	if err := validateDurableEventPayload(event.Type, event.Payload); err != nil {
		return fmt.Errorf("%w: evaluation event payload is invalid: %w", ErrInvalid, err)
	}
	return nil
}

func validateDurableEventPayload(eventType string, payload *WorkerEventPayload) error {
	if eventType == "completed" {
		if payload != nil {
			return fmt.Errorf("server-owned completed event cannot contain a worker payload")
		}
		return nil
	}
	return validateWorkerEventPayload(eventType, payload)
}

func syncEvaluationDirectory(path, description string) error {
	directory, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open %s directory: %w", description, err)
	}
	syncErr := directory.Sync()
	closeErr := directory.Close()
	if syncErr != nil {
		return fmt.Errorf("sync %s directory: %w", description, syncErr)
	}
	if closeErr != nil {
		return fmt.Errorf("close %s directory: %w", description, closeErr)
	}
	return nil
}
