package evaluationplane

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"sync"
)

const (
	runFileName      = "status.json"
	manifestFileName = "run-manifest.json"
	eventsFileName   = "control-events.jsonl"
	reportFileName   = "report.json"
	maxEventsPerRun  = uint64(8192)
	maxEventLogBytes = int64(16 * 1024 * 1024)
)

var safeIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$`)

type Store struct {
	root      string
	runsRoot  string
	mu        sync.Mutex
	sequences map[string]uint64
}

func NewStore(root string) (*Store, error) {
	if root == "" {
		return nil, fmt.Errorf("%w: evaluation data directory is required", ErrInvalid)
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("resolve evaluation data directory: %w", err)
	}
	runsRoot := filepath.Join(absRoot, "runs")
	privateDirectories := []string{
		absRoot,
		filepath.Join(absRoot, "objects"),
		filepath.Join(absRoot, "objects", "sha256"),
		runsRoot,
		filepath.Join(absRoot, "index"),
	}
	for _, directory := range privateDirectories {
		if err := os.MkdirAll(directory, 0o700); err != nil {
			return nil, fmt.Errorf("create evaluation store directory: %w", err)
		}
		if err := requirePrivateDirectory(directory); err != nil {
			return nil, err
		}
	}
	return &Store{root: absRoot, runsRoot: runsRoot, sequences: make(map[string]uint64)}, nil
}

func (s *Store) Root() string { return s.root }

func (s *Store) CreateBundle(run Run, manifest RunManifest) (string, error) {
	if err := validateResourceID(run.ID); err != nil {
		return "", err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	runDir := filepath.Join(s.runsRoot, run.ID)
	if err := os.Mkdir(runDir, 0o700); err != nil {
		if os.IsExist(err) {
			return "", fmt.Errorf("%w: run %s already exists", ErrConflict, run.ID)
		}
		return "", fmt.Errorf("create run bundle: %w", err)
	}
	if err := writeJSONAtomic(filepath.Join(runDir, runFileName), run); err != nil {
		_ = os.RemoveAll(runDir)
		return "", err
	}
	if err := writeJSONAtomic(filepath.Join(runDir, manifestFileName), manifest); err != nil {
		_ = os.RemoveAll(runDir)
		return "", err
	}
	if err := os.WriteFile(filepath.Join(runDir, eventsFileName), nil, 0o600); err != nil {
		_ = os.RemoveAll(runDir)
		return "", fmt.Errorf("initialize event log: %w", err)
	}
	return filepath.Join(runDir, manifestFileName), nil
}

func (s *Store) GetRun(id string) (Run, error) {
	runDir, err := s.checkedRunDir(id)
	if err != nil {
		return Run{}, err
	}
	var run Run
	if err := readJSON(filepath.Join(runDir, runFileName), &run); err != nil {
		return Run{}, err
	}
	return run, nil
}

func (s *Store) ListRuns() ([]Run, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	entries, err := os.ReadDir(s.runsRoot)
	if err != nil {
		return nil, fmt.Errorf("list evaluation runs: %w", err)
	}
	runs := make([]Run, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() || !safeIDPattern.MatchString(entry.Name()) {
			continue
		}
		run, readErr := s.GetRun(entry.Name())
		if readErr != nil {
			return nil, fmt.Errorf("read run bundle %s: %w", entry.Name(), readErr)
		}
		runs = append(runs, run)
	}
	sort.Slice(runs, func(i, j int) bool { return runs[i].CreatedAt.After(runs[j].CreatedAt) })
	return runs, nil
}

func (s *Store) UpdateRun(run Run) error {
	if err := validateResourceID(run.ID); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	runDir, err := s.checkedRunDir(run.ID)
	if err != nil {
		return err
	}
	return writeJSONAtomic(filepath.Join(runDir, runFileName), run)
}

func (s *Store) DeleteRun(id string) error {
	if err := validateResourceID(id); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	runDir, err := s.checkedRunDir(id)
	if err != nil {
		return err
	}
	delete(s.sequences, id)
	if err := os.RemoveAll(runDir); err != nil {
		return fmt.Errorf("delete evaluation run: %w", err)
	}
	return nil
}

func (s *Store) ManifestPath(id string) (string, error) {
	runDir, err := s.checkedRunDir(id)
	if err != nil {
		return "", err
	}
	path := filepath.Join(runDir, manifestFileName)
	file, err := openBundleFile(path, os.O_RDONLY)
	if err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("%w: run manifest", ErrNotFound)
		}
		return "", fmt.Errorf("open run manifest: %w", err)
	}
	_ = file.Close()
	return path, nil
}

func (s *Store) AppendEvent(event Event) (Event, error) {
	if err := validateResourceID(event.RunID); err != nil {
		return Event{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	runDir, err := s.checkedRunDir(event.RunID)
	if err != nil {
		return Event{}, err
	}
	sequence := s.sequences[event.RunID]
	if sequence == 0 {
		sequence, err = lastEventSequence(filepath.Join(runDir, eventsFileName))
		if err != nil {
			return Event{}, err
		}
	}
	sequence++
	if sequence > maxEventsPerRun {
		return Event{}, fmt.Errorf("%w: evaluation event log reached its per-run limit", ErrInvalid)
	}
	event.ID = strconv.FormatUint(sequence, 10)
	encoded, err := json.Marshal(event)
	if err != nil {
		return Event{}, fmt.Errorf("encode evaluation event: %w", err)
	}
	file, err := openBundleFile(filepath.Join(runDir, eventsFileName), os.O_WRONLY|os.O_APPEND)
	if err != nil {
		return Event{}, fmt.Errorf("open evaluation event log: %w", err)
	}
	if _, err = file.Write(append(encoded, '\n')); err == nil {
		err = file.Sync()
	}
	closeErr := file.Close()
	if err != nil {
		return Event{}, fmt.Errorf("append evaluation event: %w", err)
	}
	if closeErr != nil {
		return Event{}, fmt.Errorf("close evaluation event log: %w", closeErr)
	}
	s.sequences[event.RunID] = sequence
	return event, nil
}

func (s *Store) EventsAfter(id string, after uint64) ([]Event, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	runDir, err := s.checkedRunDir(id)
	if err != nil {
		return nil, err
	}
	file, err := openBundleFile(filepath.Join(runDir, eventsFileName), os.O_RDONLY)
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
	var events []Event
	var scanned uint64
	for scanner.Scan() {
		scanned++
		if scanned > maxEventsPerRun {
			return nil, fmt.Errorf("%w: evaluation event log exceeds its per-run event limit", ErrInvalid)
		}
		var event Event
		if err := json.Unmarshal(scanner.Bytes(), &event); err != nil {
			return nil, fmt.Errorf("decode evaluation event: %w", err)
		}
		sequence, err := strconv.ParseUint(event.ID, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("decode evaluation event id: %w", err)
		}
		if sequence > after {
			events = append(events, event)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan evaluation event log: %w", err)
	}
	return events, nil
}

func (s *Store) ReadReport(id string) ([]byte, error) {
	runDir, err := s.checkedRunDir(id)
	if err != nil {
		return nil, err
	}
	path := filepath.Join(runDir, reportFileName)
	file, err := openBundleFile(path, os.O_RDONLY)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("%w: evaluation report", ErrNotFound)
		}
		return nil, fmt.Errorf("read evaluation report: %w", err)
	}
	defer func() { _ = file.Close() }()
	info, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat evaluation report: %w", err)
	}
	if info.Size() > maxStructuredArtifactBytes {
		return nil, fmt.Errorf("evaluation report exceeds the structured artifact limit")
	}
	data, err := io.ReadAll(io.LimitReader(file, maxStructuredArtifactBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read evaluation report: %w", err)
	}
	if int64(len(data)) > maxStructuredArtifactBytes {
		return nil, fmt.Errorf("evaluation report exceeds the structured artifact limit")
	}
	if !json.Valid(data) {
		return nil, fmt.Errorf("evaluation report is not valid JSON")
	}
	return data, nil
}

func (s *Store) WriteReport(id string, report any) error {
	runDir, err := s.checkedRunDir(id)
	if err != nil {
		return err
	}
	return writeJSONAtomic(filepath.Join(runDir, reportFileName), report)
}

func (s *Store) checkedRunDir(id string) (string, error) {
	if err := validateResourceID(id); err != nil {
		return "", err
	}
	runDir := filepath.Join(s.runsRoot, id)
	info, err := os.Lstat(runDir)
	if err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("%w: run %s", ErrNotFound, id)
		}
		return "", fmt.Errorf("stat evaluation run: %w", err)
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return "", fmt.Errorf("evaluation run bundle is not a directory")
	}
	if err := requirePrivateDirectory(runDir); err != nil {
		return "", err
	}
	return runDir, nil
}

func validateResourceID(id string) error {
	if !safeIDPattern.MatchString(id) {
		return fmt.Errorf("%w: invalid resource id", ErrInvalid)
	}
	return nil
}
