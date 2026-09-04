package evaluationplane

import (
	"context"
	"errors"
	"fmt"
	"os"
	"reflect"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

type Options struct {
	DataDir            string
	PythonPath         string
	RouterAPIURL       string
	EnvoyURL           string
	ConfigPath         string
	CodeRevision       string
	EnvoyAPIKeyEnv     string
	MaxConcurrent      int
	WorkerTimeout      time.Duration
	Process            Process
	CredentialProvider CredentialProvider
}

const defaultWorkerTimeout = 6 * time.Hour

const maxWorkerEventsPerRun = 4096

const (
	maxSubscribersPerRun       = 16
	maxSubscribersGlobal       = 256
	maxConcurrentEvidenceReads = 8
)

var sourceRevisionPattern = regexp.MustCompile(`^(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})$`)

type Service struct {
	store              *Store
	registry           *Registry
	process            Process
	configPath         string
	codeRevision       string
	routerAPIURL       string
	envoyURL           string
	envoyAPIKeyEnv     string
	routerAuthRequired bool
	semaphore          chan struct{}
	evidenceReads      chan struct{}
	workerTimeout      time.Duration

	mu              sync.Mutex
	active          map[string]context.CancelFunc
	workerEvents    map[string]int
	subscribers     map[string]map[chan Event]struct{}
	subscriberCount int
	workers         sync.WaitGroup
	closeOnce       sync.Once
	closed          bool
}

func NewService(options Options) (*Service, error) {
	store, err := NewStore(options.DataDir)
	if err != nil {
		return nil, err
	}
	codeRevision := strings.TrimSpace(options.CodeRevision)
	if codeRevision == "" {
		codeRevision = "unavailable"
	}
	if options.EnvoyAPIKeyEnv != "" && !secretEnvPattern.MatchString(options.EnvoyAPIKeyEnv) {
		return nil, fmt.Errorf("evaluation Envoy credential reference must be an uppercase environment variable name")
	}
	snapshot, err := LoadModelArmSnapshot(options.ConfigPath, codeRevision)
	if err != nil {
		return nil, err
	}
	routerAuthRequired := routerAuthenticationRequired(options.CredentialProvider)
	registry, err := NewRegistry(options.RouterAPIURL, options.EnvoyURL, RegistryOptions{
		EnvoyAPIKey: configuredSecretRef(options.EnvoyURL, options.EnvoyAPIKeyEnv),
		ModelArms:   snapshot.ModelArms, BackendTopologyDigest: snapshot.BackendTopologyDigest,
		RouterAuthRequired: routerAuthRequired,
	})
	if err != nil {
		return nil, err
	}
	if options.MaxConcurrent <= 0 {
		options.MaxConcurrent = 2
	}
	if options.WorkerTimeout < 0 {
		return nil, fmt.Errorf("evaluation worker timeout cannot be negative")
	}
	if options.WorkerTimeout == 0 {
		options.WorkerTimeout = defaultWorkerTimeout
	}
	process := options.Process
	if process == nil {
		commandProcess := NewCommandProcess(options.PythonPath)
		commandProcess.envoyAPIKeyEnv = strings.TrimSpace(options.EnvoyAPIKeyEnv)
		process = commandProcess
	}
	service := &Service{
		store:              store,
		registry:           registry,
		process:            process,
		configPath:         options.ConfigPath,
		codeRevision:       codeRevision,
		routerAPIURL:       options.RouterAPIURL,
		envoyURL:           options.EnvoyURL,
		envoyAPIKeyEnv:     strings.TrimSpace(options.EnvoyAPIKeyEnv),
		routerAuthRequired: routerAuthRequired,
		semaphore:          make(chan struct{}, options.MaxConcurrent),
		evidenceReads:      make(chan struct{}, maxConcurrentEvidenceReads),
		workerTimeout:      options.WorkerTimeout,
		active:             make(map[string]context.CancelFunc),
		workerEvents:       make(map[string]int),
		subscribers:        make(map[string]map[chan Event]struct{}),
	}
	if err := service.RecoverInterruptedRuns(); err != nil {
		return nil, err
	}
	return service, nil
}

func (s *Service) Catalog() Catalog {
	registry, _, err := s.registrySnapshot()
	if err != nil {
		return s.registry.Catalog()
	}
	return registry.Catalog()
}

func (s *Service) CreateRun(ctx context.Context, request CreateRunRequest) (Run, error) {
	if !sourceRevisionPattern.MatchString(s.codeRevision) {
		return Run{}, fmt.Errorf("%w: evaluation source revision must be an immutable git commit or source-tree digest", ErrInvalid)
	}
	registry, snapshot, err := s.registrySnapshot()
	if err != nil {
		return Run{}, err
	}
	validated, target, err := s.validateCreateRequest(registry, request)
	if err != nil {
		return Run{}, err
	}
	evidenceLevel, err := selectedSuiteEvidenceLevel(registry, validated.SuiteIDs)
	if err != nil {
		return Run{}, err
	}
	if qualificationErr := requireQualifiedCodeRevision(evidenceLevel, s.codeRevision); qualificationErr != nil {
		return Run{}, qualificationErr
	}
	if validated.BaselineRunID != "" {
		baseline, getErr := s.store.GetRun(validated.BaselineRunID)
		if getErr != nil {
			return Run{}, fmt.Errorf("%w: baseline run is unavailable", ErrInvalid)
		}
		if baseline.Status != StatusCompleted {
			return Run{}, fmt.Errorf("%w: baseline run must be completed", ErrInvalid)
		}
		if comparisonErr := validateComparableRunRequest(validated, baseline); comparisonErr != nil {
			return Run{}, comparisonErr
		}
		if snapshotErr := s.validateComparableTargetSnapshot(validated.ChangeProfile, target, snapshot, baseline.ID); snapshotErr != nil {
			return Run{}, snapshotErr
		}
	}
	// Python datetime serialization is microsecond-precise. Freeze the shared
	// timestamp at that precision so Go/Python evidence compares byte-stably.
	now := time.Now().UTC().Truncate(time.Microsecond)
	run := Run{
		SchemaVersion: SchemaVersion,
		ID:            uuid.NewString(), Name: validated.Name, Description: validated.Description,
		Status: StatusPending, Mode: validated.Mode, EvidenceLevel: evidenceLevel,
		TargetID: validated.TargetID, ChangeProfile: validated.ChangeProfile,
		SuiteIDs: validated.SuiteIDs, TrackIDs: validated.TrackIDs,
		SampleLimit: validated.SampleLimit, Concurrency: validated.Concurrency, Seed: validated.Seed,
		BaselineRunID: validated.BaselineRunID,
		Progress:      RunProgress{Total: len(validated.TrackIDs), Message: "Run created"},
		CreatedAt:     now,
	}
	manifest := RunManifest{
		SchemaVersion: SchemaVersion, RunID: run.ID, Mode: run.Mode,
		Target: ManifestTarget{
			SchemaVersion: SchemaVersion, ID: target.Public.ID, Kind: target.Public.Kind,
			RouterAPIURL: target.RouterAPIURL, EnvoyURL: target.EnvoyURL,
			RouterAPIKey: copySecretRef(target.RouterAPIKey), EnvoyAPIKey: copySecretRef(target.EnvoyAPIKey),
			ModelArms:             copyModelArms(target.ModelArms),
			BackendTopologyDigest: target.BackendTopologyDigest,
		},
		ChangeProfile:       run.ChangeProfile,
		GateContractVersion: GateContractVersion,
		SuiteIDs:            run.SuiteIDs, SuiteRevisions: suiteRevisionSnapshot(registry, run.SuiteIDs),
		TrackIDs: run.TrackIDs, SampleLimit: run.SampleLimit,
		Concurrency: run.Concurrency, Seed: run.Seed, BaselineRunID: run.BaselineRunID,
		CreatedAt: now, CodeRevision: s.codeRevision, ConfigDigest: snapshot.ConfigDigest,
		PolicySnapshotDigest: manifestPolicySnapshotDigest(target, snapshot),
		RedactionPolicy:      "evaluation-default-v1",
	}
	manifest.ManifestDigest, err = manifestSemanticDigest(manifest)
	if err != nil {
		return Run{}, fmt.Errorf("%w: compute immutable evaluation manifest identity: %w", ErrInvalid, err)
	}
	if _, err := s.store.CreateBundle(run, manifest); err != nil {
		return Run{}, err
	}
	if _, err := s.appendEvent(Event{RunID: run.ID, Type: "snapshot", Timestamp: now, Message: "Immutable run manifest created", Progress: &run.Progress}); err != nil {
		_ = s.store.DeleteRun(run.ID)
		return Run{}, err
	}
	return run, nil
}

func requireQualifiedCodeRevision(_ EvidenceLevel, revision string) error {
	if !sourceRevisionPattern.MatchString(strings.TrimSpace(revision)) {
		return fmt.Errorf("%w: evaluation requires a full Git commit or sha256 source-tree revision", ErrInvalid)
	}
	return nil
}

func validateComparableRunRequest(candidate CreateRunRequest, baseline Run) error {
	if candidate.Mode != baseline.Mode || candidate.TargetID != baseline.TargetID ||
		candidate.ChangeProfile != baseline.ChangeProfile ||
		candidate.SampleLimit != baseline.SampleLimit || candidate.Concurrency != baseline.Concurrency || candidate.Seed != baseline.Seed ||
		!sameStringSet(candidate.SuiteIDs, baseline.SuiteIDs) || !sameTrackSet(candidate.TrackIDs, baseline.TrackIDs) {
		return fmt.Errorf("%w: candidate change_profile, mode, target, suites, tracks, sample_limit, concurrency, and seed must match the baseline", ErrInvalid)
	}
	return nil
}

func (s *Service) validateComparableTargetSnapshot(
	profile ChangeProfile,
	target targetDefinition,
	snapshot ModelArmSnapshot,
	baselineRunID string,
) error {
	manifestPath, err := s.store.ManifestPath(baselineRunID)
	if err != nil {
		return fmt.Errorf("%w: baseline manifest is unavailable", ErrInvalid)
	}
	var baseline RunManifest
	if err := readJSON(manifestPath, &baseline); err != nil {
		return fmt.Errorf("%w: baseline manifest is unavailable", ErrInvalid)
	}
	if baseline.ChangeProfile != profile {
		return fmt.Errorf("%w: baseline manifest change_profile does not match", ErrInvalid)
	}
	allowed := comparisonTreatment(profile)
	if !allowed.pool && !reflect.DeepEqual(baseline.Target.ModelArms, target.ModelArms) {
		return fmt.Errorf("%w: model pool snapshot must remain frozen for change_profile %q", ErrInvalid, profile)
	}
	if !allowed.environment && baseline.Target.BackendTopologyDigest != target.BackendTopologyDigest {
		return fmt.Errorf("%w: backend topology snapshot must remain frozen for change_profile %q", ErrInvalid, profile)
	}
	candidatePolicyDigest := manifestPolicySnapshotDigest(target, snapshot)
	if !digestPattern.MatchString(baseline.PolicySnapshotDigest) || !digestPattern.MatchString(candidatePolicyDigest) {
		return fmt.Errorf("%w: policy snapshot identity is unavailable", ErrInvalid)
	}
	if !allowed.policy && baseline.PolicySnapshotDigest != candidatePolicyDigest {
		return fmt.Errorf("%w: policy snapshot must remain frozen for change_profile %q", ErrInvalid, profile)
	}
	return nil
}

func manifestPolicySnapshotDigest(target targetDefinition, snapshot ModelArmSnapshot) string {
	if target.Public.ID == "fixture" {
		return fixturePolicySnapshotDigest
	}
	return snapshot.PolicySnapshotDigest
}

func sameStringSet(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	values := make(map[string]bool, len(left))
	for _, value := range left {
		values[value] = true
	}
	for _, value := range right {
		if !values[value] {
			return false
		}
	}
	return true
}

func sameTrackSet(left, right []TrackID) bool {
	if len(left) != len(right) {
		return false
	}
	values := make(map[TrackID]bool, len(left))
	for _, value := range left {
		values[value] = true
	}
	for _, value := range right {
		if !values[value] {
			return false
		}
	}
	return true
}

func (s *Service) ListRuns() ([]Run, error) { return s.store.ListRuns() }

func (s *Service) GetRun(id string) (Run, error) { return s.store.GetRun(id) }

func (s *Service) StartRun(_ context.Context, id string) (Run, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	run, err := s.store.GetRun(id)
	if err != nil {
		return Run{}, err
	}
	if run.Status == StatusRunning || terminalStatus(run.Status) {
		return run, nil
	}
	if s.closed {
		return Run{}, fmt.Errorf("%w: evaluation service is closed", ErrConflict)
	}
	if run.Status != StatusPending {
		return Run{}, fmt.Errorf("%w: run cannot be started from %s", ErrConflict, run.Status)
	}
	if validationErr := s.validateRunStart(run); validationErr != nil {
		return Run{}, validationErr
	}
	manifestPath, err := s.store.ManifestPath(id)
	if err != nil {
		return Run{}, err
	}
	select {
	case s.semaphore <- struct{}{}:
	default:
		return Run{}, fmt.Errorf("%w: evaluation worker capacity is full; run remains pending", ErrConflict)
	}
	releaseSlot := func() { <-s.semaphore }
	workerContext, cancel := context.WithTimeout(context.Background(), s.workerTimeout)
	now := time.Now().UTC()
	pendingRun := run
	run.Status = StatusRunning
	run.StartedAt = &now
	run.Error = ""
	run.Progress.Message = "Evaluation worker starting"
	if err := s.store.UpdateRun(run); err != nil {
		cancel()
		releaseSlot()
		return Run{}, err
	}
	if _, err := s.appendEventLocked(Event{RunID: id, Type: "progress", Timestamp: now, Message: run.Progress.Message, Progress: &run.Progress}); err != nil {
		cancel()
		releaseSlot()
		if rollbackErr := s.store.UpdateRun(pendingRun); rollbackErr != nil {
			return Run{}, errors.Join(err, fmt.Errorf("restore pending evaluation run: %w", rollbackErr))
		}
		return Run{}, err
	}
	s.active[id] = cancel
	s.workerEvents[id] = 0
	s.workers.Add(1)
	go s.execute(workerContext, id, manifestPath)
	return run, nil
}

func (s *Service) validateRunStart(run Run) error {
	manifest, _, err := s.readDurableManifest(run.ID)
	if err != nil {
		return err
	}
	if manifest.CodeRevision != s.codeRevision {
		return fmt.Errorf("%w: pending run source revision does not match the active evaluation worker", ErrConflict)
	}
	registry, _, err := s.registrySnapshot()
	if err != nil {
		return err
	}
	if manifest.GateContractVersion != GateContractVersion ||
		!reflect.DeepEqual(manifest.SuiteRevisions, suiteRevisionSnapshot(registry, manifest.SuiteIDs)) {
		return fmt.Errorf("%w: pending run suite or change-profile contract revision does not match the active evaluation worker", ErrConflict)
	}
	_, _, err = s.validateCreateRequest(registry, CreateRunRequest{
		Name: run.Name, Description: run.Description,
		SuiteIDs: run.SuiteIDs, TrackIDs: run.TrackIDs,
		Mode: run.Mode, TargetID: run.TargetID, ChangeProfile: run.ChangeProfile,
		SampleLimit: run.SampleLimit, Concurrency: run.Concurrency, Seed: run.Seed,
		BaselineRunID: run.BaselineRunID,
	})
	if err != nil {
		return fmt.Errorf("%w: run target is no longer supported", ErrConflict)
	}
	return nil
}

func suiteRevisionSnapshot(registry *Registry, suiteIDs []string) map[string]string {
	revisions := make(map[string]string, len(suiteIDs))
	for _, suiteID := range suiteIDs {
		if suite, ok := registry.suite(suiteID); ok {
			revisions[suiteID] = suite.Revision
		}
	}
	return revisions
}

// Close prevents new workers from starting, cancels every active worker, and
// waits until each worker has released its process and concurrency slot.
func (s *Service) Close() error {
	s.closeOnce.Do(func() {
		s.mu.Lock()
		s.closed = true
		cancellations := make([]context.CancelFunc, 0, len(s.active))
		for _, cancel := range s.active {
			cancellations = append(cancellations, cancel)
		}
		s.mu.Unlock()

		for _, cancel := range cancellations {
			cancel()
		}
		s.workers.Wait()

		s.mu.Lock()
		for runID, subscribers := range s.subscribers {
			for subscriber := range subscribers {
				close(subscriber)
			}
			delete(s.subscribers, runID)
		}
		s.subscriberCount = 0
		s.mu.Unlock()
	})
	return nil
}

func (s *Service) CancelRun(id string) (Run, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	run, err := s.store.GetRun(id)
	if err != nil {
		return Run{}, err
	}
	if terminalStatus(run.Status) {
		return run, nil
	}
	now := time.Now().UTC()
	run.Status = StatusCancelled
	run.CompletedAt = &now
	run.Progress.Message = "Run cancelled"
	if err := s.store.UpdateRun(run); err != nil {
		return Run{}, err
	}
	if cancel, ok := s.active[id]; ok {
		cancel()
	}
	if _, err := s.appendEventLocked(Event{RunID: id, Type: "cancelled", Timestamp: now, Message: "Run cancelled by user", Progress: &run.Progress}); err != nil {
		return Run{}, err
	}
	return run, nil
}

func (s *Service) DeleteRun(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	run, err := s.store.GetRun(id)
	if err != nil {
		return err
	}
	if _, active := s.active[id]; active {
		return fmt.Errorf("%w: evaluation worker is still exiting", ErrConflict)
	}
	if run.Status == StatusRunning {
		return fmt.Errorf("%w: cancel a running evaluation before deletion", ErrConflict)
	}
	for subscriber := range s.subscribers[id] {
		close(subscriber)
		s.subscriberCount--
	}
	delete(s.subscribers, id)
	return s.store.DeleteRun(id)
}

func (s *Service) registrySnapshot() (*Registry, ModelArmSnapshot, error) {
	snapshot, err := LoadModelArmSnapshot(s.configPath, s.codeRevision)
	if err != nil {
		return nil, ModelArmSnapshot{}, err
	}
	registry, err := NewRegistry(s.routerAPIURL, s.envoyURL, RegistryOptions{
		EnvoyAPIKey: configuredSecretRef(s.envoyURL, s.envoyAPIKeyEnv),
		ModelArms:   snapshot.ModelArms, BackendTopologyDigest: snapshot.BackendTopologyDigest,
		RouterAuthRequired: s.routerAuthRequired,
	})
	if err != nil {
		return nil, ModelArmSnapshot{}, err
	}
	return registry, snapshot, nil
}

func routerAuthenticationRequired(provider CredentialProvider) bool {
	if provider == nil {
		return false
	}
	token, err := provider.ManagementCredential()
	if errors.Is(err, os.ErrNotExist) {
		return false
	}
	return err != nil || strings.TrimSpace(token) != ""
}

func configuredSecretRef(endpointURL, envName string) *SecretRef {
	if strings.TrimSpace(endpointURL) == "" || strings.TrimSpace(envName) == "" {
		return nil
	}
	return &SecretRef{SchemaVersion: SchemaVersion, Env: strings.TrimSpace(envName)}
}

func terminalStatus(status RunStatus) bool {
	return status == StatusCompleted || status == StatusFailed || status == StatusCancelled
}
