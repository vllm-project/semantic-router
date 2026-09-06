package evaluationplane

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"
)

const (
	runFileName      = "status.json"
	manifestFileName = "run-manifest.json"
	eventsFileName   = "control-events.jsonl"
	reportFileName   = "report.json"
	maxEventsPerRun  = uint64(8192)
	maxEventLogBytes = int64(16 * 1024 * 1024)
)

type Store struct {
	root                  string
	runsRoot              string
	suiteRoot             string
	attestationRoot       string
	controlledPairRoot    string
	campaignRoot          string
	lifecycleRoot         string
	collectionRoot        string
	lifecycleAuditRoot    string
	lifecycleBindingRoot  string
	mu                    sync.Mutex
	runIndex              *runMetadataIndex
	lifecycle             *evaluationRootCoordinator
	lifecyclePolicy       *lifecycleStorePolicy
	lifecycleNow          func() time.Time
	lifecyclePersistence  lifecyclePolicyPersistence
	collectionPersistence lifecycleCollectionPersistence
	lifecycleAuditWriter  lifecycleAuditWriter
	lifecycleCleaner      lifecycleCheckpointCleaner
	statusPersistence     runStatusPersistence
	eventPersistence      runEventPersistence
	runPersistence        runNamespacePersistence
	pairPersistence       controlledPairPersistence
	campaignPersistence   campaignNamespacePersistence
}

type evaluationStoreLayout struct {
	root                 string
	runsRoot             string
	suiteRoot            string
	attestationRoot      string
	controlledPairRoot   string
	campaignRoot         string
	lifecycleRoot        string
	collectionRoot       string
	lifecycleAuditRoot   string
	lifecycleBindingRoot string
}

func prepareEvaluationStoreLayout(root string, startupAuthority bool) (evaluationStoreLayout, error) {
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return evaluationStoreLayout{}, fmt.Errorf("resolve evaluation data directory: %w", err)
	}
	layout := evaluationStoreLayout{
		root: absRoot, runsRoot: filepath.Join(absRoot, "runs"), suiteRoot: filepath.Join(absRoot, "suites"),
		attestationRoot:    filepath.Join(absRoot, "attestations"),
		controlledPairRoot: filepath.Join(absRoot, "controlled-pairs"),
		campaignRoot:       filepath.Join(absRoot, "campaigns"),
		lifecycleRoot:      filepath.Join(absRoot, "lifecycle"),
	}
	layout.collectionRoot = filepath.Join(layout.lifecycleRoot, "collection")
	layout.lifecycleAuditRoot = filepath.Join(layout.lifecycleRoot, lifecycleAuditDirectoryName)
	layout.lifecycleBindingRoot = filepath.Join(layout.lifecycleRoot, lifecycleBindingDirectoryName)
	privateDirectories := []string{
		layout.root, layout.campaignRoot,
		filepath.Join(layout.root, "objects"), filepath.Join(layout.root, "objects", "sha256"),
		layout.runsRoot, layout.suiteRoot, layout.attestationRoot, layout.controlledPairRoot,
		layout.lifecycleRoot, layout.collectionRoot, layout.lifecycleAuditRoot, layout.lifecycleBindingRoot,
		filepath.Join(layout.suiteRoot, "objects", "visible", "sha256"),
		filepath.Join(layout.suiteRoot, "objects", "grading", "sha256"),
		filepath.Join(layout.suiteRoot, "objects", "metadata", "sha256"),
		filepath.Join(layout.suiteRoot, "manifests", "sha256"), filepath.Join(layout.suiteRoot, "index"),
	}
	for _, directory := range privateDirectories {
		if startupAuthority {
			if err := ensureDurablePrivateDirectoryTree(directory); err != nil {
				return evaluationStoreLayout{}, fmt.Errorf("create evaluation store directory: %w", err)
			}
		}
		if err := requirePrivateDirectory(directory); err != nil {
			return evaluationStoreLayout{}, err
		}
	}
	return layout, nil
}

func newStoreWithRootCoordinator(
	root string,
	requestedLimits LifecycleLimits,
	coordinator *evaluationRootCoordinator,
	startupAuthority bool,
) (*Store, error) {
	if root == "" {
		return nil, fmt.Errorf("%w: evaluation data directory is required", ErrInvalid)
	}
	limits := requestedLimits
	if requestedLimits != (LifecycleLimits{}) {
		var err error
		limits, err = normalizeLifecycleLimits(requestedLimits)
		if err != nil {
			return nil, err
		}
	}
	layout, err := prepareEvaluationStoreLayout(root, startupAuthority)
	if err != nil {
		return nil, err
	}
	if coordinator == nil || coordinator.root != layout.root {
		return nil, fmt.Errorf("evaluation root coordinator does not match the store root")
	}
	store := &Store{
		root: layout.root, runsRoot: layout.runsRoot, suiteRoot: layout.suiteRoot,
		attestationRoot:    layout.attestationRoot,
		controlledPairRoot: layout.controlledPairRoot,
		campaignRoot:       layout.campaignRoot,
		lifecycleRoot:      layout.lifecycleRoot, collectionRoot: layout.collectionRoot,
		lifecycleAuditRoot:    layout.lifecycleAuditRoot,
		lifecycleBindingRoot:  layout.lifecycleBindingRoot,
		runIndex:              coordinator.runIndex,
		lifecycle:             coordinator,
		lifecyclePolicy:       &coordinator.policy,
		lifecycleNow:          func() time.Time { return time.Now().UTC() },
		lifecyclePersistence:  atomicLifecyclePolicyPersistence{},
		collectionPersistence: atomicLifecycleCollectionPersistence{},
		lifecycleAuditWriter:  atomicLifecycleAuditWriter{},
		lifecycleCleaner:      atomicLifecycleCheckpointCleaner{},
		statusPersistence:     atomicRunStatusPersistence{},
		eventPersistence:      atomicRunEventPersistence{},
		runPersistence:        atomicRunNamespacePersistence{},
		pairPersistence:       atomicControlledPairPersistence{},
		campaignPersistence:   atomicCampaignNamespacePersistence{},
	}
	if err := store.initializeLifecycleDurability(limits, startupAuthority); err != nil {
		return nil, err
	}
	if err := store.recoverLifecycleEvidenceAndIndex(startupAuthority); err != nil {
		return nil, err
	}
	if err := store.validateLifecycleRunBindings(startupAuthority); err != nil {
		return nil, err
	}
	if err := store.validateLifecycleCampaignBindings(startupAuthority); err != nil {
		return nil, err
	}
	if startupAuthority {
		if err := store.finishLifecycleCheckpointCleanup(); err != nil {
			return nil, err
		}
	}
	if err := store.validateRunReferenceIntegrity(); err != nil {
		return nil, err
	}
	if err := store.validateCampaignReferenceIntegrity(startupAuthority); err != nil {
		return nil, err
	}
	if err := store.recoverLifecycleCollection(startupAuthority); err != nil {
		return nil, err
	}
	// CAS collection is startup recovery, not peer validation. A peer opener
	// must never delete objects or append startup audit records while another
	// Service still owns live publication state for this root.
	if startupAuthority {
		store.recoverCASGarbage()
	}
	return store, nil
}

func (s *Store) initializeLifecycleDurability(limits LifecycleLimits, startupAuthority bool) error {
	s.lifecycle.mu.Lock()
	defer s.lifecycle.mu.Unlock()
	if !startupAuthority {
		if err := s.validatePeerLifecyclePolicyUnlocked(limits); err != nil {
			return err
		}
		if err := requireNoLifecycleAuditTemps(s.lifecycleAuditRoot); err != nil {
			return err
		}
		if err := requireNoLifecycleAuditTemps(s.lifecycleBindingRoot); err != nil {
			return err
		}
		if err := s.requireNoPendingLifecycleResources(); err != nil {
			return err
		}
		if err := s.requireNoPendingCampaignPublications(); err != nil {
			return err
		}
		if err := s.validatePeerLifecycleAuditUnlocked(); err != nil {
			return err
		}
		s.lifecycle.evidenceMu.Lock()
		defer s.lifecycle.evidenceMu.Unlock()
		if err := s.requireNoCampaignDeletionIntentsUnlocked(); err != nil {
			return err
		}
		return requireNoStagedCampaigns(s.campaignRoot)
	}
	if err := s.initializeLifecyclePolicyUnlocked(limits); err != nil {
		return err
	}
	if err := recoverLifecycleAuditTemps(s.lifecycleAuditRoot); err != nil {
		return err
	}
	if err := recoverLifecycleAuditTemps(s.lifecycleBindingRoot); err != nil {
		return err
	}
	if err := s.validateLifecycleAuditUnlocked(); err != nil {
		return err
	}
	s.lifecycle.evidenceMu.Lock()
	defer s.lifecycle.evidenceMu.Unlock()
	if err := s.recoverCampaignDeletionsUnlocked(); err != nil {
		return err
	}
	if err := recoverStagedCampaigns(s.campaignRoot); err != nil {
		return err
	}
	// A prior Campaign create may have completed its rename before the parent
	// fsync reported failure. Only the first startup owner may close that
	// namespace boundary; a peer opener must not recover another live Service's
	// in-flight publication.
	return s.recoverCampaignPublicationDurability()
}

func ensureDurablePrivateDirectoryTree(path string) error {
	return ensureDurablePrivateDirectoryTreeWithSync(path, syncEvaluationDirectory)
}

func ensureDurablePrivateDirectoryTreeWithSync(
	path string,
	syncDirectory func(string, string) error,
) error {
	path = filepath.Clean(path)
	missing := make([]string, 0, 4)
	for current := path; ; current = filepath.Dir(current) {
		if _, err := os.Lstat(current); err == nil {
			break
		} else if !os.IsNotExist(err) {
			return err
		}
		missing = append(missing, current)
		parent := filepath.Dir(current)
		if parent == current {
			return fmt.Errorf("cannot locate existing parent for evaluation directory")
		}
	}
	for index := len(missing) - 1; index >= 0; index-- {
		directory := missing[index]
		if err := os.Mkdir(directory, 0o700); err != nil && !os.IsExist(err) {
			return err
		}
		if err := requirePrivateDirectory(directory); err != nil {
			return err
		}
		if err := syncDirectory(filepath.Dir(directory), "evaluation directory hierarchy"); err != nil {
			// The directory is still empty at this point. Remove the uncertain
			// namespace entry so a retry must recreate and resync this same parent.
			_ = os.Remove(directory)
			return err
		}
	}
	// Also sync an already-existing final entry. If an earlier process returned
	// after mkdir but its parent fsync failed, this retry closes that exact
	// durability uncertainty before descendants are used.
	return syncDirectory(filepath.Dir(path), "evaluation directory hierarchy retry")
}

func (s *Store) Root() string { return s.root }

func (s *Store) SuiteRoot() string { return s.suiteRoot }

func (s *Store) GetRun(id string) (Run, error) {
	run, _, err := s.getRunWithLifecycleUnlocked(id)
	return run, err
}

func (s *Store) getRunWithLifecycleUnlocked(id string) (Run, RunLifecycle, error) {
	run, readErr := s.getRunUnlocked(id)
	if readErr != nil {
		return Run{}, RunLifecycle{}, readErr
	}
	if err := s.requireRunPublicationDurable(id); err != nil {
		return Run{}, RunLifecycle{}, err
	}
	lifecycle, err := s.readRunLifecycle(run)
	if err != nil {
		return Run{}, RunLifecycle{}, err
	}
	return run, lifecycle, nil
}

// getRunForCreateRetry reads a complete visible bundle without accepting its
// namespace durability. It is intentionally restricted to CreateRun's exact
// actor/request reconciliation path, which owns the keyed parent-sync barrier.
func (s *Store) getRunForCreateRetry(id string) (Run, error) {
	return s.getRunUnlocked(id)
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
	runDir, err := s.checkedRunDirPhysical(id)
	if err != nil {
		return "", err
	}
	if _, _, err := s.controlledPairRunSnapshot(id, runDir); err != nil {
		return "", err
	}
	return runDir, nil
}

func (s *Store) checkedRunDirPhysical(id string) (string, error) {
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

func (s *Store) getRunUnlocked(id string) (Run, error) {
	runDir, err := s.checkedRunDir(id)
	if err != nil {
		return Run{}, err
	}
	if snapshot, visible, err := s.controlledPairRunSnapshot(id, runDir); err != nil {
		return Run{}, err
	} else if visible {
		return snapshot, nil
	}
	var run Run
	if err := readJSON(filepath.Join(runDir, runFileName), &run); err != nil {
		return Run{}, err
	}
	if err := validateStoredRun(id, run); err != nil {
		return Run{}, fmt.Errorf("validate evaluation run status: %w", err)
	}
	return run, nil
}
