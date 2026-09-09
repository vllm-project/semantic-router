package looper

import (
	"path/filepath"
	"sync"
	"time"
)

// workflowFileStoreRegistration is the process-wide owner of one file-backed
// store for a resolved directory. Router reload constructs a new
// WorkflowStateService before the previous generation has drained Put/Take, so
// overlapping constructors must share the store, its lock, and currentBytes.
type workflowFileStoreRegistration struct {
	mu    sync.Mutex
	store *workflowFileToolStateStore
	refs  int
}

var (
	workflowFileStoreRegistryMu sync.Mutex
	workflowFileStoreRegistry   = map[string]*workflowFileStoreRegistration{}
)

func resolvedWorkflowFileStoreDir(dir string) string {
	storeDir := filepath.Clean(workflowStateFileDir(dir))
	abs, err := filepath.Abs(storeDir)
	if err != nil {
		return storeDir
	}
	return abs
}

func workflowFileStoreRegistrationFor(key string) *workflowFileStoreRegistration {
	workflowFileStoreRegistryMu.Lock()
	defer workflowFileStoreRegistryMu.Unlock()
	reg := workflowFileStoreRegistry[key]
	if reg == nil {
		reg = &workflowFileStoreRegistration{}
		workflowFileStoreRegistry[key] = reg
	}
	return reg
}

func newWorkflowFileToolStateStore(dir string, ttl time.Duration) *workflowFileToolStateStore {
	key := resolvedWorkflowFileStoreDir(dir)
	reg := workflowFileStoreRegistrationFor(key)
	reg.mu.Lock()
	defer reg.mu.Unlock()
	if reg.store != nil {
		reg.refs++
		return reg.store
	}

	s := &workflowFileToolStateStore{
		dir:          key,
		ttl:          ttl,
		done:         make(chan struct{}),
		currentBytes: cleanupStateStoreDirAndGetInitialBytes(key),
	}
	s.wg.Add(1)
	go s.sweepLoop()
	reg.store = s
	reg.refs = 1
	return s
}

func (s *workflowFileToolStateStore) Close() error {
	if s == nil {
		return nil
	}
	reg := workflowFileStoreRegistrationFor(resolvedWorkflowFileStoreDir(s.dir))
	reg.mu.Lock()
	defer reg.mu.Unlock()
	if reg.store != s {
		return nil
	}
	if reg.refs > 0 {
		reg.refs--
	}
	if reg.refs > 0 {
		return nil
	}
	err := s.stopSweeper()
	reg.store = nil
	return err
}
