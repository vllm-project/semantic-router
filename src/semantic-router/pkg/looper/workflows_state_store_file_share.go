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
	return resolveExistingDirPrefix(abs)
}

// resolveExistingDirPrefix canonicalizes directory identity across symlink
// aliases. EvalSymlinks requires an existing path, so a missing leaf is
// joined onto the longest existing resolved prefix.
func resolveExistingDirPrefix(abs string) string {
	if resolved, err := filepath.EvalSymlinks(abs); err == nil {
		return resolved
	}
	var missing []string
	cur := abs
	for {
		parent := filepath.Dir(cur)
		if parent == cur {
			return abs
		}
		missing = append([]string{filepath.Base(cur)}, missing...)
		if resolved, err := filepath.EvalSymlinks(parent); err == nil {
			return filepath.Join(append([]string{resolved}, missing...)...)
		}
		cur = parent
	}
}

func newWorkflowFileToolStateStore(dir string, ttl time.Duration) *workflowFileToolStateStore {
	key := resolvedWorkflowFileStoreDir(dir)
	workflowFileStoreRegistryMu.Lock()
	defer workflowFileStoreRegistryMu.Unlock()
	if reg := workflowFileStoreRegistry[key]; reg != nil && reg.store != nil {
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
	workflowFileStoreRegistry[key] = &workflowFileStoreRegistration{store: s, refs: 1}
	return s
}

func (s *workflowFileToolStateStore) Close() error {
	if s == nil {
		return nil
	}
	key := resolvedWorkflowFileStoreDir(s.dir)
	workflowFileStoreRegistryMu.Lock()
	defer workflowFileStoreRegistryMu.Unlock()
	reg := workflowFileStoreRegistry[key]
	if reg == nil || reg.store != s {
		return nil
	}
	if reg.refs > 0 {
		reg.refs--
	}
	if reg.refs > 0 {
		return nil
	}
	delete(workflowFileStoreRegistry, key)
	return s.stopSweeper()
}
