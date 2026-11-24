package store

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
)

// FileStore keeps live records in memory and persists ended records as JSON Lines.
// One writer per path appends in batches and checkpoints the retained history.
type FileStore struct {
	*MemoryStore

	path string
	lock *os.File
	refs int

	mu             sync.Mutex
	limit          int
	pending        []Record
	rewritePending bool
	writerStarted  bool
	closed         bool
	wake           chan struct{}
	stop           chan struct{}
	done           chan struct{}

	file     *os.File
	lines    int
	closeErr error
}

// openFileStores lets every Router generation of a process share the store of
// a path, so that a configuration reload keeps the records and no generation
// rewrites the file another still appends to.
var openFileStores = struct {
	sync.Mutex
	byPath map[string]*FileStore
}{byPath: map[string]*FileStore{}}

// OpenFileStore opens the store kept at path, creating its directory private
// to its owner, or shares the one this process has open there. It fails while
// another process holds the file.
func OpenFileStore(path string, maxRecords, ttlSeconds int) (*FileStore, error) {
	return openFileStore(path, maxRecords, ttlSeconds, true)
}

// PrepareFileStore opens or shares a store for a Router candidate. loadLimit
// bounds the history loaded on a new open. The candidate must commit its
// retention policy before serving, or close its reference without committing.
func PrepareFileStore(path string, loadLimit, ttlSeconds int) (*FileStore, error) {
	return openFileStore(path, loadLimit, ttlSeconds, false)
}

func openFileStore(path string, maxRecords, ttlSeconds int, activate bool) (*FileStore, error) {
	path, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	openFileStores.Lock()
	defer openFileStores.Unlock()
	if store := openFileStores.byPath[path]; store != nil {
		store.refs++
		if activate {
			store.CommitRetention(maxRecords)
		}
		return store, nil
	}
	store := &FileStore{MemoryStore: NewMemoryStore(maxRecords, ttlSeconds), path: path, refs: 1}
	store.limit = store.maxRecords
	if err = os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, err
	}
	if store.lock, err = lockFile(path + ".lock"); err != nil {
		return nil, err
	}
	if err = store.load(); err != nil {
		_ = store.lock.Close()
		return nil, err
	}
	store.wake = make(chan struct{}, 1)
	store.stop = make(chan struct{})
	store.done = make(chan struct{})
	if activate {
		store.CommitRetention(maxRecords)
	}
	openFileStores.byPath[path] = store
	return store, nil
}

// CommitRetention applies the serving Router's record limit. For a newly
// prepared store it also starts the writer, so discarding a candidate that
// opened the file cannot checkpoint its provisional history.
func (f *FileStore) CommitRetention(maxRecords int) {
	if maxRecords <= 0 {
		maxRecords = DefaultMaxRecords
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.MemoryStore.SetMaxRecords(maxRecords)
	f.limit = maxRecords
	if !f.writerStarted {
		f.writerStarted = true
		go f.writeLoop()
	}
}

// lockFile takes the lock that keeps a second process off a store's file. It
// is a file of its own, which no rewrite replaces, and the lock is released
// when the process exits.
func lockFile(path string) (*os.File, error) {
	lock, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	if err = syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		_ = lock.Close()
		return nil, fmt.Errorf("another process holds %s: %w", path, err)
	}
	return lock, nil
}

// load removes what an interrupted rewrite left, reads the records back and
// opens the file for appending.
func (f *FileStore) load() error {
	dir := filepath.Dir(f.path)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), f.rewritePrefix()) {
			_ = os.Remove(filepath.Join(dir, entry.Name()))
		}
	}
	data, err := os.ReadFile(f.path)
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	latest := map[string]Record{}
	var order []string
	for _, line := range bytes.Split(data, []byte{'\n'}) {
		var record Record
		if len(line) == 0 {
			continue
		}
		f.lines++
		if json.Unmarshal(line, &record) != nil || record.ID == "" || !recordEnded(record.LifecycleState) {
			continue
		}
		if _, seen := latest[record.ID]; !seen {
			order = append(order, record.ID)
		}
		latest[record.ID] = record
	}
	now := time.Now()
	for _, id := range order {
		if record := latest[id]; !f.expired(record, now) {
			if _, err = f.MemoryStore.Add(context.Background(), record); err != nil {
				return err
			}
		}
	}
	if f.file, err = os.OpenFile(f.path, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o600); err != nil {
		return err
	}
	err = f.file.Chmod(0o600)
	if err == nil && len(data) > 0 && data[len(data)-1] != '\n' {
		_, err = f.file.Write([]byte{'\n'})
	}
	if err != nil {
		_ = f.file.Close()
	}
	return err
}

func (f *FileStore) rewritePrefix() string {
	if filepath.Base(f.path) == "router-replay.jsonl" {
		return ".router-replay-"
	}
	return filepath.Base(f.path) + ".rewrite-"
}

// expired reports whether a record is older than the store's TTL.
func (f *FileStore) expired(record Record, now time.Time) bool {
	return f.ttl > 0 && record.Timestamp.Before(now.Add(-time.Duration(f.ttl)*time.Second))
}

// Add publishes the record immediately and queues it if it has already ended.
func (f *FileStore) Add(ctx context.Context, record Record) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.closed {
		return "", ErrStorageClosed
	}
	id, err := f.MemoryStore.Add(ctx, record)
	if err == nil && recordEnded(record.LifecycleState) {
		f.queueRecord(id)
	}
	return id, err
}

// UpdateLifecycle queues the first transition to a terminal state.
func (f *FileStore) UpdateLifecycle(ctx context.Context, id string, state string, endedAt time.Time, durationMS int64, reason string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.closed {
		return ErrStorageClosed
	}
	changed, err := f.MemoryStore.updateLifecycle(id, state, endedAt, durationMS, reason)
	if changed && recordEnded(state) {
		f.queueRecord(id)
	}
	return err
}

func recordEnded(state string) bool {
	return state == LifecycleCompleted || state == LifecycleFailed || state == LifecycleAborted
}

// Close releases this user's share of the store. Once no Router generation
// uses it, it checkpoints the ended records and closes the file. A prepared
// store that never served closes without a checkpoint.
func (f *FileStore) Close() error {
	openFileStores.Lock()
	defer openFileStores.Unlock()
	if f.refs == 0 {
		return f.closeErr
	}
	if f.refs--; f.refs > 0 {
		return nil
	}
	delete(openFileStores.byPath, f.path)
	f.mu.Lock()
	f.closed = true
	writerStarted := f.writerStarted
	close(f.stop)
	f.mu.Unlock()
	if !writerStarted {
		f.closeErr = errors.Join(f.file.Close(), f.lock.Close())
		return f.closeErr
	}
	<-f.done
	return f.closeErr
}

// syncDir makes a rename in dir durable.
func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	return errors.Join(d.Sync(), d.Close())
}
