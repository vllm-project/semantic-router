package store

import (
	"bufio"
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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// FileStore is a MemoryStore that writes each record to a JSON Lines file
// once, when it ends (completed, failed or aborted), and reads the ended
// records back when it opens: the last line of each, at most maxRecords of
// them. A record still in progress is not on disk. What reaches an ended
// record later, such as a late outcome or a review, is written when the file
// is next rewritten: once it holds twice maxRecords lines, and when the store
// closes. Reading the file back and rewriting it drop records older than the
// TTL, which the running store still lists, as the memory store does. A line
// cut short by a crash is skipped. The file is readable by its owner only, and
// one process at a time holds it.
type FileStore struct {
	*MemoryStore

	mu    sync.Mutex
	path  string
	file  *os.File
	lock  *os.File
	lines int
	limit int
	refs  int
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
	path, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	openFileStores.Lock()
	defer openFileStores.Unlock()
	if store := openFileStores.byPath[path]; store != nil {
		store.refs++
		store.SetMaxRecords(maxRecords)
		store.mu.Lock()
		store.limit = store.maxRecords
		store.mu.Unlock()
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
	openFileStores.byPath[path] = store
	return store, nil
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
		if strings.HasPrefix(entry.Name(), rewritePrefix) {
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

// rewritePrefix names the temporary file of a rewrite.
const rewritePrefix = ".router-replay-"

// expired reports whether a record is older than the store's TTL.
func (f *FileStore) expired(record Record, now time.Time) bool {
	return f.ttl > 0 && record.Timestamp.Before(now.Add(-time.Duration(f.ttl)*time.Second))
}

// appendLocked appends the record id holds to the file.
func (f *FileStore) appendLocked(id string) error {
	record, found, _ := f.MemoryStore.Get(context.Background(), id)
	if !found {
		return nil
	}
	line, err := json.Marshal(record)
	if err != nil {
		return err
	}
	if _, err = f.file.Write(append(line, '\n')); err != nil {
		return err
	}
	// The record is written; a failed rewrite leaves the longer file in place
	// and is tried again after another limit lines.
	if f.lines++; f.lines >= 2*f.limit {
		if err = f.compactLocked(); err != nil {
			f.lines = f.limit
			logging.Warnf("router replay could not rewrite %s: %v", f.path, err)
		}
	}
	return nil
}

// compactLocked replaces the file with one line for each record kept.
func (f *FileStore) compactLocked() error {
	records, err := f.MemoryStore.List(context.Background())
	if err != nil {
		return err
	}
	dir := filepath.Dir(f.path)
	tmp, err := os.CreateTemp(dir, rewritePrefix+"*")
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(tmp.Name()) }()
	writer, lines, now := bufio.NewWriter(tmp), 0, time.Now()
	for i := len(records) - 1; i >= 0 && err == nil; i-- {
		if !recordEnded(records[i].LifecycleState) || f.expired(records[i], now) {
			continue
		}
		var line []byte
		if line, err = json.Marshal(records[i]); err == nil {
			_, err = writer.Write(append(line, '\n'))
			lines++
		}
	}
	if err = errors.Join(err, writer.Flush(), tmp.Sync(), tmp.Close()); err != nil {
		return err
	}
	if err = os.Rename(tmp.Name(), f.path); err != nil {
		return err
	}
	if err = syncDir(dir); err != nil {
		return err
	}
	file, err := os.OpenFile(f.path, os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	_ = f.file.Close()
	f.file, f.lines = file, lines
	return nil
}

// Add keeps the record, and writes it now when it has already ended.
func (f *FileStore) Add(ctx context.Context, record Record) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.file == nil {
		return "", ErrStorageClosed
	}
	id, err := f.MemoryStore.Add(ctx, record)
	if err == nil && recordEnded(record.LifecycleState) {
		err = f.appendLocked(id)
	}
	return id, err
}

// UpdateLifecycle ends the record, and writes it when this call ended it.
func (f *FileStore) UpdateLifecycle(ctx context.Context, id string, state string, endedAt time.Time, durationMS int64, reason string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.file == nil {
		return ErrStorageClosed
	}
	ended := f.ended(id)
	if err := f.MemoryStore.UpdateLifecycle(ctx, id, state, endedAt, durationMS, reason); err != nil || ended || !f.ended(id) {
		return err
	}
	return f.appendLocked(id)
}

// ended reports whether the record id holds has ended.
func (f *FileStore) ended(id string) bool {
	f.MemoryStore.mu.RLock()
	defer f.MemoryStore.mu.RUnlock()
	record := f.byID[id]
	return record != nil && recordEnded(record.LifecycleState)
}

func recordEnded(state string) bool {
	return state == LifecycleCompleted || state == LifecycleFailed || state == LifecycleAborted
}

// Close releases this user's share of the store. Once no Router generation
// uses it, it rewrites the file with the ended records, as they are now, and
// closes it.
func (f *FileStore) Close() error {
	openFileStores.Lock()
	defer openFileStores.Unlock()
	if f.refs == 0 {
		return nil
	}
	if f.refs--; f.refs > 0 {
		return nil
	}
	delete(openFileStores.byPath, f.path)
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.file == nil {
		return nil
	}
	err := errors.Join(f.compactLocked(), f.file.Close(), f.lock.Close())
	f.file, f.lock = nil, nil
	return err
}

// syncDir makes a rename in dir durable.
func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	return errors.Join(d.Sync(), d.Close())
}
