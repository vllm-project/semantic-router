/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package lookuptable

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// StoredLookupTables is the on-disk JSON structure persisted by FileStorage.
type StoredLookupTables struct {
	// Version allows future format migrations.
	Version int `json:"version"`

	// LastUpdated records when the file was last written.
	LastUpdated time.Time `json:"last_updated"`

	// Entries maps canonical key strings to their entries.
	Entries map[string]Entry `json:"entries"`
}

// FileStorage is a file-backed LookupTableStorage. It writes JSON atomically
// (write to .tmp → rename) and supports background auto-save.
//
// The in-memory cache is authoritative; the file is only read during Load().
type FileStorage struct {
	path string

	mu      sync.RWMutex
	entries map[string]Entry

	dirty           atomic.Bool
	autoSaveStarted atomic.Bool
	stopChan        chan struct{}
	doneChan        chan struct{}
}

// NewFileStorage creates a FileStorage backed by the given path.
// The directory is created if it does not exist.
// Call Load() to populate the cache from an existing file.
func NewFileStorage(path string) (*FileStorage, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("lookuptable: failed to create storage directory: %w", err)
	}

	s := &FileStorage{
		path:     path,
		entries:  make(map[string]Entry),
		stopChan: make(chan struct{}),
		doneChan: make(chan struct{}),
	}

	logging.ComponentEvent("selection", "lookuptable_file_storage_initialized", map[string]interface{}{
		"storage_path": path,
	})
	return s, nil
}

// Get implements LookupTable.
func (f *FileStorage) Get(key Key) (Entry, bool) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	e, ok := f.entries[key.String()]
	return e, ok
}

// QualityGap implements LookupTable.
func (f *FileStorage) QualityGap(taskFamily, currentModel, candidateModel string) (float64, bool) {
	e, ok := f.Get(QualityGapKey(taskFamily, currentModel, candidateModel))
	return e.Value, ok
}

// HandoffPenalty implements LookupTable.
func (f *FileStorage) HandoffPenalty(fromModel, toModel string) (float64, bool) {
	e, ok := f.Get(HandoffPenaltyKey(fromModel, toModel))
	return e.Value, ok
}

// RemainingTurnPrior implements LookupTable.
func (f *FileStorage) RemainingTurnPrior(intentOrDomain string) (float64, bool) {
	e, ok := f.Get(RemainingTurnPriorKey(intentOrDomain))
	return e.Value, ok
}

// Set implements LookupTableStorage.
func (f *FileStorage) Set(key Key, entry Entry) error {
	f.mu.Lock()
	f.entries[key.String()] = entry
	f.mu.Unlock()
	f.dirty.Store(true)
	return nil
}

// SetBatch implements LookupTableStorage.
func (f *FileStorage) SetBatch(entries map[Key]Entry) error {
	f.mu.Lock()
	for k, v := range entries {
		f.entries[k.String()] = v
	}
	f.mu.Unlock()
	f.dirty.Store(true)
	return nil
}

// Delete implements LookupTableStorage.
func (f *FileStorage) Delete(key Key) error {
	f.mu.Lock()
	delete(f.entries, key.String())
	f.mu.Unlock()
	f.dirty.Store(true)
	return nil
}

// All implements LookupTableStorage.
func (f *FileStorage) All() map[string]Entry {
	f.mu.RLock()
	defer f.mu.RUnlock()
	cp := make(map[string]Entry, len(f.entries))
	for k, v := range f.entries {
		cp[k] = v
	}
	return cp
}

// Load reads the backing file and populates the in-memory cache.
// It is safe to call Load even if the file does not exist (returns nil).
func (f *FileStorage) Load() error {
	f.mu.Lock()
	defer f.mu.Unlock()

	stored, err := f.readFile()
	if err != nil {
		return err
	}
	f.entries = stored.Entries
	return nil
}

// Save flushes the in-memory cache to the backing file atomically.
func (f *FileStorage) Save() error {
	f.mu.RLock()
	cp := make(map[string]Entry, len(f.entries))
	for k, v := range f.entries {
		cp[k] = v
	}
	f.mu.RUnlock()

	return f.writeFile(cp)
}

// MarkDirty signals that entries have changed and a save is needed.
func (f *FileStorage) MarkDirty() {
	f.dirty.Store(true)
}

// StartAutoSave launches a background goroutine that saves whenever the dirty
// flag is set, on the given interval. Call Close() to stop it.
// Subsequent calls after the first are no-ops.
func (f *FileStorage) StartAutoSave(interval time.Duration) {
	if !f.autoSaveStarted.CompareAndSwap(false, true) {
		return
	}
	go func() {
		defer close(f.doneChan)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-f.stopChan:
				// Final save before shutdown.
				if f.dirty.Swap(false) {
					if err := f.Save(); err != nil {
						logging.Errorf("[LookupTableStorage] Final save failed: %v", err)
					}
				}
				return
			case <-ticker.C:
				if f.dirty.Swap(false) {
					if err := f.Save(); err != nil {
						f.dirty.Store(true)
						logging.Errorf("[LookupTableStorage] Auto-save failed: %v", err)
					}
				}
			}
		}
	}()
}

// Close stops the auto-save goroutine (if running) and releases resources.
func (f *FileStorage) Close() error {
	select {
	case <-f.stopChan:
		// Already closed.
		return nil
	default:
		close(f.stopChan)
	}
	// Wait for the goroutine only if StartAutoSave was called.
	if f.autoSaveStarted.Load() {
		<-f.doneChan
	}
	logging.Infof("[LookupTableStorage] Closed file storage: %s", f.path)
	return nil
}

// readFile reads and parses the backing file, returning an empty structure
// if the file does not exist or is empty.
func (f *FileStorage) readFile() (*StoredLookupTables, error) {
	data, err := os.ReadFile(f.path)
	if err != nil {
		if os.IsNotExist(err) {
			return emptyStored(), nil
		}
		return nil, fmt.Errorf("lookuptable: failed to read %s: %w", f.path, err)
	}

	if len(data) == 0 {
		logging.Warnf("[LookupTableStorage] Storage file is empty, starting fresh: %s", f.path)
		return emptyStored(), nil
	}

	var stored StoredLookupTables
	if err := json.Unmarshal(data, &stored); err != nil {
		// Back up the corrupted file before returning the error.
		backupPath := f.path + ".corrupted"
		if writeErr := os.WriteFile(backupPath, data, 0o644); writeErr == nil {
			logging.Errorf("[LookupTableStorage] Corrupted file backed up to: %s", backupPath)
		}
		return nil, fmt.Errorf("lookuptable: failed to parse %s (backup at %s.corrupted): %w", f.path, f.path, err)
	}

	if stored.Entries == nil {
		stored.Entries = make(map[string]Entry)
	}
	return &stored, nil
}

// writeFile marshals entries and writes them atomically (temp → rename).
func (f *FileStorage) writeFile(entries map[string]Entry) error {
	stored := &StoredLookupTables{
		Version:     1,
		LastUpdated: time.Now(),
		Entries:     entries,
	}

	data, err := json.MarshalIndent(stored, "", "  ")
	if err != nil {
		return fmt.Errorf("lookuptable: failed to marshal entries: %w", err)
	}

	tmpPath := f.path + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0o600); err != nil {
		return fmt.Errorf("lookuptable: failed to write temp file: %w", err)
	}

	if err := os.Rename(tmpPath, f.path); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("lookuptable: failed to rename temp file: %w", err)
	}

	logging.Debugf("[LookupTableStorage] Saved %d entries to %s", len(entries), f.path)
	return nil
}

func emptyStored() *StoredLookupTables {
	return &StoredLookupTables{
		Version: 1,
		Entries: make(map[string]Entry),
	}
}
