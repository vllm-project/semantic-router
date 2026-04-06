package vectorstore

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

const (
	localMetadataRegistryFile = ".vectorstore-metadata.json"
	restartInterruptedCode    = "interrupted"
)

type localMetadataState struct {
	Stores   map[string]*VectorStore     `json:"stores,omitempty"`
	Files    map[string]*FileRecord      `json:"files,omitempty"`
	Statuses map[string]*VectorStoreFile `json:"statuses,omitempty"`
}

// LocalMetadataRegistry persists vector-store control-plane metadata under the
// file-storage root so store/file/status records can be recovered after a
// router restart without introducing a second mutable runtime source of truth.
type LocalMetadataRegistry struct {
	path  string
	mu    sync.Mutex
	state localMetadataState
}

func NewLocalMetadataRegistry(baseDir string) (*LocalMetadataRegistry, error) {
	if err := os.MkdirAll(baseDir, 0o755); err != nil {
		return nil, fmt.Errorf("failed to create vector store metadata directory: %w", err)
	}

	registry := &LocalMetadataRegistry{
		path: filepath.Join(baseDir, localMetadataRegistryFile),
		state: localMetadataState{
			Stores:   map[string]*VectorStore{},
			Files:    map[string]*FileRecord{},
			Statuses: map[string]*VectorStoreFile{},
		},
	}
	if err := registry.load(); err != nil {
		return nil, err
	}
	return registry, nil
}

func (r *LocalMetadataRegistry) load() error {
	data, err := os.ReadFile(r.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("failed to read vector store metadata registry: %w", err)
	}

	var state localMetadataState
	if err := json.Unmarshal(data, &state); err != nil {
		return fmt.Errorf("failed to decode vector store metadata registry: %w", err)
	}

	r.state.Stores = cloneStoreMap(state.Stores)
	r.state.Files = cloneFileRecordMap(state.Files)
	r.state.Statuses = cloneStatusMap(state.Statuses)
	return nil
}

func (r *LocalMetadataRegistry) persistLocked() error {
	data, err := json.MarshalIndent(r.state, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to encode vector store metadata registry: %w", err)
	}

	tmpPath := r.path + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0o644); err != nil {
		return fmt.Errorf("failed to write vector store metadata registry: %w", err)
	}
	if err := os.Rename(tmpPath, r.path); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("failed to replace vector store metadata registry: %w", err)
	}
	return nil
}

func (r *LocalMetadataRegistry) Stores() map[string]*VectorStore {
	if r == nil {
		return map[string]*VectorStore{}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return cloneStoreMap(r.state.Stores)
}

func (r *LocalMetadataRegistry) Files() map[string]*FileRecord {
	if r == nil {
		return map[string]*FileRecord{}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return cloneFileRecordMap(r.state.Files)
}

func (r *LocalMetadataRegistry) Statuses() map[string]*VectorStoreFile {
	if r == nil {
		return map[string]*VectorStoreFile{}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return cloneStatusMap(r.state.Statuses)
}

func (r *LocalMetadataRegistry) UpsertStore(store *VectorStore) error {
	if r == nil || store == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.state.Stores[store.ID] = cloneVectorStore(store)
	return r.persistLocked()
}

func (r *LocalMetadataRegistry) ReplaceStores(stores map[string]*VectorStore) error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.state.Stores = cloneStoreMap(stores)
	return r.persistLocked()
}

func (r *LocalMetadataRegistry) DeleteStore(id string) error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.state.Stores, id)
	for statusID, status := range r.state.Statuses {
		if status.VectorStoreID == id {
			delete(r.state.Statuses, statusID)
		}
	}
	return r.persistLocked()
}

func (r *LocalMetadataRegistry) UpsertFile(record *FileRecord) error {
	if r == nil || record == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.state.Files[record.ID] = cloneFileRecord(record)
	return r.persistLocked()
}

func (r *LocalMetadataRegistry) DeleteFile(id string) error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.state.Files, id)
	return r.persistLocked()
}

func (r *LocalMetadataRegistry) UpsertStatus(status *VectorStoreFile) error {
	if r == nil || status == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.state.Statuses[status.ID] = cloneVectorStoreFile(status)
	return r.persistLocked()
}

func (r *LocalMetadataRegistry) DeleteStatus(id string) error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.state.Statuses, id)
	return r.persistLocked()
}

func (r *LocalMetadataRegistry) RecoverInterruptedStatuses(message string) (map[string]*VectorStoreFile, error) {
	if r == nil {
		return map[string]*VectorStoreFile{}, nil
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	changed := false
	for id, status := range r.state.Statuses {
		if status.Status != "in_progress" {
			continue
		}
		next := cloneVectorStoreFile(status)
		next.Status = "failed"
		next.LastError = &FileError{
			Code:    restartInterruptedCode,
			Message: message,
		}
		r.state.Statuses[id] = next
		changed = true
	}

	if changed {
		if err := r.persistLocked(); err != nil {
			return nil, err
		}
	}

	return cloneStatusMap(r.state.Statuses), nil
}

func cloneStoreMap(input map[string]*VectorStore) map[string]*VectorStore {
	if len(input) == 0 {
		return map[string]*VectorStore{}
	}
	out := make(map[string]*VectorStore, len(input))
	for key, store := range input {
		out[key] = cloneVectorStore(store)
	}
	return out
}

func cloneFileRecordMap(input map[string]*FileRecord) map[string]*FileRecord {
	if len(input) == 0 {
		return map[string]*FileRecord{}
	}
	out := make(map[string]*FileRecord, len(input))
	for key, record := range input {
		out[key] = cloneFileRecord(record)
	}
	return out
}

func cloneStatusMap(input map[string]*VectorStoreFile) map[string]*VectorStoreFile {
	if len(input) == 0 {
		return map[string]*VectorStoreFile{}
	}
	out := make(map[string]*VectorStoreFile, len(input))
	for key, status := range input {
		out[key] = cloneVectorStoreFile(status)
	}
	return out
}

func cloneVectorStore(store *VectorStore) *VectorStore {
	if store == nil {
		return nil
	}
	clone := *store
	clone.Metadata = cloneInterfaceMap(store.Metadata)
	if store.ExpiresAfter != nil {
		expires := *store.ExpiresAfter
		clone.ExpiresAfter = &expires
	}
	return &clone
}

func cloneFileRecord(record *FileRecord) *FileRecord {
	if record == nil {
		return nil
	}
	clone := *record
	return &clone
}

func cloneVectorStoreFile(status *VectorStoreFile) *VectorStoreFile {
	if status == nil {
		return nil
	}
	clone := *status
	if status.ChunkingStrategy != nil {
		strategy := *status.ChunkingStrategy
		if status.ChunkingStrategy.Static != nil {
			static := *status.ChunkingStrategy.Static
			strategy.Static = &static
		}
		clone.ChunkingStrategy = &strategy
	}
	if status.LastError != nil {
		lastError := *status.LastError
		clone.LastError = &lastError
	}
	return &clone
}

func cloneInterfaceMap(input map[string]interface{}) map[string]interface{} {
	if len(input) == 0 {
		return nil
	}
	data, err := json.Marshal(input)
	if err != nil {
		out := make(map[string]interface{}, len(input))
		for key, value := range input {
			out[key] = value
		}
		return out
	}
	var out map[string]interface{}
	if err := json.Unmarshal(data, &out); err != nil {
		out = make(map[string]interface{}, len(input))
		for key, value := range input {
			out[key] = value
		}
	}
	return out
}
