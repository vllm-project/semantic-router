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

package vectorstore

import (
	"context"
	"fmt"
	"sync"
)

// MemoryMetadataRegistry is an ephemeral StoreRegistry/FileRegistry
// backed by in-memory maps. Data does not survive process restarts.
type MemoryMetadataRegistry struct {
	mu     sync.RWMutex
	stores map[string]*VectorStore
	files  map[string]*FileRecord
}

// NewMemoryMetadataRegistry creates a registry that keeps metadata in
// process memory only (no durability).
func NewMemoryMetadataRegistry() *MemoryMetadataRegistry {
	return &MemoryMetadataRegistry{
		stores: make(map[string]*VectorStore),
		files:  make(map[string]*FileRecord),
	}
}

func (r *MemoryMetadataRegistry) SaveStore(_ context.Context, vs *VectorStore) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.stores[vs.ID] = vs
	return nil
}

func (r *MemoryMetadataRegistry) GetStore(_ context.Context, id string) (*VectorStore, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	vs, ok := r.stores[id]
	if !ok {
		return nil, fmt.Errorf("vector store not found: %s", id)
	}
	return vs, nil
}

func (r *MemoryMetadataRegistry) ListStores(_ context.Context) ([]*VectorStore, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*VectorStore, 0, len(r.stores))
	for _, vs := range r.stores {
		out = append(out, vs)
	}
	return out, nil
}

func (r *MemoryMetadataRegistry) DeleteStore(_ context.Context, id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.stores, id)
	return nil
}

func (r *MemoryMetadataRegistry) SaveFile(_ context.Context, fr *FileRecord) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.files[fr.ID] = fr
	return nil
}

func (r *MemoryMetadataRegistry) GetFile(_ context.Context, id string) (*FileRecord, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	fr, ok := r.files[id]
	if !ok {
		return nil, fmt.Errorf("file not found: %s", id)
	}
	return fr, nil
}

func (r *MemoryMetadataRegistry) ListFiles(_ context.Context) ([]*FileRecord, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*FileRecord, 0, len(r.files))
	for _, fr := range r.files {
		out = append(out, fr)
	}
	return out, nil
}

func (r *MemoryMetadataRegistry) DeleteFile(_ context.Context, id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.files, id)
	return nil
}
