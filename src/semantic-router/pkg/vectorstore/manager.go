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
	"sort"
	"sync"
	"time"
)

// Manager orchestrates vector store CRUD operations and coordinates
// between the in-memory store registry and the backend.
type Manager struct {
	backend            VectorStoreBackend
	mu                 sync.RWMutex
	stores             map[string]*VectorStore // id -> store
	embeddingDim       int
	defaultBackendType string
}

// NewManager creates a new vector store manager.
func NewManager(backend VectorStoreBackend, embeddingDim int, backendType string) *Manager {
	return &Manager{
		backend:            backend,
		stores:             make(map[string]*VectorStore),
		embeddingDim:       embeddingDim,
		defaultBackendType: backendType,
	}
}

// CreateStoreRequest holds parameters for creating a vector store.
type CreateStoreRequest struct {
	Name         string                 `json:"name"`
	ExpiresAfter *ExpirationPolicy      `json:"expires_after,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// UpdateStoreRequest holds parameters for updating a vector store.
type UpdateStoreRequest struct {
	Name         *string                `json:"name,omitempty"`
	ExpiresAfter *ExpirationPolicy      `json:"expires_after,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// ListStoresParams holds parameters for listing vector stores.
type ListStoresParams struct {
	Limit  int    // max results (default 20, max 100)
	Order  string // "asc" or "desc" (default "desc")
	After  string // cursor for pagination
	Before string // cursor for pagination
}

// CreateStore creates a new vector store and its backing collection.
func (m *Manager) CreateStore(ctx context.Context, req CreateStoreRequest) (*VectorStore, error) {
	id := GenerateVectorStoreID()

	if err := m.backend.CreateCollection(ctx, id, m.embeddingDim); err != nil {
		return nil, fmt.Errorf("failed to create backend collection: %w", err)
	}

	vs := &VectorStore{
		ID:           id,
		Object:       "vector_store",
		Name:         req.Name,
		CreatedAt:    time.Now().Unix(),
		Status:       "active",
		FileCounts:   FileCounts{},
		ExpiresAfter: req.ExpiresAfter,
		Metadata:     req.Metadata,
		BackendType:  m.defaultBackendType,
	}

	m.mu.Lock()
	m.stores[id] = vs
	m.mu.Unlock()

	return vs, nil
}

// GetStore returns a vector store by ID.
func (m *Manager) GetStore(id string) (*VectorStore, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	vs, ok := m.stores[id]
	if !ok {
		return nil, fmt.Errorf("vector store not found: %s", id)
	}
	return vs, nil
}

// ListStores returns vector stores with pagination.
func (m *Manager) ListStores(params ListStoresParams) []*VectorStore {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if params.Limit <= 0 {
		params.Limit = 20
	}
	if params.Limit > 100 {
		params.Limit = 100
	}

	// Collect all stores into a slice.
	all := make([]*VectorStore, 0, len(m.stores))
	for _, vs := range m.stores {
		all = append(all, vs)
	}

	// Sort by created_at.
	if params.Order == "asc" {
		sort.Slice(all, func(i, j int) bool { return all[i].CreatedAt < all[j].CreatedAt })
	} else {
		sort.Slice(all, func(i, j int) bool { return all[i].CreatedAt > all[j].CreatedAt })
	}

	// Apply cursor-based pagination.
	startIdx := 0
	if params.After != "" {
		for i, vs := range all {
			if vs.ID == params.After {
				startIdx = i + 1
				break
			}
		}
	}

	if startIdx >= len(all) {
		return nil
	}

	end := startIdx + params.Limit
	if end > len(all) {
		end = len(all)
	}

	return all[startIdx:end]
}

// UpdateStore updates a vector store's metadata.
func (m *Manager) UpdateStore(id string, req UpdateStoreRequest) (*VectorStore, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	vs, ok := m.stores[id]
	if !ok {
		return nil, fmt.Errorf("vector store not found: %s", id)
	}

	if req.Name != nil {
		vs.Name = *req.Name
	}
	if req.ExpiresAfter != nil {
		vs.ExpiresAfter = req.ExpiresAfter
	}
	if req.Metadata != nil {
		vs.Metadata = req.Metadata
	}

	return vs, nil
}

// DeleteStore deletes a vector store and its backing collection.
func (m *Manager) DeleteStore(ctx context.Context, id string) error {
	m.mu.Lock()
	_, ok := m.stores[id]
	if !ok {
		m.mu.Unlock()
		return fmt.Errorf("vector store not found: %s", id)
	}
	delete(m.stores, id)
	m.mu.Unlock()

	if err := m.backend.DeleteCollection(ctx, id); err != nil {
		return fmt.Errorf("failed to delete backend collection: %w", err)
	}

	return nil
}

// Backend returns the underlying vector store backend for direct operations.
func (m *Manager) Backend() VectorStoreBackend {
	return m.backend
}

// UpdateFileCounts updates the file counts for a vector store.
func (m *Manager) UpdateFileCounts(id string, fn func(*FileCounts)) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	vs, ok := m.stores[id]
	if !ok {
		return fmt.Errorf("vector store not found: %s", id)
	}

	fn(&vs.FileCounts)
	return nil
}
