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
	"math"
	"sort"
	"sync"
)

// MemoryBackendConfig holds configuration for the in-memory backend.
type MemoryBackendConfig struct {
	MaxEntriesPerStore int // Maximum entries per collection (0 = unlimited)
}

// memoryCollection holds the data for one in-memory vector store collection.
type memoryCollection struct {
	dimension int
	chunks    map[string]EmbeddedChunk // chunk ID -> chunk
}

// MemoryBackend implements VectorStoreBackend using in-memory storage
// with brute-force cosine similarity search. Intended for development
// and testing — data is not persisted across restarts.
type MemoryBackend struct {
	mu          sync.RWMutex
	collections map[string]*memoryCollection // vectorStoreID -> collection
	maxEntries  int
}

// NewMemoryBackend creates a new in-memory vector store backend.
func NewMemoryBackend(cfg MemoryBackendConfig) *MemoryBackend {
	return &MemoryBackend{
		collections: make(map[string]*memoryCollection),
		maxEntries:  cfg.MaxEntriesPerStore,
	}
}

// CreateCollection creates a new in-memory collection.
func (m *MemoryBackend) CreateCollection(_ context.Context, vectorStoreID string, dimension int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.collections[vectorStoreID]; exists {
		return fmt.Errorf("collection already exists: %s", vectorStoreID)
	}

	m.collections[vectorStoreID] = &memoryCollection{
		dimension: dimension,
		chunks:    make(map[string]EmbeddedChunk),
	}
	return nil
}

// DeleteCollection removes an in-memory collection.
func (m *MemoryBackend) DeleteCollection(_ context.Context, vectorStoreID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.collections[vectorStoreID]; !exists {
		return fmt.Errorf("collection not found: %s", vectorStoreID)
	}

	delete(m.collections, vectorStoreID)
	return nil
}

// CollectionExists checks if a collection exists.
func (m *MemoryBackend) CollectionExists(_ context.Context, vectorStoreID string) (bool, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	_, exists := m.collections[vectorStoreID]
	return exists, nil
}

// InsertChunks inserts embedded chunks into the collection.
func (m *MemoryBackend) InsertChunks(_ context.Context, vectorStoreID string, chunks []EmbeddedChunk) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	col, exists := m.collections[vectorStoreID]
	if !exists {
		return fmt.Errorf("collection not found: %s", vectorStoreID)
	}

	for _, chunk := range chunks {
		if m.maxEntries > 0 && len(col.chunks) >= m.maxEntries {
			return fmt.Errorf("collection %s has reached maximum entries (%d)", vectorStoreID, m.maxEntries)
		}
		col.chunks[chunk.ID] = chunk
	}

	return nil
}

// DeleteByFileID removes all chunks associated with a file.
func (m *MemoryBackend) DeleteByFileID(_ context.Context, vectorStoreID string, fileID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	col, exists := m.collections[vectorStoreID]
	if !exists {
		return fmt.Errorf("collection not found: %s", vectorStoreID)
	}

	for id, chunk := range col.chunks {
		if chunk.FileID == fileID {
			delete(col.chunks, id)
		}
	}

	return nil
}

// Search performs brute-force cosine similarity search.
func (m *MemoryBackend) Search(
	_ context.Context, vectorStoreID string, queryEmbedding []float32,
	topK int, threshold float32, filter map[string]interface{},
) ([]SearchResult, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	col, exists := m.collections[vectorStoreID]
	if !exists {
		return nil, fmt.Errorf("collection not found: %s", vectorStoreID)
	}

	// Extract optional file_id filter.
	var filterFileID string
	if filter != nil {
		if fid, ok := filter["file_id"].(string); ok {
			filterFileID = fid
		}
	}

	type scored struct {
		result SearchResult
		score  float64
	}

	var candidates []scored
	for _, chunk := range col.chunks {
		if filterFileID != "" && chunk.FileID != filterFileID {
			continue
		}

		sim := cosineSimilarity(queryEmbedding, chunk.Embedding)
		if sim >= float64(threshold) {
			candidates = append(candidates, scored{
				result: SearchResult{
					FileID:     chunk.FileID,
					Filename:   chunk.Filename,
					Content:    chunk.Content,
					Score:      sim,
					ChunkIndex: chunk.ChunkIndex,
				},
				score: sim,
			})
		}
	}

	// Sort by score descending.
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	if topK > 0 && len(candidates) > topK {
		candidates = candidates[:topK]
	}

	results := make([]SearchResult, len(candidates))
	for i, c := range candidates {
		results[i] = c.result
	}
	return results, nil
}

// Close is a no-op for the in-memory backend.
func (m *MemoryBackend) Close() error {
	return nil
}

// cosineSimilarity computes the cosine similarity between two vectors.
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}
