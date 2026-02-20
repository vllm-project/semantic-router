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
	dimension  int
	chunks     map[string]EmbeddedChunk // chunk ID -> chunk
	bm25Index  *BM25Index              // rebuilt on chunk mutations
	ngramIndex *NgramIndex             // rebuilt on chunk mutations
}

const defaultNgramSize = 3

// rebuildTextIndexes rebuilds the BM25 and n-gram indexes from current chunks.
func (col *memoryCollection) rebuildTextIndexes() {
	col.bm25Index = NewBM25Index(col.chunks)
	col.ngramIndex = NewNgramIndex(col.chunks, defaultNgramSize)
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

	col.rebuildTextIndexes()
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

	deleted := false
	for id, chunk := range col.chunks {
		if chunk.FileID == fileID {
			delete(col.chunks, id)
			deleted = true
		}
	}

	if deleted {
		col.rebuildTextIndexes()
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

// HybridSearch performs hybrid search combining vector similarity, BM25, and
// n-gram scoring. Implements the HybridSearcher interface.
func (m *MemoryBackend) HybridSearch(
	_ context.Context, vectorStoreID string,
	query string, queryEmbedding []float32,
	topK int, threshold float32,
	filter map[string]interface{},
	config *HybridSearchConfig,
) ([]SearchResult, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	col, exists := m.collections[vectorStoreID]
	if !exists {
		return nil, fmt.Errorf("collection not found: %s", vectorStoreID)
	}

	if config == nil {
		config = &HybridSearchConfig{}
	}
	config.applyDefaults()

	var filterFileID string
	if filter != nil {
		if fid, ok := filter["file_id"].(string); ok {
			filterFileID = fid
		}
	}

	// Build the set of candidate chunk IDs (respecting file_id filter).
	candidates := make(map[string]EmbeddedChunk)
	for id, chunk := range col.chunks {
		if filterFileID != "" && chunk.FileID != filterFileID {
			continue
		}
		candidates[id] = chunk
	}

	// 1. Vector similarity scores.
	vectorScores := make(map[string]float64, len(candidates))
	for id, chunk := range candidates {
		sim := cosineSimilarity(queryEmbedding, chunk.Embedding)
		if sim > 0 {
			vectorScores[id] = sim
		}
	}

	// 2. BM25 scores (use the pre-built index, score with request-time k1/b).
	var bm25Scores map[string]float64
	if col.bm25Index != nil {
		allBM25 := col.bm25Index.Score(query, config.BM25K1, config.BM25B)
		if filterFileID != "" {
			bm25Scores = make(map[string]float64)
			for id, s := range allBM25 {
				if _, ok := candidates[id]; ok {
					bm25Scores[id] = s
				}
			}
		} else {
			bm25Scores = allBM25
		}
	}

	// 3. N-gram scores. Rebuild if requested size differs from pre-built index.
	var ngramScores map[string]float64
	ngramIdx := col.ngramIndex
	if ngramIdx != nil && ngramIdx.NgramN() != config.NgramSize {
		ngramIdx = NewNgramIndex(col.chunks, config.NgramSize)
	}
	if ngramIdx != nil {
		allNgram := ngramIdx.Score(query)
		if filterFileID != "" {
			ngramScores = make(map[string]float64)
			for id, s := range allNgram {
				if _, ok := candidates[id]; ok {
					ngramScores[id] = s
				}
			}
		} else {
			ngramScores = allNgram
		}
	}

	// 4. Fuse scores.
	fused := FuseScores(vectorScores, bm25Scores, ngramScores, config)

	// 5. Apply threshold and topK, build results.
	results := make([]SearchResult, 0, topK)
	for _, fc := range fused {
		if fc.finalScore < float64(threshold) {
			continue
		}
		chunk, ok := col.chunks[fc.chunkID]
		if !ok {
			continue
		}
		vs := fc.vectorScore
		bs := fc.bm25Score
		ns := fc.ngramScore
		results = append(results, SearchResult{
			FileID:      chunk.FileID,
			Filename:    chunk.Filename,
			Content:     chunk.Content,
			Score:       fc.finalScore,
			ChunkIndex:  chunk.ChunkIndex,
			VectorScore: &vs,
			BM25Score:   &bs,
			NgramScore:  &ns,
		})
		if topK > 0 && len(results) >= topK {
			break
		}
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
