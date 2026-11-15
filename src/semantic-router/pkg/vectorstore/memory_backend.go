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
	"slices"
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
	bm25Index  *BM25Index               // rebuilt on chunk mutations
	ngramIndex *NgramIndex              // rebuilt on chunk mutations
	// inserted orders the chunks by when they were first inserted, for a
	// listing newest first.
	inserted map[string]uint64
	next     uint64
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
		inserted:  make(map[string]uint64),
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
		if col.dimension > 0 && len(chunk.Embedding) != col.dimension {
			return fmt.Errorf("embedding dimension mismatch for collection %s: expected %d, got %d", vectorStoreID, col.dimension, len(chunk.Embedding))
		}
	}

	if m.maxEntries > 0 {
		newIDs := make(map[string]struct{}, len(chunks))
		for _, chunk := range chunks {
			if _, exists := col.chunks[chunk.ID]; exists {
				continue
			}
			newIDs[chunk.ID] = struct{}{}
		}
		if len(col.chunks)+len(newIDs) > m.maxEntries {
			return fmt.Errorf("collection %s has reached maximum entries (%d)", vectorStoreID, m.maxEntries)
		}
	}

	for _, chunk := range chunks {
		if _, exists := col.chunks[chunk.ID]; !exists {
			col.next++
			col.inserted[chunk.ID] = col.next
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
			delete(col.inserted, id)
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
	fileIDs := filterFileIDs(filter)

	type scored struct {
		result SearchResult
		score  float64
	}

	var candidates []scored
	for _, chunk := range col.chunks {
		if len(fileIDs) > 0 && !slices.Contains(fileIDs, chunk.FileID) {
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

	fileIDs := filterFileIDs(filter)

	// Build the set of candidate chunk IDs (respecting file_id filter).
	candidates := make(map[string]EmbeddedChunk)
	for id, chunk := range col.chunks {
		if len(fileIDs) > 0 && !slices.Contains(fileIDs, chunk.FileID) {
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
		if len(fileIDs) > 0 {
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
		if len(fileIDs) > 0 {
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
	resultCapacity := topK
	if resultCapacity < 0 {
		resultCapacity = 0
	}
	results := make([]SearchResult, 0, resultCapacity)
	for _, fc := range fused {
		if fc.FinalScore < float64(threshold) {
			continue
		}
		chunk, ok := col.chunks[fc.ChunkID]
		if !ok {
			continue
		}
		vs := fc.VectorScore
		bs := fc.BM25Score
		ns := fc.NgramScore
		results = append(results, SearchResult{
			FileID:      chunk.FileID,
			Filename:    chunk.Filename,
			Content:     chunk.Content,
			Score:       fc.FinalScore,
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

// LexicalSearch scores the collection's chunks against the query by BM25
// alone, without a query embedding: the most similar first, at most topK
// (0 for all), each scoring at least threshold, and none without a query
// term. Implements LexicalSearcher.
func (m *MemoryBackend) LexicalSearch(
	_ context.Context, vectorStoreID, query string,
	topK int, threshold float32, filter map[string]interface{},
) ([]SearchResult, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	col, exists := m.collections[vectorStoreID]
	if !exists {
		return nil, fmt.Errorf("collection not found: %s", vectorStoreID)
	}
	if col.bm25Index == nil {
		return nil, nil
	}
	fileIDs := filterFileIDs(filter)
	scores := col.bm25Index.Score(query, DefaultBM25K1, DefaultBM25B)
	ids := make([]string, 0, len(scores))
	for id, score := range scores {
		if (len(fileIDs) == 0 || slices.Contains(fileIDs, col.chunks[id].FileID)) && score >= float64(threshold) {
			ids = append(ids, id)
		}
	}
	// Ties go to the most recently inserted chunk.
	sort.Slice(ids, func(i, j int) bool {
		if scores[ids[i]] != scores[ids[j]] {
			return scores[ids[i]] > scores[ids[j]]
		}
		return col.inserted[ids[i]] > col.inserted[ids[j]]
	})
	if topK > 0 && len(ids) > topK {
		ids = ids[:topK]
	}
	results := make([]SearchResult, len(ids))
	for index, id := range ids {
		chunk, score := col.chunks[id], scores[id]
		results[index] = SearchResult{FileID: chunk.FileID, Filename: chunk.Filename, Content: chunk.Content, Score: score, ChunkIndex: chunk.ChunkIndex, BM25Score: &score}
	}
	return results, nil
}

// ListChunks lists the collection's chunks, the most recently inserted first,
// at most limit of them, and how many it holds. Implements ChunkLister.
func (m *MemoryBackend) ListChunks(
	_ context.Context, vectorStoreID string, limit int, filter map[string]interface{},
) ([]SearchResult, int, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	col, exists := m.collections[vectorStoreID]
	if !exists {
		return nil, 0, fmt.Errorf("collection not found: %s", vectorStoreID)
	}
	fileIDs := filterFileIDs(filter)
	ids := make([]string, 0, len(col.chunks))
	for id, chunk := range col.chunks {
		if len(fileIDs) == 0 || slices.Contains(fileIDs, chunk.FileID) {
			ids = append(ids, id)
		}
	}
	sort.Slice(ids, func(i, j int) bool { return col.inserted[ids[i]] > col.inserted[ids[j]] })
	held := len(ids)
	results := make([]SearchResult, 0, min(limit, held))
	for _, id := range ids[:min(limit, held)] {
		chunk := col.chunks[id]
		results = append(results, SearchResult{FileID: chunk.FileID, Filename: chunk.Filename, Content: chunk.Content, ChunkIndex: chunk.ChunkIndex})
	}
	return results, held, nil
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
