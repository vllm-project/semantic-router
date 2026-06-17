package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

func (m *MilvusStore) Retrieve(ctx context.Context, opts RetrieveOptions) ([]*RetrieveResult, error) {
	startTime := time.Now()
	backend := "milvus"
	operation := "retrieve"
	status := "success"
	resultCount := 0

	defer func() {
		duration := time.Since(startTime).Seconds()
		RecordMemoryRetrieval(backend, operation, status, opts.UserID, duration, resultCount)
	}()

	limit, threshold, err := m.normalizeRetrieveOpts(opts)
	if err != nil {
		status = "error"
		return nil, err
	}

	logging.Debugf("MilvusStore.Retrieve: query='%s', user_id='%s', limit=%d, threshold=%.4f, hybrid=%v (mode=%s)",
		opts.Query, opts.UserID, limit, threshold, opts.HybridSearch, opts.HybridMode)

	embedding, err := GenerateEmbedding(opts.Query, m.embeddingConfig)
	if err != nil {
		status = "error"
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	searchResult, err := m.searchMemoryVectors(ctx, embedding, retrieveFilterExpr(opts.UserID, opts.Types), retrieveSearchTopK(limit, opts.HybridSearch))
	if err != nil {
		status = "error"
		return nil, err
	}

	if len(searchResult) == 0 || searchResult[0].ResultCount == 0 {
		logging.Debugf("MilvusStore.Retrieve: no results found")
		status = "miss"
		return []*RetrieveResult{}, nil
	}

	results := m.finalizeRetrieveResults(searchResult[0], opts, limit, threshold)
	resultCount = len(results)
	if resultCount > 0 {
		status = "hit"
	} else {
		status = "miss"
	}
	return results, nil
}

func (m *MilvusStore) normalizeRetrieveOpts(opts RetrieveOptions) (limit int, threshold float32, err error) {
	if !m.enabled {
		return 0, 0, fmt.Errorf("milvus store is not enabled")
	}
	limit = opts.Limit
	if limit <= 0 {
		limit = m.config.DefaultRetrievalLimit
	}
	threshold = opts.Threshold
	if threshold <= 0 {
		threshold = m.config.DefaultSimilarityThreshold
	}
	if opts.Query == "" {
		return 0, 0, fmt.Errorf("query is required")
	}
	if opts.UserID == "" {
		return 0, 0, fmt.Errorf("user id is required")
	}
	return limit, threshold, nil
}

func retrieveFilterExpr(userID string, types []MemoryType) string {
	filterExpr := milvusUserScopeFilter(userID)
	if tf := buildTypeFilter(types); tf != "" {
		filterExpr = fmt.Sprintf("%s && %s", filterExpr, tf)
	}
	return filterExpr
}

func retrieveSearchTopK(limit int, hybridSearch bool) int {
	searchTopK := limit * 4
	if hybridSearch {
		searchTopK = limit * 8
	}
	if searchTopK < 20 {
		searchTopK = 20
	}
	return searchTopK
}

func (m *MilvusStore) searchMemoryVectors(ctx context.Context, embedding []float32, filterExpr string, searchTopK int) ([]client.SearchResult, error) {
	logging.Debugf("MilvusStore.Retrieve: filter expression: %s", filterExpr)

	searchParam, err := entity.NewIndexHNSWSearchParam(64)
	if err != nil {
		return nil, fmt.Errorf("failed to create search parameters: %w", err)
	}

	var searchResult []client.SearchResult
	err = m.retryWithBackoff(ctx, func() error {
		var retryErr error
		searchResult, retryErr = m.client.Search(
			ctx,
			m.collectionName,
			[]string{},
			filterExpr,
			[]string{"id", "content", "memory_type", "metadata"},
			[]entity.Vector{entity.FloatVector(embedding)},
			"embedding",
			entity.COSINE,
			searchTopK,
			searchParam,
		)
		return retryErr
	})
	if err != nil {
		return nil, fmt.Errorf("milvus search failed after retries: %w", err)
	}
	return searchResult, nil
}

func (m *MilvusStore) finalizeRetrieveResults(sr client.SearchResult, opts RetrieveOptions, limit int, threshold float32) []*RetrieveResult {
	candidates := m.parseCandidates(sr, opts.UserID)
	if opts.HybridSearch && len(candidates) > 1 {
		candidates = m.hybridRerank(candidates, opts)
	}
	if opts.AdaptiveThreshold && len(candidates) > 1 {
		threshold = adaptiveThresholdElbow(candidates, threshold)
	}

	results := applyRetrieveThreshold(candidates, limit, threshold)
	if len(results) > 0 {
		ids := make([]string, len(results))
		for i, r := range results {
			ids[i] = r.Memory.ID
		}
		go m.recordRetrievalBatch(ids)
	}
	return results
}

func applyRetrieveThreshold(candidates []*RetrieveResult, limit int, threshold float32) []*RetrieveResult {
	results := make([]*RetrieveResult, 0, limit)
	for _, c := range candidates {
		if c.Score < threshold {
			continue
		}
		results = append(results, c)
		if len(results) >= limit {
			break
		}
	}
	return results
}

func adaptiveThresholdElbow(candidates []*RetrieveResult, floor float32) float32 {
	if len(candidates) < 2 {
		return floor
	}

	maxGap := float32(0)
	elbowIdx := 0
	for i := 1; i < len(candidates); i++ {
		gap := candidates[i-1].Score - candidates[i].Score
		if gap > maxGap {
			maxGap = gap
			elbowIdx = i
		}
	}

	if maxGap < 0.05 {
		return floor
	}

	adaptive := candidates[elbowIdx-1].Score - maxGap/2
	if adaptive > floor {
		return adaptive
	}
	return floor
}

func (m *MilvusStore) parseCandidates(sr client.SearchResult, defaultUserID string) []*RetrieveResult {
	fieldIdx := indexSearchResultFields(sr.Fields)
	results := make([]*RetrieveResult, 0, len(sr.Scores))
	for i, score := range sr.Scores {
		if candidate := retrieveCandidateFromRow(sr.Fields, fieldIdx, i, score, defaultUserID); candidate != nil {
			results = append(results, candidate)
		}
	}
	return results
}

type searchFieldIndex struct {
	id, content, memoryType, metadata int
}

func indexSearchResultFields(fields []entity.Column) searchFieldIndex {
	idx := searchFieldIndex{id: -1, content: -1, memoryType: -1, metadata: -1}
	for i, field := range fields {
		switch field.Name() {
		case "id":
			idx.id = i
		case "content":
			idx.content = i
		case "memory_type":
			idx.memoryType = i
		case "metadata":
			idx.metadata = i
		}
	}
	return idx
}

func retrieveCandidateFromRow(fields []entity.Column, idx searchFieldIndex, row int, score float32, defaultUserID string) *RetrieveResult {
	id := varcharFieldValue(fields, idx.id, row)
	content := varcharFieldValue(fields, idx.content, row)
	memType := varcharFieldValue(fields, idx.memoryType, row)
	if id == "" || content == "" {
		return nil
	}

	metadata := parseRetrieveMetadata(fields, idx.metadata, row)
	mem := &Memory{ID: id, Content: content, Type: MemoryType(memType)}
	populateMemoryFromRetrieveMetadata(mem, metadata, defaultUserID)
	return &RetrieveResult{Memory: mem, Score: score}
}

func varcharFieldValue(fields []entity.Column, fieldIdx, row int) string {
	if fieldIdx < 0 || fieldIdx >= len(fields) {
		return ""
	}
	col, ok := fields[fieldIdx].(*entity.ColumnVarChar)
	if !ok || col.Len() <= row {
		return ""
	}
	val, err := col.ValueByIdx(row)
	if err != nil {
		return ""
	}
	return val
}

func parseRetrieveMetadata(fields []entity.Column, fieldIdx, row int) map[string]interface{} {
	metadata := make(map[string]interface{})
	if fieldIdx < 0 || fieldIdx >= len(fields) {
		return metadata
	}
	col, ok := fields[fieldIdx].(*entity.ColumnVarChar)
	if !ok || col.Len() <= row {
		return metadata
	}
	metadataVal, err := col.ValueByIdx(row)
	if err != nil || metadataVal == "" {
		return metadata
	}
	if err := json.Unmarshal([]byte(metadataVal), &metadata); err != nil {
		metadata["raw"] = metadataVal
	} else {
		metadata["_raw_source"] = metadataVal
	}
	return metadata
}

func populateMemoryFromRetrieveMetadata(mem *Memory, metadata map[string]interface{}, defaultUserID string) {
	if userID, ok := metadata["user_id"].(string); ok {
		mem.UserID = userID
	} else if defaultUserID != "" {
		mem.UserID = defaultUserID
	}
	if projectID, ok := metadata["project_id"].(string); ok {
		mem.ProjectID = projectID
	}
	if source, ok := metadata["source"].(string); ok {
		mem.Source = source
	}
	if importance, ok := metadata["importance"].(float64); ok {
		mem.Importance = float32(importance)
	} else if importance, ok := metadata["importance"].(float32); ok {
		mem.Importance = importance
	}
	if accessCount, ok := metadata["access_count"].(float64); ok {
		mem.AccessCount = int(accessCount)
	}
	if lastAccessed, ok := metadata["last_accessed"].(float64); ok {
		mem.LastAccessed = time.Unix(int64(lastAccessed), 0)
	}
}

func (m *MilvusStore) hybridRerank(candidates []*RetrieveResult, opts RetrieveOptions) []*RetrieveResult {
	pseudoChunks := make(map[string]vectorstore.EmbeddedChunk, len(candidates))
	vectorScores := make(map[string]float64, len(candidates))
	keyToCandidate := make(map[string]*RetrieveResult, len(candidates))

	for i, c := range candidates {
		key := fmt.Sprintf("_mem_%d", i)
		pseudoChunks[key] = vectorstore.EmbeddedChunk{ID: key, Content: c.Memory.Content}
		vectorScores[key] = float64(c.Score)
		keyToCandidate[key] = c
	}

	hybridCfg := &vectorstore.HybridSearchConfig{Mode: opts.HybridMode}
	bm25K1 := hybridCfg.BM25K1
	if bm25K1 == 0 {
		bm25K1 = 1.2
	}
	bm25B := hybridCfg.BM25B
	if bm25B == 0 {
		bm25B = 0.75
	}
	ngramSize := hybridCfg.NgramSize
	if ngramSize <= 0 {
		ngramSize = 3
	}

	bm25Idx := vectorstore.NewBM25Index(pseudoChunks)
	bm25Scores := bm25Idx.Score(opts.Query, bm25K1, bm25B)
	ngramIdx := vectorstore.NewNgramIndex(pseudoChunks, ngramSize)
	ngramScores := ngramIdx.Score(opts.Query)
	fused := vectorstore.FuseScores(vectorScores, bm25Scores, ngramScores, hybridCfg)

	reranked := make([]*RetrieveResult, 0, len(fused))
	for _, fc := range fused {
		c, ok := keyToCandidate[fc.ChunkID]
		if !ok {
			continue
		}
		c.Score = float32(fc.FinalScore)
		reranked = append(reranked, c)
	}
	return reranked
}
