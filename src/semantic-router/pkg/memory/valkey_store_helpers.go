package memory

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

// ---------------------------------------------------------------------------
// Background access tracking
// ---------------------------------------------------------------------------

// recordRetrievalBatch updates LastAccessed and AccessCount for each retrieved memory in the background.
// Uses targeted HINCRBY + HSET instead of full read-modify-write for efficiency.
// The user-facing behavior matches the Milvus backend (access_count incremented, timestamps updated).
func (v *ValkeyStore) recordRetrievalBatch(ids []string) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	for _, id := range ids {
		if err := v.recordRetrieval(ctx, id); err != nil {
			logging.Warnf("ValkeyStore.recordRetrievalBatch: id=%s: %v", id, err)
		}
	}
}

// recordRetrieval updates LastAccessed and AccessCount for a single memory (reinforcement: S += 1, t = 0).
//
// The authoritative access_count and updated_at live as top-level HASH fields and are updated
// atomically via HINCRBY / HSET. Do not rewrite duplicated values inside the metadata JSON blob
// here: a separate read-modify-write of metadata can race with concurrent retrievals and move
// access_count / last_accessed backwards. Callers should use the top-level HASH fields as the
// source of truth for these mutable values.
func (v *ValkeyStore) recordRetrieval(ctx context.Context, id string) error {
	key := v.hashKey(id)
	now := time.Now()
	nowUnix := strconv.FormatInt(now.UnixMilli(), 10)

	// Increment access_count atomically.
	_, err := v.client.CustomCommand(ctx, []string{"HINCRBY", key, "access_count", "1"})
	if err != nil {
		return fmt.Errorf("HINCRBY access_count failed: %w", err)
	}

	// Update timestamps.
	_, err = v.client.HSet(ctx, key, map[string]string{
		"updated_at": nowUnix,
	})
	if err != nil {
		return fmt.Errorf("HSET timestamps failed: %w", err)
	}

	return nil
}

// ---------------------------------------------------------------------------
// Hybrid re-ranking
// ---------------------------------------------------------------------------

// hybridRerank applies BM25 + n-gram scoring on top of vector results.
func (v *ValkeyStore) hybridRerank(candidates []*RetrieveResult, opts RetrieveOptions) []*RetrieveResult {
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

// ---------------------------------------------------------------------------
// Result parsing
// ---------------------------------------------------------------------------

// valkeyIterateSearchDocs extracts field maps from an FT.SEARCH result array.
// It skips the total-count header and yields each document's field map.
func valkeyIterateSearchDocs(result any) []map[string]interface{} {
	arr, ok := result.([]interface{})
	if !ok || len(arr) < 1 {
		return nil
	}

	totalCount := valkeyToInt64(arr[0])
	if totalCount == 0 {
		return nil
	}

	var docs []map[string]interface{}
	for i := 1; i < len(arr); i++ {
		docMap, ok := arr[i].(map[string]interface{})
		if !ok {
			continue
		}
		for _, docValue := range docMap {
			fieldsMap, mapOk := docValue.(map[string]interface{})
			if !mapOk {
				continue
			}
			docs = append(docs, fieldsMap)
		}
	}
	return docs
}

func (v *ValkeyStore) parseSearchCandidates(result any, defaultUserID string) []*RetrieveResult {
	docs := valkeyIterateSearchDocs(result)
	if len(docs) == 0 {
		return nil
	}

	var candidates []*RetrieveResult

	for _, fieldsMap := range docs {
		id := fmt.Sprint(fieldsMap["id"])
		content := fmt.Sprint(fieldsMap["content"])
		memType := fmt.Sprint(fieldsMap["memory_type"])

		if id == "" || id == "<nil>" || content == "" || content == "<nil>" {
			continue
		}

		score := valkeyParseScoreFromMap(fieldsMap, "vector_distance", v.metricType)

		mem := &Memory{
			ID:      id,
			Content: content,
			Type:    MemoryType(memType),
		}

		if metadataStr, ok := fieldsMap["metadata"].(string); ok {
			valkeyParseMetadata(mem, metadataStr)
		}

		if mem.UserID == "" {
			mem.UserID = defaultUserID
		}

		candidates = append(candidates, &RetrieveResult{Memory: mem, Score: float32(score)})
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Score > candidates[j].Score
	})

	return candidates
}

// parseListSearchResults parses FT.SEARCH results for List operations (no vector_distance).
func (v *ValkeyStore) parseListSearchResults(result any) []*Memory {
	docs := valkeyIterateSearchDocs(result)
	if len(docs) == 0 {
		return nil
	}

	var memories []*Memory
	for _, fieldsMap := range docs {
		mem := valkeyFieldsMapToMemory(fieldsMap)
		if mem.ID == "" {
			continue
		}
		memories = append(memories, mem)
	}

	// valkey-glide returns matched docs in a single map; iterating that map in
	// Go does not preserve FT.SEARCH SORTBY order (same issue as vectorstore
	// parseSearchResults). Re-sort client-side by created_at descending.
	sort.SliceStable(memories, func(i, j int) bool {
		ti, tj := memories[i].CreatedAt, memories[j].CreatedAt
		if ti.Equal(tj) {
			return memories[i].ID > memories[j].ID
		}
		return ti.After(tj)
	})

	return memories
}

// valkeyMatchesProjectFilter checks whether a document's metadata contains the expected project_id.
func valkeyMatchesProjectFilter(fieldsMap map[string]interface{}, projectIDFilter string) bool {
	metadataStr, ok := fieldsMap["metadata"].(string)
	if !ok || metadataStr == "" {
		return false
	}
	var metadata map[string]interface{}
	if err := json.Unmarshal([]byte(metadataStr), &metadata); err != nil {
		return false
	}
	projectID, ok := metadata["project_id"].(string)
	return ok && projectID == projectIDFilter
}

// extractIDsFromSearchResult extracts memory IDs from FT.SEARCH results, optionally filtering by project_id.
func (v *ValkeyStore) extractIDsFromSearchResult(result any, projectIDFilter string) []string {
	docs := valkeyIterateSearchDocs(result)
	if len(docs) == 0 {
		return nil
	}

	var ids []string
	for _, fieldsMap := range docs {
		id := fmt.Sprint(fieldsMap["id"])
		if id == "" || id == "<nil>" {
			continue
		}
		if projectIDFilter != "" && !valkeyMatchesProjectFilter(fieldsMap, projectIDFilter) {
			continue
		}
		ids = append(ids, id)
	}

	return ids
}

// extractHashKeysFromSearchResult extracts the full hash keys (document keys)
// from an FT.SEARCH result. These are the actual Valkey keys that can be passed
// to DEL for batch deletion. Follows the same pattern as
// extractKeysFromSearchResult in pkg/vectorstore/valkey_backend.go.
func (v *ValkeyStore) extractHashKeysFromSearchResult(result any) []string {
	arr, ok := result.([]interface{})
	if !ok || len(arr) < 2 {
		return nil
	}

	var keys []string
	for i := 1; i < len(arr); i++ {
		switch val := arr[i].(type) {
		case string:
			keys = append(keys, val)
		case map[string]interface{}:
			for docKey := range val {
				keys = append(keys, docKey)
			}
		}
	}
	return keys
}

// extractTotalCount extracts the total match count from the FT.SEARCH result header.
// The first element of the result array is always the total count of matching documents,
// regardless of the LIMIT clause.
func (v *ValkeyStore) extractTotalCount(result any) int {
	arr, ok := result.([]interface{})
	if !ok || len(arr) < 1 {
		return 0
	}
	return int(valkeyToInt64(arr[0]))
}

// ---------------------------------------------------------------------------
// Retry logic
// ---------------------------------------------------------------------------

// retryWithBackoff retries an operation with exponential backoff for transient errors.
func (v *ValkeyStore) retryWithBackoff(ctx context.Context, operation func() error) error {
	var lastErr error

	for attempt := 0; attempt < v.maxRetries; attempt++ {
		lastErr = operation()

		if lastErr == nil || !isTransientError(lastErr) {
			return lastErr
		}

		if attempt == v.maxRetries-1 {
			logging.Warnf("ValkeyStore: operation failed after %d retries: %v", v.maxRetries, lastErr)
			return lastErr
		}

		exponent := attempt
		if exponent < 0 {
			exponent = 0
		} else if exponent > 30 {
			exponent = 30
		}
		delay := v.retryBaseDelay * time.Duration(1<<exponent)

		logging.Debugf("ValkeyStore: transient error on attempt %d/%d, retrying in %v: %v",
			attempt+1, v.maxRetries, delay, lastErr)

		select {
		case <-ctx.Done():
			return fmt.Errorf("context cancelled during retry: %w", ctx.Err())
		case <-time.After(delay):
		}
	}

	return lastErr
}

// ---------------------------------------------------------------------------
// Helper functions (prefixed with valkey to avoid collisions)
// ---------------------------------------------------------------------------

// valkeyFloat32ToBytes converts a float32 slice to a little-endian byte slice
// suitable for Valkey vector storage.
func valkeyFloat32ToBytes(floats []float32) []byte {
	buf := make([]byte, len(floats)*4)
	for i, f := range floats {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// valkeyBytesToFloat32 converts a little-endian byte slice back to float32 slice.
func valkeyBytesToFloat32(data []byte) []float32 {
	if len(data)%4 != 0 {
		return nil
	}
	floats := make([]float32, len(data)/4)
	for i := range floats {
		bits := binary.LittleEndian.Uint32(data[i*4:])
		floats[i] = math.Float32frombits(bits)
	}
	return floats
}

// valkeyEscapeTagValue escapes punctuation and whitespace in a string so it can
// be safely used inside a Valkey TAG query expression (@field:{value}).
// TAG queries treat punctuation characters as token separators; backslash-
// escaping them preserves the literal value.
// Follows the same comprehensive escaping as the Valkey cache backend
// (pkg/cache/valkey_cache_helpers.go escapeTagValue).
// Reference: https://forum.redis.com/t/tag-fields-and-escaping/96
func valkeyEscapeTagValue(val string) string {
	const specialChars = " \t,.<>{}[]\"':;!@#$%^&*()-+=~|/\\"

	var b strings.Builder
	b.Grow(len(val) + 8)
	for _, c := range val {
		if strings.ContainsRune(specialChars, c) {
			b.WriteByte('\\')
		}
		b.WriteRune(c)
	}
	return b.String()
}

// valkeyDistanceToSimilarity converts a vector distance to a similarity score based on the metric type.
// Follows the same conversion as the Valkey cache backend (pkg/cache/valkey_cache_helpers.go)
// and the Valkey vector store backend (pkg/vectorstore/valkey_backend.go).
//
// Valkey Search COSINE distance is in [0, 2] where 0 = identical vectors.
// The formula 1 - d/2 maps [0, 2] → [1, 0], producing the same similarity
// range [0, 1] that Milvus returns directly, so threshold values are portable
// across backends.
func valkeyDistanceToSimilarity(metricType string, distance float64) float64 {
	switch strings.ToUpper(metricType) {
	case "COSINE":
		return 1.0 - distance/2.0
	case "L2":
		return 1.0 / (1.0 + distance)
	case "IP":
		return distance
	default:
		logging.Warnf("ValkeyStore: unknown metric type %q in distance conversion, using 1-d fallback", metricType)
		return 1.0 - distance
	}
}

// valkeyParseScoreFromMap extracts a distance value from the fields map and converts it to similarity.
func valkeyParseScoreFromMap(fields map[string]interface{}, key string, metricType string) float64 {
	raw, exists := fields[key]
	if !exists {
		return 0
	}
	distance, err := strconv.ParseFloat(fmt.Sprint(raw), 64)
	if err != nil {
		return 0
	}
	return valkeyDistanceToSimilarity(metricType, distance)
}

// valkeyBuildHashFields builds the HSET field map for storing a memory in Valkey.
func valkeyBuildHashFields(memory *Memory, embedding []float32) (map[string]string, error) {
	// access_count is stored as a top-level HASH field only (updated atomically via HINCRBY).
	// It is intentionally excluded from the metadata JSON blob to prevent concurrent
	// recordRetrieval goroutines from overwriting each other's incremented counts.
	metadata := map[string]interface{}{
		"user_id":       memory.UserID,
		"project_id":    memory.ProjectID,
		"source":        memory.Source,
		"importance":    memory.Importance,
		"last_accessed": memory.LastAccessed.Unix(),
	}
	metadataJSON, err := json.Marshal(metadata)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal metadata: %w", err)
	}

	projectID := memory.ProjectID
	if projectID == "" {
		projectID = "default"
	}
	source := memory.Source
	if source == "" {
		source = "extraction"
	}

	return map[string]string{
		"id":           memory.ID,
		"user_id":      memory.UserID,
		"project_id":   projectID,
		"memory_type":  string(memory.Type),
		"content":      memory.Content,
		"source":       source,
		"metadata":     string(metadataJSON),
		"embedding":    string(valkeyFloat32ToBytes(embedding)),
		"created_at":   strconv.FormatInt(memory.CreatedAt.UnixMilli(), 10),
		"updated_at":   strconv.FormatInt(memory.UpdatedAt.UnixMilli(), 10),
		"access_count": strconv.Itoa(memory.AccessCount),
		"importance":   strconv.FormatFloat(float64(memory.Importance), 'f', -1, 32),
	}, nil
}

// valkeyValidateRetrieveOpts checks required fields on RetrieveOptions.
func valkeyValidateRetrieveOpts(opts RetrieveOptions) error {
	if opts.Query == "" {
		return fmt.Errorf("query is required")
	}
	if opts.UserID == "" {
		return fmt.Errorf("user id is required")
	}
	return nil
}

// valkeyValidateMemory checks required fields on a Memory before storing.
func valkeyValidateMemory(memory *Memory) error {
	if memory.ID == "" {
		return fmt.Errorf("memory ID is required")
	}
	if memory.Content == "" {
		return fmt.Errorf("memory content is required")
	}
	if memory.UserID == "" {
		return fmt.Errorf("user ID is required")
	}
	return nil
}

// valkeyToInt64 converts an interface{} to int64, handling int64, float64, and string.
func valkeyToInt64(v interface{}) int64 {
	switch val := v.(type) {
	case int64:
		return val
	case float64:
		return int64(val)
	case string:
		n, _ := strconv.ParseInt(val, 10, 64)
		return n
	default:
		return 0
	}
}

// valkeyApplyMetadata populates Memory fields from a parsed metadata JSON map.
func valkeyApplyMetadata(mem *Memory, metadata map[string]interface{}) {
	if userID, ok := metadata["user_id"].(string); ok && mem.UserID == "" {
		mem.UserID = userID
	}
	if projectID, ok := metadata["project_id"].(string); ok {
		mem.ProjectID = projectID
	}
	if source, ok := metadata["source"].(string); ok {
		mem.Source = source
	}
	if importance, ok := metadata["importance"].(float64); ok {
		mem.Importance = float32(importance)
	}
	// access_count is NOT in metadata JSON — it lives in the top-level HASH field only.
	// Read it via valkeyFieldsToMemory from the "access_count" HASH field instead.
	if lastAccessed, ok := metadata["last_accessed"].(float64); ok {
		mem.LastAccessed = time.Unix(int64(lastAccessed), 0)
	}
}

// valkeyParseMetadata unmarshals a metadata JSON string and applies it to the Memory.
func valkeyParseMetadata(mem *Memory, metadataStr string) {
	if metadataStr == "" {
		return
	}
	var metadata map[string]interface{}
	if err := json.Unmarshal([]byte(metadataStr), &metadata); err == nil {
		valkeyApplyMetadata(mem, metadata)
	}
}

// valkeyFieldsToMemory converts HGETALL fields (map[string]string) to a Memory struct.
func valkeyFieldsToMemory(fields map[string]string) *Memory {
	mem := &Memory{
		ID:      fields["id"],
		Content: fields["content"],
		UserID:  fields["user_id"],
		Type:    MemoryType(fields["memory_type"]),
	}

	valkeyParseMetadata(mem, fields["metadata"])

	// access_count is authoritative in the top-level HASH field (updated atomically
	// via HINCRBY). Override whatever valkeyParseMetadata may have set.
	if acStr := fields["access_count"]; acStr != "" {
		if ac, err := strconv.Atoi(acStr); err == nil {
			mem.AccessCount = ac
		}
	}

	if createdAtStr := fields["created_at"]; createdAtStr != "" {
		if ts, err := strconv.ParseInt(createdAtStr, 10, 64); err == nil {
			mem.CreatedAt = time.UnixMilli(ts)
		}
	}
	if updatedAtStr := fields["updated_at"]; updatedAtStr != "" {
		if ts, err := strconv.ParseInt(updatedAtStr, 10, 64); err == nil {
			mem.UpdatedAt = time.UnixMilli(ts)
		}
	}
	if embStr := fields["embedding"]; embStr != "" {
		mem.Embedding = valkeyBytesToFloat32([]byte(embStr))
	}

	return mem
}

// valkeyFieldsMapToMemory converts FT.SEARCH result fields (map[string]interface{}) to a Memory struct.
func valkeyFieldsMapToMemory(fields map[string]interface{}) *Memory {
	mem := &Memory{}

	if id, ok := fields["id"].(string); ok {
		mem.ID = id
	}
	if content, ok := fields["content"].(string); ok {
		mem.Content = content
	}
	if userID, ok := fields["user_id"].(string); ok {
		mem.UserID = userID
	}
	if memType, ok := fields["memory_type"].(string); ok {
		mem.Type = MemoryType(memType)
	}

	if metadataStr, ok := fields["metadata"].(string); ok {
		valkeyParseMetadata(mem, metadataStr)
	}

	if createdAtStr, ok := fields["created_at"].(string); ok && createdAtStr != "" {
		if ts, err := strconv.ParseInt(createdAtStr, 10, 64); err == nil {
			mem.CreatedAt = time.UnixMilli(ts)
		}
	}
	if updatedAtStr, ok := fields["updated_at"].(string); ok && updatedAtStr != "" {
		if ts, err := strconv.ParseInt(updatedAtStr, 10, 64); err == nil {
			mem.UpdatedAt = time.UnixMilli(ts)
		}
	}

	return mem
}
