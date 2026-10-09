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

	glideoptions "github.com/valkey-io/valkey-glide/go/v2/options"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	valkeyutil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/valkey"
)

// ---------------------------------------------------------------------------
// Background access tracking
// ---------------------------------------------------------------------------

// recordRetrievalBatch updates LastAccessed and AccessCount for each retrieved memory in the background.
// Uses a targeted Valkey script instead of full read-modify-write for efficiency.
// Reads update access metadata only; they must not change UpdatedAt because that
// timestamp is the write-version used by atomic consolidation.
func (v *ValkeyStore) recordRetrievalBatch(ids []string) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	for _, id := range ids {
		if err := v.recordRetrieval(ctx, id); err != nil {
			logging.Warnf("ValkeyStore.recordRetrievalBatch: id=%s: %v", id, err)
		}
	}
}

// recordRetrieval updates LastAccessed and AccessCount for a live memory
// (reinforcement: S += 1, t = 0). The ID check and metadata writes execute in
// one script so queued tracking cannot recreate a hash that consolidation or
// Forget has deleted. UpdatedAt changes only on Store and Update, so it
// remains a write-version for atomic consolidation.
func (v *ValkeyStore) recordRetrieval(ctx context.Context, id string) error {
	key := v.hashKey(id)
	nowUnixMilli := strconv.FormatInt(time.Now().UnixMilli(), 10)
	scriptOptions := glideoptions.NewScriptOptions().
		WithKeys([]string{key}).
		WithArgs([]string{id, nowUnixMilli})

	// HINCRBY is not idempotent. Retrying after a lost reply applies the
	// increment twice, so each retrieval records access at most once.
	_, err := v.client.InvokeScriptWithOptions(ctx, *valkeyTrackRetrievalScript(), *scriptOptions)
	if err != nil {
		return fmt.Errorf("record retrieval metadata failed: %w", err)
	}

	return nil
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
		mem := valkeyFieldsMapToMemory(fieldsMap)
		if mem.ID == "" || mem.Content == "" {
			continue
		}

		score := valkeyParseScoreFromMap(fieldsMap, "vector_distance", v.metricType)

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
	return valkeyutil.DistanceToSimilarity(metricType, distance)
}

// valkeyBuildHashFields builds the HSET field map for storing a memory in Valkey.
func valkeyBuildHashFields(memory *Memory, embedding []float32) (map[string]string, error) {
	// Access metadata is stored in top-level HASH fields. access_count is
	// intentionally excluded from metadata JSON to prevent concurrent
	// recordRetrieval goroutines from overwriting incremented counts.
	metadata := map[string]interface{}{
		"user_id":    memory.UserID,
		"project_id": memory.ProjectID,
		"source":     memory.Source,
		"importance": memory.Importance,
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
		"id":            memory.ID,
		"user_id":       memory.UserID,
		"project_id":    projectID,
		"memory_type":   string(memory.Type),
		"content":       memory.Content,
		"source":        source,
		"metadata":      string(metadataJSON),
		"embedding":     string(valkeyFloat32ToBytes(embedding)),
		"created_at":    strconv.FormatInt(memory.CreatedAt.UnixMilli(), 10),
		"updated_at":    strconv.FormatInt(memory.UpdatedAt.UnixMilli(), 10),
		"last_accessed": strconv.FormatInt(memory.LastAccessed.UnixMilli(), 10),
		"access_count":  strconv.Itoa(memory.AccessCount),
		"importance":    strconv.FormatFloat(float64(memory.Importance), 'f', -1, 32),
	}, nil
}

// valkeyBuildUpdateHashFields omits access metadata so a content update cannot
// overwrite a retrieval that increments it concurrently. Store and atomic
// consolidation use valkeyBuildHashFields to initialize those fields instead.
func valkeyBuildUpdateHashFields(memory *Memory) (map[string]string, error) {
	fields, err := valkeyBuildHashFields(memory, memory.Embedding)
	if err != nil {
		return nil, err
	}
	delete(fields, "access_count")
	delete(fields, "last_accessed")
	return fields, nil
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

	// access_count and last_accessed are authoritative in top-level HASH fields,
	// so they can be updated without rewriting metadata JSON. Override the legacy
	// metadata values when a top-level value exists.
	if acStr := fields["access_count"]; acStr != "" {
		if ac, err := strconv.Atoi(acStr); err == nil {
			mem.AccessCount = ac
		}
	}
	if lastAccessedStr := fields["last_accessed"]; lastAccessedStr != "" {
		if ts, err := strconv.ParseInt(lastAccessedStr, 10, 64); err == nil {
			mem.LastAccessed = time.UnixMilli(ts)
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

	mem.ID = valkeySearchStringField(fields, "id")
	mem.Content = valkeySearchContentField(fields)
	mem.UserID = valkeySearchStringField(fields, "user_id")
	if memType := valkeySearchStringField(fields, "memory_type"); memType != "" {
		mem.Type = MemoryType(memType)
	}

	if metadataStr, ok := fields["metadata"].(string); ok {
		valkeyParseMetadata(mem, metadataStr)
	}

	if ts, ok := valkeySearchMillisField(fields, "created_at"); ok {
		mem.CreatedAt = time.UnixMilli(ts)
	}
	if ts, ok := valkeySearchMillisField(fields, "updated_at"); ok {
		mem.UpdatedAt = time.UnixMilli(ts)
	}
	if ts, ok := valkeySearchMillisField(fields, "last_accessed"); ok {
		mem.LastAccessed = time.UnixMilli(ts)
	}
	if accessCount, ok := valkeySearchIntField(fields, "access_count"); ok {
		mem.AccessCount = accessCount
	}

	return mem
}

func valkeySearchStringField(fields map[string]interface{}, key string) string {
	raw, ok := fields[key]
	if !ok || raw == nil {
		return ""
	}
	value := strings.TrimSpace(fmt.Sprint(raw))
	if value == "" || value == "<nil>" {
		return ""
	}
	return value
}

// valkeySearchContentField keeps the stored text, including leading and
// trailing whitespace. Identifier fields are trimmed; content is not.
func valkeySearchContentField(fields map[string]interface{}) string {
	raw, ok := fields["content"]
	if !ok || raw == nil {
		return ""
	}
	switch val := raw.(type) {
	case string:
		return val
	case []byte:
		return string(val)
	default:
		return fmt.Sprint(val)
	}
}

func valkeySearchMillisField(fields map[string]interface{}, key string) (int64, bool) {
	return valkeySearchInt64Field(fields, key)
}

func valkeySearchIntField(fields map[string]interface{}, key string) (int, bool) {
	value, ok := valkeySearchInt64Field(fields, key)
	if !ok {
		return 0, false
	}
	return int(value), true
}

func valkeySearchInt64Field(fields map[string]interface{}, key string) (int64, bool) {
	raw, ok := fields[key]
	if !ok || raw == nil {
		return 0, false
	}
	switch val := raw.(type) {
	case int:
		return int64(val), true
	case int64:
		return val, true
	case float64:
		if math.IsNaN(val) || math.IsInf(val, 0) {
			return 0, false
		}
		return int64(val), true
	case string:
		if val == "" {
			return 0, false
		}
		parsed, err := strconv.ParseInt(val, 10, 64)
		return parsed, err == nil
	case []byte:
		parsed, err := strconv.ParseInt(string(val), 10, 64)
		return parsed, err == nil
	default:
		parsed, err := strconv.ParseInt(strings.TrimSpace(fmt.Sprint(val)), 10, 64)
		return parsed, err == nil
	}
}
