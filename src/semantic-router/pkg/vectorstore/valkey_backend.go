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
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	glide "github.com/valkey-io/valkey-glide/go/v2"
	glideconfig "github.com/valkey-io/valkey-glide/go/v2/config"
)

// ValkeyBackendConfig holds configuration for the Valkey vector store backend.
type ValkeyBackendConfig struct {
	// Host is the Valkey server hostname (default "localhost").
	Host string
	// Port is the Valkey server port (default 6379).
	Port int
	// Password for Valkey authentication (optional).
	Password string
	// Database number (default 0).
	Database int
	// CollectionPrefix is the prefix for hash keys and index names (default "vsr_vs_").
	CollectionPrefix string
	// IndexM is the HNSW M parameter (default 16).
	IndexM int
	// IndexEf is the HNSW efConstruction parameter (default 200).
	IndexEf int
	// MetricType is the distance metric: "COSINE", "L2", or "IP" (default "COSINE").
	MetricType string
	// ConnectTimeout in seconds (default 10).
	ConnectTimeout int
}

// ValkeyBackend implements VectorStoreBackend using Valkey with the valkey-search module.
// All FT.* commands are issued via CustomCommand since valkey-glide Go client
// does not yet have native search API bindings (expected in v2.4, Q2 2026).
type ValkeyBackend struct {
	client           *glide.Client
	collectionPrefix string
	indexM           int
	indexEf          int
	metricType       string
}

// NewValkeyBackend creates a new Valkey vector store backend.
func NewValkeyBackend(cfg ValkeyBackendConfig) (*ValkeyBackend, error) {
	host := cfg.Host
	if host == "" {
		host = "localhost"
	}
	port := cfg.Port
	if port <= 0 {
		port = 6379
	}
	prefix := cfg.CollectionPrefix
	if prefix == "" {
		prefix = "vsr_vs_"
	}
	indexM := cfg.IndexM
	if indexM <= 0 {
		indexM = 16
	}
	indexEf := cfg.IndexEf
	if indexEf <= 0 {
		indexEf = 200
	}
	metricType := cfg.MetricType
	if metricType == "" {
		metricType = "COSINE"
	}
	timeout := cfg.ConnectTimeout
	if timeout <= 0 {
		timeout = 10
	}

	clientConfig := glideconfig.NewClientConfiguration().
		WithAddress(&glideconfig.NodeAddress{Host: host, Port: port}).
		WithRequestTimeout(time.Duration(timeout) * time.Second)

	if cfg.Password != "" {
		clientConfig = clientConfig.WithCredentials(
			glideconfig.NewServerCredentials("", cfg.Password),
		)
	}
	if cfg.Database != 0 {
		clientConfig = clientConfig.WithDatabaseId(cfg.Database)
	}

	client, err := glide.NewClient(clientConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Valkey at %s:%d: %w", host, port, err)
	}

	// Verify connectivity.
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
	defer cancel()
	_, err = client.Ping(ctx)
	if err != nil {
		client.Close()
		return nil, fmt.Errorf("failed to ping Valkey at %s:%d: %w", host, port, err)
	}

	return &ValkeyBackend{
		client:           client,
		collectionPrefix: prefix,
		indexM:           indexM,
		indexEf:          indexEf,
		metricType:       metricType,
	}, nil
}

// indexName returns the FT index name for a vector store collection.
func (v *ValkeyBackend) indexName(vectorStoreID string) string {
	return v.collectionPrefix + vectorStoreID + "_idx"
}

// keyPrefix returns the hash key prefix for a vector store collection.
func (v *ValkeyBackend) keyPrefix(vectorStoreID string) string {
	return v.collectionPrefix + vectorStoreID + ":"
}

// chunkKey returns the full hash key for a chunk within a collection.
func (v *ValkeyBackend) chunkKey(vectorStoreID string, chunkID string) string {
	return v.keyPrefix(vectorStoreID) + chunkID
}

// CreateCollection creates a Valkey FT index with HNSW vector field for the given collection.
// Maps to: FT.CREATE <index> ON HASH PREFIX 1 <prefix> SCHEMA ...
func (v *ValkeyBackend) CreateCollection(ctx context.Context, vectorStoreID string, dimension int) error {
	idxName := v.indexName(vectorStoreID)
	prefix := v.keyPrefix(vectorStoreID)

	// Check if index already exists.
	_, err := v.client.CustomCommand(ctx, []string{"FT.INFO", idxName})
	if err == nil {
		return fmt.Errorf("collection already exists: %s", vectorStoreID)
	}

	// Build FT.CREATE command:
	// FT.CREATE <idx> ON HASH PREFIX 1 <prefix> SCHEMA
	//   id TAG
	//   file_id TAG
	//   filename TAG
	//   content TEXT
	//   chunk_index NUMERIC
	//   created_at NUMERIC
	//   embedding VECTOR HNSW 10 TYPE FLOAT32 DIM <dim> DISTANCE_METRIC <metric> M <m> EF_CONSTRUCTION <ef>
	createCmd := []string{
		"FT.CREATE", idxName,
		"ON", "HASH",
		"PREFIX", "1", prefix,
		"SCHEMA",
		"id", "TAG",
		"file_id", "TAG",
		"filename", "TAG",
		"content", "TEXT",
		"chunk_index", "NUMERIC",
		"created_at", "NUMERIC",
		"embedding", "VECTOR", "HNSW", "10",
		"TYPE", "FLOAT32",
		"DIM", strconv.Itoa(dimension),
		"DISTANCE_METRIC", v.metricType,
		"M", strconv.Itoa(v.indexM),
		"EF_CONSTRUCTION", strconv.Itoa(v.indexEf),
	}

	_, err = v.client.CustomCommand(ctx, createCmd)
	if err != nil {
		return fmt.Errorf("failed to create index %s: %w", idxName, err)
	}

	return nil
}

// DeleteCollection drops the FT index and deletes all associated hash keys.
// Maps to: FT.DROPINDEX <index> followed by SCAN+DEL cleanup.
func (v *ValkeyBackend) DeleteCollection(ctx context.Context, vectorStoreID string) error {
	idxName := v.indexName(vectorStoreID)

	// FT.DROPINDEX <name> — valkey-search does not support the DD flag.
	// Associated hash keys remain but are no longer indexed.
	_, err := v.client.CustomCommand(ctx, []string{"FT.DROPINDEX", idxName})
	if err != nil {
		return fmt.Errorf("failed to drop index %s: %w", idxName, err)
	}

	// Clean up hash keys that were under this index prefix.
	prefix := v.keyPrefix(vectorStoreID)
	v.deleteKeysByPrefix(ctx, prefix)

	return nil
}

// deleteKeysByPrefix uses SCAN to find and DEL all keys matching the given prefix.
// Best-effort: errors during cleanup are silently ignored since the index is already dropped.
func (v *ValkeyBackend) deleteKeysByPrefix(ctx context.Context, prefix string) {
	cursor := "0"
	pattern := prefix + "*"
	for {
		result, err := v.client.CustomCommand(ctx, []string{"SCAN", cursor, "MATCH", pattern, "COUNT", "100"})
		if err != nil {
			return
		}
		arr, ok := result.([]interface{})
		if !ok || len(arr) < 2 {
			return
		}
		cursor = fmt.Sprint(arr[0])

		var keys []string
		switch keyList := arr[1].(type) {
		case []interface{}:
			for _, k := range keyList {
				if s, ok := k.(string); ok {
					keys = append(keys, s)
				}
			}
		}
		if len(keys) > 0 {
			v.client.Del(ctx, keys)
		}
		if cursor == "0" {
			return
		}
	}
}

// CollectionExists checks if a Valkey FT index exists for the given collection.
// Maps to: FT.INFO <index>
func (v *ValkeyBackend) CollectionExists(ctx context.Context, vectorStoreID string) (bool, error) {
	idxName := v.indexName(vectorStoreID)

	_, err := v.client.CustomCommand(ctx, []string{"FT.INFO", idxName})
	if err != nil {
		// FT.INFO returns an error if the index does not exist.
		// valkey-search: "Index: with name '...' not found in database N"
		// Redis Stack: "Unknown index name" or "unknown"
		errMsg := strings.ToLower(err.Error())
		if strings.Contains(errMsg, "not found") || strings.Contains(errMsg, "unknown") {
			return false, nil
		}
		return false, fmt.Errorf("failed to check index %s: %w", idxName, err)
	}
	return true, nil
}

// InsertChunks inserts embedded chunks into the Valkey collection as hash keys.
// Each chunk is stored as a hash with fields: id, file_id, filename, content,
// chunk_index, created_at, and embedding (as raw float32 bytes).
func (v *ValkeyBackend) InsertChunks(ctx context.Context, vectorStoreID string, chunks []EmbeddedChunk) error {
	if len(chunks) == 0 {
		return nil
	}

	now := strconv.FormatInt(time.Now().Unix(), 10)

	for _, chunk := range chunks {
		key := v.chunkKey(vectorStoreID, chunk.ID)

		fields := map[string]string{
			"id":          chunk.ID,
			"file_id":     chunk.FileID,
			"filename":    chunk.Filename,
			"content":     chunk.Content,
			"chunk_index": strconv.Itoa(chunk.ChunkIndex),
			"created_at":  now,
			"embedding":   string(float32SliceToBytes(chunk.Embedding)),
		}

		_, err := v.client.HSet(ctx, key, fields)
		if err != nil {
			return fmt.Errorf("failed to insert chunk %s into %s: %w", chunk.ID, vectorStoreID, err)
		}
	}

	return nil
}

// DeleteByFileID removes all chunks associated with a file from the collection.
// Uses FT.SEARCH to find matching keys by file_id tag, then DEL to remove them.
func (v *ValkeyBackend) DeleteByFileID(ctx context.Context, vectorStoreID string, fileID string) error {
	if !safeIdentifierPattern.MatchString(fileID) {
		return fmt.Errorf("invalid file ID: contains disallowed characters")
	}

	idxName := v.indexName(vectorStoreID)

	// Search for all chunks with this file_id.
	// Avoid NOCONTENT and RETURN 0 — valkey-glide v2 cannot parse those response
	// formats (BulkString error). Use RETURN 1 with a lightweight field instead,
	// which produces the standard map-based response that valkey-glide expects.
	searchCmd := []string{
		"FT.SEARCH", idxName,
		fmt.Sprintf("@file_id:{%s}", escapeTagValue(fileID)),
		"RETURN", "1", "id",
		"LIMIT", "0", "10000",
	}

	result, err := v.client.CustomCommand(ctx, searchCmd)
	if err != nil {
		return fmt.Errorf("failed to search for file %s in %s: %w", fileID, vectorStoreID, err)
	}

	keys := extractKeysFromSearchResult(result)
	if len(keys) == 0 {
		return nil
	}

	_, err = v.client.Del(ctx, keys)
	if err != nil {
		return fmt.Errorf("failed to delete chunks for file %s from %s: %w", fileID, vectorStoreID, err)
	}

	return nil
}

// Search performs a KNN vector similarity search in the Valkey collection.
// Uses the AS vector_distance alias (same pattern as the Valkey cache backend).
// Results are automatically sorted by distance (ascending).
func (v *ValkeyBackend) Search(
	ctx context.Context, vectorStoreID string, queryEmbedding []float32,
	topK int, threshold float32, filter map[string]interface{},
) ([]SearchResult, error) {
	idxName := v.indexName(vectorStoreID)
	embeddingBytes := float32SliceToBytes(queryEmbedding)

	// Build filter expression.
	filterExpr := "*"
	if filter != nil {
		if fid, ok := filter["file_id"].(string); ok && fid != "" {
			if !safeIdentifierPattern.MatchString(fid) {
				return nil, fmt.Errorf("invalid file_id filter: contains disallowed characters")
			}
			filterExpr = fmt.Sprintf("@file_id:{%s}", escapeTagValue(fid))
		}
	}

	// Build KNN query with AS alias, following the same pattern as the Valkey cache backend (PR #1540).
	// valkey-glide returns the aliased field in the per-document field map.
	query := fmt.Sprintf("(%s)=>[KNN %d @embedding $BLOB AS vector_distance]", filterExpr, topK)

	searchCmd := []string{
		"FT.SEARCH", idxName, query,
		"PARAMS", "2", "BLOB", string(embeddingBytes),
		"LIMIT", "0", strconv.Itoa(topK),
		"DIALECT", "2",
	}

	result, err := v.client.CustomCommand(ctx, searchCmd)
	if err != nil {
		return nil, fmt.Errorf("failed to search in %s: %w", idxName, err)
	}

	return v.parseSearchResults(result, threshold)
}

// Close releases the Valkey client connection.
func (v *ValkeyBackend) Close() error {
	if v.client != nil {
		v.client.Close()
	}
	return nil
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

// float32SliceToBytes converts a float32 slice to a little-endian byte slice
// suitable for Valkey vector storage.
func float32SliceToBytes(floats []float32) []byte {
	buf := make([]byte, len(floats)*4)
	for i, f := range floats {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// escapeTagValue escapes special characters in a Valkey tag filter value.
// Tag values containing hyphens, dots, or other special chars need escaping.
func escapeTagValue(val string) string {
	replacer := strings.NewReplacer(
		"-", "\\-",
		".", "\\.",
		":", "\\:",
		"/", "\\/",
		" ", "\\ ",
	)
	return replacer.Replace(val)
}

// extractKeysFromSearchResult extracts document keys from an FT.SEARCH NOCONTENT result.
// valkey-glide v2 returns NOCONTENT results as: [totalCount, "key1", "key2", ...]
// but may also return map-based results: [totalCount, map[key: map[...]], ...].
// This function handles both formats.
func extractKeysFromSearchResult(result any) []string {
	arr, ok := result.([]interface{})
	if !ok || len(arr) < 2 {
		return nil
	}

	var keys []string
	for i := 1; i < len(arr); i++ {
		switch v := arr[i].(type) {
		case string:
			// NOCONTENT flat format: element is the key string directly.
			keys = append(keys, v)
		case map[string]interface{}:
			// Map format: key is the map key.
			for docKey := range v {
				keys = append(keys, docKey)
			}
		}
	}
	return keys
}

// parseSearchResults parses the FT.SEARCH result into SearchResult slice.
// valkey-glide v2 CustomCommand returns results as:
//
//	[totalCount, map[docKey1: map[fields...], docKey2: map[fields...], ...]]
//
// All documents are returned in a single map (the second element).
// This follows the same pattern as the Valkey cache backend (PR #1540).
func (v *ValkeyBackend) parseSearchResults(result any, threshold float32) ([]SearchResult, error) {
	arr, ok := result.([]interface{})
	if !ok || len(arr) < 1 {
		return nil, nil
	}

	totalCount := toInt64(arr[0])
	if totalCount == 0 {
		return nil, nil
	}

	var results []SearchResult

	// valkey-glide may return docs as a single map or multiple map entries.
	for i := 1; i < len(arr); i++ {
		docMap, ok := arr[i].(map[string]interface{})
		if !ok {
			continue
		}

		// Each key in docMap is a document key, value is the fields map.
		for _, docValue := range docMap {
			fieldsMap, mapOk := docValue.(map[string]interface{})
			if !mapOk {
				continue
			}

			score := parseScoreFromMap(fieldsMap, "vector_distance", v.metricType)
			if score < float64(threshold) {
				continue
			}

			chunkIndex, _ := strconv.Atoi(fmt.Sprint(fieldsMap["chunk_index"]))

			results = append(results, SearchResult{
				FileID:     fmt.Sprint(fieldsMap["file_id"]),
				Filename:   fmt.Sprint(fieldsMap["filename"]),
				Content:    fmt.Sprint(fieldsMap["content"]),
				Score:      score,
				ChunkIndex: chunkIndex,
			})
		}
	}

	// Sort by score descending so callers always get best matches first.
	// FT.SEARCH returns results ordered by distance, but Go map iteration
	// in the valkey-glide response loses that ordering.
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results, nil
}

// parseScoreFromMap extracts a distance value from the fields map and converts it to similarity.
func parseScoreFromMap(fields map[string]interface{}, key string, metricType string) float64 {
	raw, exists := fields[key]
	if !exists {
		return 0
	}
	var distance float64
	if _, err := fmt.Sscanf(fmt.Sprint(raw), "%f", &distance); err != nil {
		return 0
	}
	return distanceToSimilarity(metricType, distance)
}


// distanceToSimilarity converts a vector distance to a similarity score based on the metric type.
// Follows the same conversion as the Valkey cache backend (PR #1540).
func distanceToSimilarity(metricType string, distance float64) float64 {
	switch strings.ToUpper(metricType) {
	case "COSINE":
		return float64(1.0 - float32(distance)/2.0)
	case "L2":
		return float64(1.0 / (1.0 + float32(distance)))
	case "IP":
		return distance
	default:
		return float64(1.0 - float32(distance))
	}
}

// toInt64 converts an interface{} to int64, handling both int64 and string representations.
func toInt64(v interface{}) int64 {
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
