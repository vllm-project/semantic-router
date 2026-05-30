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
	"strconv"
	"strings"
	"time"

	glide "github.com/valkey-io/valkey-glide/go/v2"
	glideconfig "github.com/valkey-io/valkey-glide/go/v2/config"

	valkeyutil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/valkey"
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

// valkeyDefaults applies default values to a ValkeyBackendConfig, returning
// the resolved host, port, prefix, indexM, indexEf, metricType, and timeout.
// Returns an error if the metric type is unsupported.
func valkeyDefaults(cfg ValkeyBackendConfig) (string, int, string, int, int, string, int, error) {
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
	metricType := strings.ToUpper(cfg.MetricType)
	if metricType == "" {
		metricType = "COSINE"
	}
	switch metricType {
	case "COSINE", "L2", "IP":
		// valid
	default:
		return "", 0, "", 0, 0, "", 0, fmt.Errorf("unsupported metric type: %s (supported: COSINE, L2, IP)", metricType)
	}
	timeout := cfg.ConnectTimeout
	if timeout <= 0 {
		timeout = 10
	}
	return host, port, prefix, indexM, indexEf, metricType, timeout, nil
}

// NewValkeyBackend creates a new Valkey vector store backend.
func NewValkeyBackend(cfg ValkeyBackendConfig) (*ValkeyBackend, error) {
	host, port, prefix, indexM, indexEf, metricType, timeout, err := valkeyDefaults(cfg)
	if err != nil {
		return nil, err
	}

	clientConfig := glideconfig.NewClientConfiguration().
		WithAddress(&glideconfig.NodeAddress{Host: host, Port: port}).
		WithClientName("vllm_vector_store_client").
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

	versionCtx, versionCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer versionCancel()
	if err := valkeyutil.EnsureSearchModuleVersion(versionCtx, client, valkeyutil.SearchModuleMinVersion); err != nil {
		client.Close()
		return nil, err
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

	// Delete all chunks in batches of pageSize until none remain.
	// We always search at LIMIT 0 (offset 0): once a batch is deleted, the
	// remaining documents shift to the front of the result set, so re-scanning
	// from offset 0 is the only correct approach when deleting between pages.
	const pageSize = 1000

	for {
		searchCmd := []string{
			"FT.SEARCH", idxName,
			fmt.Sprintf("@file_id:{%s}", escapeTagValue(fileID)),
			"RETURN", "1", "id",
			"LIMIT", "0", strconv.Itoa(pageSize),
		}

		result, err := v.client.CustomCommand(ctx, searchCmd)
		if err != nil {
			return fmt.Errorf("failed to search for file %s in %s: %w", fileID, vectorStoreID, err)
		}

		keys := extractKeysFromSearchResult(result)
		if len(keys) == 0 {
			break
		}

		_, err = v.client.Del(ctx, keys)
		if err != nil {
			return fmt.Errorf("failed to delete chunks for file %s from %s: %w", fileID, vectorStoreID, err)
		}
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

	// Build KNN query with AS alias. The filter expression is wrapped in parentheses
	// so it can be either "*" (match-all) or a field predicate like "@file_id:{...}".
	// valkey-glide returns the aliased field in the per-document field map.
	query := fmt.Sprintf("(%s)=>[KNN %d @embedding $BLOB AS vector_distance]", filterExpr, topK)

	searchCmd := []string{
		"FT.SEARCH", idxName, query,
		"PARAMS", "2", "BLOB", string(embeddingBytes),
		"RETURN", "5", "file_id", "filename", "content", "chunk_index", "vector_distance",
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
