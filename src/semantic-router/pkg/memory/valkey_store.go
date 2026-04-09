package memory

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	glide "github.com/valkey-io/valkey-glide/go/v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ValkeyStore provides memory storage and retrieval using Valkey with the Search module.
// Implements the Store interface with HASH-based storage and FT.SEARCH for vector similarity.
type ValkeyStore struct {
	client           *glide.Client
	config           config.MemoryConfig
	valkeyConfig     *config.MemoryValkeyConfig
	enabled          bool
	maxRetries       int
	retryBaseDelay   time.Duration
	embeddingConfig  EmbeddingConfig
	indexName        string
	collectionPrefix string
	metricType       string
	dimension        int
}

// ValkeyStoreOptions contains configuration for creating a ValkeyStore.
type ValkeyStoreOptions struct {
	// Client is the valkey-glide client instance.
	Client *glide.Client
	// Config is the memory configuration.
	Config config.MemoryConfig
	// ValkeyConfig is the Valkey-specific configuration.
	ValkeyConfig *config.MemoryValkeyConfig
	// Enabled controls whether the store is active.
	Enabled bool
	// EmbeddingConfig is the unified embedding configuration.
	EmbeddingConfig *EmbeddingConfig
}

// NewValkeyStore creates a new ValkeyStore instance.
func NewValkeyStore(options ValkeyStoreOptions) (*ValkeyStore, error) {
	if !options.Enabled {
		logging.Debugf("ValkeyStore: disabled, returning stub")
		return &ValkeyStore{enabled: false}, nil
	}

	if options.Client == nil {
		return nil, fmt.Errorf("valkey client is required")
	}

	if options.ValkeyConfig == nil {
		return nil, fmt.Errorf("valkey config is required")
	}

	cfg := options.Config
	if cfg.EmbeddingModel == "" {
		cfg = DefaultMemoryConfig()
	}

	var embeddingCfg EmbeddingConfig
	if options.EmbeddingConfig != nil {
		embeddingCfg = *options.EmbeddingConfig
	} else {
		embeddingCfg = EmbeddingConfig{Model: EmbeddingModelBERT}
	}

	vc := options.ValkeyConfig

	indexName := vc.IndexName
	if indexName == "" {
		indexName = "mem_idx"
	}
	collectionPrefix := vc.CollectionPrefix
	if collectionPrefix == "" {
		collectionPrefix = "mem:"
	}
	metricType := vc.MetricType
	if metricType == "" {
		metricType = "COSINE"
	}
	dimension := vc.Dimension
	if dimension <= 0 {
		dimension = 384
	}

	store := &ValkeyStore{
		client:           options.Client,
		config:           cfg,
		valkeyConfig:     vc,
		enabled:          options.Enabled,
		maxRetries:       DefaultMaxRetries,
		retryBaseDelay:   DefaultRetryBaseDelay * time.Millisecond,
		embeddingConfig:  embeddingCfg,
		indexName:        indexName,
		collectionPrefix: collectionPrefix,
		metricType:       strings.ToUpper(metricType),
		dimension:        dimension,
	}

	// Use the configured timeout for index initialization (default 10s).
	indexTimeout := 10 * time.Second
	if vc.Timeout > 0 {
		indexTimeout = time.Duration(vc.Timeout) * time.Second
	}
	ctx, cancel := context.WithTimeout(context.Background(), indexTimeout)
	defer cancel()
	if err := store.ensureIndex(ctx); err != nil {
		return nil, fmt.Errorf("failed to ensure index exists: %w", err)
	}

	logging.Infof("ValkeyStore: initialized with index='%s', prefix='%s', embedding_model='%s', dimension=%d",
		store.indexName, store.collectionPrefix, store.embeddingConfig.Model, store.dimension)

	return store, nil
}

// ensureIndex checks if the FT index exists and creates it if not.
func (v *ValkeyStore) ensureIndex(ctx context.Context) error {
	_, err := v.client.CustomCommand(ctx, []string{"FT.INFO", v.indexName})
	if err == nil {
		logging.Debugf("ValkeyStore: index '%s' already exists", v.indexName)
		return nil
	}

	logging.Infof("ValkeyStore: creating index '%s' with dimension %d", v.indexName, v.dimension)

	indexM := v.valkeyConfig.IndexM
	if indexM <= 0 {
		indexM = 16
	}
	efConstruction := v.valkeyConfig.IndexEfConstruction
	if efConstruction <= 0 {
		efConstruction = 256
	}

	createCmd := []string{
		"FT.CREATE", v.indexName,
		"ON", "HASH",
		"PREFIX", "1", v.collectionPrefix,
		"SCHEMA",
		"id", "TAG",
		"user_id", "TAG",
		"project_id", "TAG",
		"memory_type", "TAG",
		"content", "TEXT",
		"source", "TAG",
		"embedding", "VECTOR", "HNSW", "10",
		"TYPE", "FLOAT32",
		"DIM", strconv.Itoa(v.dimension),
		"DISTANCE_METRIC", v.metricType,
		"M", strconv.Itoa(indexM),
		"EF_CONSTRUCTION", strconv.Itoa(efConstruction),
		"created_at", "NUMERIC", "SORTABLE",
		"updated_at", "NUMERIC",
		"access_count", "NUMERIC",
		"importance", "NUMERIC",
	}

	_, err = v.client.CustomCommand(ctx, createCmd)
	if err != nil {
		return fmt.Errorf("FT.CREATE failed: %w", err)
	}

	logging.Infof("ValkeyStore: index '%s' created successfully", v.indexName)
	return nil
}

// hashKey returns the HASH key for a memory document.
func (v *ValkeyStore) hashKey(id string) string {
	return v.collectionPrefix + id
}

// Store saves a new memory to Valkey.
// Generates embedding for the content and inserts as a HASH key.
func (v *ValkeyStore) Store(ctx context.Context, memory *Memory) error {
	startTime := time.Now()
	backend := "valkey"
	operation := "store"
	status := "success"

	defer func() {
		duration := time.Since(startTime).Seconds()
		RecordMemoryStoreOperation(backend, operation, status, duration)
	}()

	if !v.enabled {
		status = "error"
		return fmt.Errorf("valkey store is not enabled")
	}

	if err := valkeyValidateMemory(memory); err != nil {
		status = "error"
		return err
	}

	logging.Debugf("ValkeyStore.Store: id=%s, user=%s, type=%s, content_len=%d",
		memory.ID, memory.UserID, memory.Type, len(memory.Content))

	var embedding []float32
	if len(memory.Embedding) > 0 {
		embedding = memory.Embedding
	} else {
		var err error
		embedding, err = GenerateEmbedding(memory.Content, v.embeddingConfig)
		if err != nil {
			status = "error"
			return fmt.Errorf("failed to generate embedding: %w", err)
		}
	}

	now := time.Now()
	if memory.CreatedAt.IsZero() {
		memory.CreatedAt = now
	}
	memory.UpdatedAt = now
	if memory.LastAccessed.IsZero() {
		memory.LastAccessed = now
	}

	fields, err := valkeyBuildHashFields(memory, embedding)
	if err != nil {
		status = "error"
		return fmt.Errorf("failed to build hash fields: %w", err)
	}

	key := v.hashKey(memory.ID)

	err = v.retryWithBackoff(ctx, func() error {
		_, hsetErr := v.client.HSet(ctx, key, fields)
		return hsetErr
	})
	if err != nil {
		status = "error"
		return fmt.Errorf("valkey HSET failed for memory id=%s: %w", memory.ID, err)
	}

	logging.Debugf("ValkeyStore.Store: successfully stored memory id=%s", memory.ID)
	return nil
}

// rerankAndFilter applies hybrid re-ranking, adaptive threshold, score filtering, and access tracking.
func (v *ValkeyStore) rerankAndFilter(candidates []*RetrieveResult, opts RetrieveOptions, threshold float32, limit int) []*RetrieveResult {
	if opts.HybridSearch && len(candidates) > 1 {
		candidates = v.hybridRerank(candidates, opts)
	}

	if opts.AdaptiveThreshold && len(candidates) > 1 {
		threshold = adaptiveThresholdElbow(candidates, threshold)
	}

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

	if len(results) > 0 {
		ids := make([]string, len(results))
		for i, r := range results {
			ids[i] = r.Memory.ID
		}
		go v.recordRetrievalBatch(ids)
	}

	return results
}

// buildRetrieveSearchCmd constructs the FT.SEARCH command for vector similarity retrieval.
// Note: RETURN fetches fields from the underlying HASH document, not from the
// index schema. The "metadata" field is stored in the HASH but not indexed as
// a schema field, which is intentional — it's only needed for result parsing.
func (v *ValkeyStore) buildRetrieveSearchCmd(opts RetrieveOptions, embedding []float32, limit int) []string {
	filterExpr := fmt.Sprintf("@user_id:{%s}", valkeyEscapeTagValue(opts.UserID))

	if len(opts.Types) > 0 {
		typeValues := make([]string, len(opts.Types))
		for i, memType := range opts.Types {
			typeValues[i] = valkeyEscapeTagValue(string(memType))
		}
		filterExpr = fmt.Sprintf("%s @memory_type:{%s}", filterExpr, strings.Join(typeValues, " | "))
	}

	searchTopK := limit * 4
	if opts.HybridSearch {
		searchTopK = limit * 8
	}
	if searchTopK < 20 {
		searchTopK = 20
	}

	embeddingBytes := valkeyFloat32ToBytes(embedding)
	query := fmt.Sprintf("(%s)=>[KNN %d @embedding $BLOB AS vector_distance]", filterExpr, searchTopK)

	return []string{
		"FT.SEARCH", v.indexName, query,
		"PARAMS", "2", "BLOB", string(embeddingBytes),
		"RETURN", "5", "id", "content", "memory_type", "metadata", "vector_distance",
		"LIMIT", "0", strconv.Itoa(searchTopK),
		"DIALECT", "2",
	}
}

// Retrieve searches for memories in Valkey with similarity threshold filtering.
func (v *ValkeyStore) Retrieve(ctx context.Context, opts RetrieveOptions) ([]*RetrieveResult, error) {
	startTime := time.Now()
	backend := "valkey"
	operation := "retrieve"
	status := "success"
	resultCount := 0

	defer func() {
		duration := time.Since(startTime).Seconds()
		RecordMemoryRetrieval(backend, operation, status, opts.UserID, duration, resultCount)
	}()

	if !v.enabled {
		status = "error"
		return nil, fmt.Errorf("valkey store is not enabled")
	}

	limit := opts.Limit
	if limit <= 0 {
		limit = v.config.DefaultRetrievalLimit
	}

	threshold := opts.Threshold
	if threshold <= 0 {
		threshold = v.config.DefaultSimilarityThreshold
	}

	if err := valkeyValidateRetrieveOpts(opts); err != nil {
		status = "error"
		return nil, err
	}

	logging.Debugf("ValkeyStore.Retrieve: query='%s', user_id='%s', limit=%d, threshold=%.4f, hybrid=%v (mode=%s)",
		opts.Query, opts.UserID, limit, threshold, opts.HybridSearch, opts.HybridMode)

	embedding, err := GenerateEmbedding(opts.Query, v.embeddingConfig)
	if err != nil {
		status = "error"
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	searchCmd := v.buildRetrieveSearchCmd(opts, embedding, limit)

	var searchResult any
	err = v.retryWithBackoff(ctx, func() error {
		var retryErr error
		searchResult, retryErr = v.client.CustomCommand(ctx, searchCmd)
		return retryErr
	})
	if err != nil {
		status = "error"
		return nil, fmt.Errorf("valkey FT.SEARCH failed after retries: %w", err)
	}

	candidates := v.parseSearchCandidates(searchResult, opts.UserID)
	if len(candidates) == 0 {
		status = "miss"
		return []*RetrieveResult{}, nil
	}

	results := v.rerankAndFilter(candidates, opts, threshold, limit)
	resultCount = len(results)
	if resultCount > 0 {
		status = "hit"
	} else {
		status = "miss"
	}

	return results, nil
}

// Get retrieves a memory by ID from Valkey.
func (v *ValkeyStore) Get(ctx context.Context, id string) (*Memory, error) {
	if !v.enabled {
		return nil, fmt.Errorf("valkey store is not enabled")
	}

	if id == "" {
		return nil, fmt.Errorf("memory ID is required")
	}

	logging.Debugf("ValkeyStore.Get: retrieving memory id=%s", id)

	key := v.hashKey(id)

	var fields map[string]string
	err := v.retryWithBackoff(ctx, func() error {
		var retryErr error
		fields, retryErr = v.client.HGetAll(ctx, key)
		return retryErr
	})
	if err != nil {
		return nil, fmt.Errorf("valkey HGETALL failed for memory id=%s: %w", id, err)
	}

	if len(fields) == 0 {
		return nil, fmt.Errorf("memory not found: %s", id)
	}

	mem := valkeyFieldsToMemory(fields)
	if mem.ID == "" {
		return nil, fmt.Errorf("memory not found: %s", id)
	}

	logging.Debugf("ValkeyStore.Get: found memory id=%s, user_id=%s", mem.ID, mem.UserID)
	return mem, nil
}

// Update modifies an existing memory in Valkey using HSET (atomic overwrite).
// Preserves CreatedAt from the existing row and sets UpdatedAt to now.
func (v *ValkeyStore) Update(ctx context.Context, id string, memory *Memory) error {
	startTime := time.Now()
	backend := "valkey"
	operation := "update"
	status := "success"

	defer func() {
		duration := time.Since(startTime).Seconds()
		RecordMemoryStoreOperation(backend, operation, status, duration)
	}()

	if !v.enabled {
		status = "error"
		return fmt.Errorf("valkey store is not enabled")
	}

	if id == "" {
		status = "error"
		return fmt.Errorf("memory ID is required")
	}

	logging.Debugf("ValkeyStore.Update: upserting memory id=%s", id)

	memory.ID = id
	memory.UpdatedAt = time.Now()

	// If CreatedAt or Embedding are missing, fetch from the existing row so we don't lose data
	if memory.CreatedAt.IsZero() || len(memory.Embedding) == 0 {
		existing, err := v.Get(ctx, id)
		if err != nil {
			status = "error"
			return fmt.Errorf("memory not found: %s", id)
		}
		if memory.CreatedAt.IsZero() {
			memory.CreatedAt = existing.CreatedAt
		}
		if len(memory.Embedding) == 0 {
			memory.Embedding = existing.Embedding
		}
	}

	err := v.upsert(ctx, memory)
	if err != nil {
		status = "error"
		return err
	}
	return nil
}

// upsert atomically replaces a memory in Valkey by HSET on its hash key.
// The memory must be fully populated (including Embedding, timestamps, etc.).
// Reuses valkeyBuildHashFields to avoid duplicating field-building logic.
func (v *ValkeyStore) upsert(ctx context.Context, memory *Memory) error {
	if len(memory.Embedding) == 0 {
		return fmt.Errorf("embedding is required for upsert")
	}

	fields, err := valkeyBuildHashFields(memory, memory.Embedding)
	if err != nil {
		return fmt.Errorf("failed to build hash fields: %w", err)
	}

	key := v.hashKey(memory.ID)

	err = v.retryWithBackoff(ctx, func() error {
		_, hsetErr := v.client.HSet(ctx, key, fields)
		return hsetErr
	})
	if err != nil {
		return fmt.Errorf("valkey HSET upsert failed for memory id=%s: %w", memory.ID, err)
	}

	logging.Debugf("ValkeyStore.upsert: successfully upserted memory id=%s", memory.ID)
	return nil
}

// List returns memories matching the filter criteria with pagination.
// Uses FT.SEARCH with TAG filters (no vector search) and returns paginated results.
// Total count comes from the FT.SEARCH header (first element of the result array),
// which reports the full match count regardless of LIMIT, so we only fetch the
// requested page.
func (v *ValkeyStore) List(ctx context.Context, opts ListOptions) (*ListResult, error) {
	if !v.enabled {
		return nil, fmt.Errorf("valkey store is not enabled")
	}

	if opts.UserID == "" {
		return nil, fmt.Errorf("user ID is required for listing memories")
	}

	limit := opts.Limit
	if limit <= 0 {
		limit = 20
	}
	if limit > 100 {
		limit = 100
	}

	logging.Debugf("ValkeyStore.List: user_id=%s, types=%v, limit=%d",
		opts.UserID, opts.Types, limit)

	// Build filter expression
	filterExpr := fmt.Sprintf("@user_id:{%s}", valkeyEscapeTagValue(opts.UserID))

	if len(opts.Types) > 0 {
		typeValues := make([]string, len(opts.Types))
		for i, memType := range opts.Types {
			typeValues[i] = valkeyEscapeTagValue(string(memType))
		}
		filterExpr = fmt.Sprintf("%s @memory_type:{%s}", filterExpr, strings.Join(typeValues, " | "))
	}

	// Fetch limit+1 pages worth of data so we can sort client-side and still
	// respect the limit. We over-fetch by a factor to allow client-side sorting
	// by created_at (FT.SEARCH does not support ORDER BY on NUMERIC fields
	// without SORTABLE in all Valkey Search versions).
	// The total count comes from the FT.SEARCH header element.
	fetchLimit := limit * 5
	if fetchLimit < 100 {
		fetchLimit = 100
	}
	if fetchLimit > 10000 {
		fetchLimit = 10000
	}

	searchCmd := []string{
		"FT.SEARCH", v.indexName, filterExpr,
		"RETURN", "7", "id", "content", "user_id", "memory_type", "metadata", "created_at", "updated_at",
		"LIMIT", "0", strconv.Itoa(fetchLimit),
		"DIALECT", "2",
	}

	result, err := v.client.CustomCommand(ctx, searchCmd)
	if err != nil {
		return nil, fmt.Errorf("valkey FT.SEARCH list failed: %w", err)
	}

	// Extract total count from the FT.SEARCH header.
	total := v.extractTotalCount(result)

	memories := v.parseListSearchResults(result)

	// Sort by created_at descending for deterministic results.
	sort.Slice(memories, func(i, j int) bool {
		return memories[i].CreatedAt.After(memories[j].CreatedAt)
	})

	if limit < len(memories) {
		memories = memories[:limit]
	}

	logging.Debugf("ValkeyStore.List: found %d total, returning %d (limit=%d)",
		total, len(memories), limit)

	return &ListResult{
		Memories: memories,
		Total:    total,
		Limit:    limit,
	}, nil
}

// Forget deletes a memory by ID from Valkey.
func (v *ValkeyStore) Forget(ctx context.Context, id string) error {
	startTime := time.Now()
	backend := "valkey"
	operation := "forget"
	status := "success"

	defer func() {
		duration := time.Since(startTime).Seconds()
		RecordMemoryStoreOperation(backend, operation, status, duration)
	}()

	if !v.enabled {
		status = "error"
		return fmt.Errorf("valkey store is not enabled")
	}

	if id == "" {
		status = "error"
		return fmt.Errorf("memory ID is required")
	}

	logging.Debugf("ValkeyStore.Forget: deleting memory id=%s", id)

	key := v.hashKey(id)

	err := v.retryWithBackoff(ctx, func() error {
		deleted, delErr := v.client.Del(ctx, []string{key})
		if delErr != nil {
			return delErr
		}
		if deleted == 0 {
			logging.Debugf("ValkeyStore.Forget: key %s did not exist (already deleted)", key)
		}
		return nil
	})
	if err != nil {
		status = "error"
		return fmt.Errorf("valkey DEL failed for memory id=%s: %w", id, err)
	}

	logging.Debugf("ValkeyStore.Forget: successfully deleted memory id=%s", id)
	return nil
}

// ForgetByScope deletes all memories matching the scope from Valkey.
// Scope includes UserID (required), ProjectID (optional), Types (optional).
// Uses batch DEL for efficiency instead of one-by-one deletion.
func (v *ValkeyStore) ForgetByScope(ctx context.Context, scope MemoryScope) error {
	startTime := time.Now()
	backend := "valkey"
	operation := "forget_by_scope"
	status := "success"

	defer func() {
		duration := time.Since(startTime).Seconds()
		RecordMemoryStoreOperation(backend, operation, status, duration)
	}()

	if !v.enabled {
		status = "error"
		return fmt.Errorf("valkey store is not enabled")
	}

	if scope.UserID == "" {
		status = "error"
		return fmt.Errorf("user ID is required for scope deletion")
	}

	logging.Debugf("ValkeyStore.ForgetByScope: deleting memories for user_id=%s, project_id=%s, types=%v",
		scope.UserID, scope.ProjectID, scope.Types)

	// Build filter expression.
	// project_id is indexed as a TAG field in the schema, so we can filter
	// directly when the caller specifies a ProjectID.
	filterExpr := fmt.Sprintf("@user_id:{%s}", valkeyEscapeTagValue(scope.UserID))

	if scope.ProjectID != "" {
		filterExpr = fmt.Sprintf("%s @project_id:{%s}", filterExpr, valkeyEscapeTagValue(scope.ProjectID))
	}

	if len(scope.Types) > 0 {
		typeValues := make([]string, len(scope.Types))
		for i, memType := range scope.Types {
			typeValues[i] = valkeyEscapeTagValue(string(memType))
		}
		filterExpr = fmt.Sprintf("%s @memory_type:{%s}", filterExpr, strings.Join(typeValues, " | "))
	}

	// Delete in batches: always search from offset 0, delete found keys, repeat
	// until none remain. We intentionally re-query at offset 0 each iteration
	// because the previous batch's DEL removes those keys from the index, so the
	// next FT.SEARCH at offset 0 naturally returns the next page of results.
	// Incrementing the offset would skip keys that shifted forward after deletion.
	// This follows the same pattern as ValkeyBackend.DeleteByFileID in
	// pkg/vectorstore/valkey_backend.go.
	const pageSize = 1000
	totalDeleted := 0

	for {
		searchCmd := []string{
			"FT.SEARCH", v.indexName, filterExpr,
			"RETURN", "1", "id",
			"LIMIT", "0", strconv.Itoa(pageSize),
			"DIALECT", "2",
		}

		result, err := v.client.CustomCommand(ctx, searchCmd)
		if err != nil {
			status = "error"
			return fmt.Errorf("valkey FT.SEARCH failed for scope deletion: %w", err)
		}

		keys := v.extractHashKeysFromSearchResult(result)
		if len(keys) == 0 {
			break
		}

		_, err = v.client.Del(ctx, keys)
		if err != nil {
			status = "error"
			return fmt.Errorf("valkey batch DEL failed for scope deletion: %w", err)
		}

		totalDeleted += len(keys)
	}

	logging.Debugf("ValkeyStore.ForgetByScope: deleted %d memories", totalDeleted)
	return nil
}

// IsEnabled returns whether the store is enabled.
func (v *ValkeyStore) IsEnabled() bool {
	return v.enabled
}

// CheckConnection verifies the Valkey connection is healthy.
func (v *ValkeyStore) CheckConnection(ctx context.Context) error {
	if !v.enabled {
		return nil
	}

	if v.client == nil {
		return fmt.Errorf("valkey client is not initialized")
	}

	// Verify the FT index exists
	_, err := v.client.CustomCommand(ctx, []string{"FT.INFO", v.indexName})
	if err != nil {
		return fmt.Errorf("valkey FT.INFO failed for index '%s': %w", v.indexName, err)
	}

	return nil
}

// Close releases resources held by the store.
func (v *ValkeyStore) Close() error {
	// The caller is responsible for managing the client lifecycle
	return nil
}
