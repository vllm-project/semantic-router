package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

// DefaultMaxRetries is the default number of retry attempts for transient errors
// DefaultRetryBaseDelay is the base delay for exponential backoff (in milliseconds)
const (
	DefaultMaxRetries     = 3
	DefaultRetryBaseDelay = 100
)

// MilvusStore provides memory retrieval from Milvus with similarity threshold filtering
type MilvusStore struct {
	client          client.Client
	collectionName  string
	config          config.MemoryConfig
	enabled         bool
	maxRetries      int
	retryBaseDelay  time.Duration
	embeddingConfig EmbeddingConfig // Unified embedding configuration
}

// MilvusStoreOptions contains configuration for creating a MilvusStore
//
//	Client is the Milvus client instance
//	CollectionName is the name of the Milvus collection
//	Config is the memory configuration
//	Enabled controls whether the store is active
//	EmbeddingConfig is the unified embedding configuration (optional, defaults to mmbert/768)
type MilvusStoreOptions struct {
	Client          client.Client
	CollectionName  string
	Config          config.MemoryConfig
	Enabled         bool
	EmbeddingConfig *EmbeddingConfig // Optional: if nil, derived from Config.Embedding
}

// NewMilvusStore creates a new MilvusStore instance
func NewMilvusStore(options MilvusStoreOptions) (*MilvusStore, error) {
	if !options.Enabled {
		logging.Debugf("MilvusStore: disabled, returning stub")
		return &MilvusStore{
			enabled: false,
		}, nil
	}

	if options.Client == nil {
		return nil, fmt.Errorf("milvus client is required")
	}

	if options.CollectionName == "" {
		return nil, fmt.Errorf("collection name is required")
	}

	// Use default config if not provided
	cfg := options.Config
	if cfg.EmbeddingModel == "" {
		cfg = DefaultMemoryConfig()
	}

	// Initialize embedding configuration
	var embeddingCfg EmbeddingConfig
	if options.EmbeddingConfig != nil {
		embeddingCfg = *options.EmbeddingConfig
	} else {
		embeddingCfg = EmbeddingConfig{Model: EmbeddingModelBERT}
	}

	store := &MilvusStore{
		client:          options.Client,
		collectionName:  options.CollectionName,
		config:          cfg,
		enabled:         options.Enabled,
		maxRetries:      DefaultMaxRetries,
		retryBaseDelay:  DefaultRetryBaseDelay * time.Millisecond,
		embeddingConfig: embeddingCfg,
	}

	// Auto-create collection if it doesn't exist
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := store.ensureCollection(ctx); err != nil {
		return nil, fmt.Errorf("failed to ensure collection exists: %w", err)
	}

	logging.Infof("MilvusStore: initialized with collection='%s', embedding_model='%s'",
		store.collectionName, store.embeddingConfig.Model)

	return store, nil
}

// ensureCollection checks if the collection exists and creates it if not
func (m *MilvusStore) ensureCollection(ctx context.Context) error {
	hasCollection, err := m.client.HasCollection(ctx, m.collectionName)
	if err != nil {
		return fmt.Errorf("failed to check collection existence: %w", err)
	}

	if hasCollection {
		logging.Debugf("MilvusStore: collection '%s' already exists", m.collectionName)
		// Load collection to make it queryable
		if loadErr := m.client.LoadCollection(ctx, m.collectionName, false); loadErr != nil {
			logging.Warnf("MilvusStore: failed to load collection: %v (may already be loaded)", loadErr)
		}
		return nil
	}

	logging.Infof("MilvusStore: creating collection '%s' with dimension %d", m.collectionName, m.config.Milvus.Dimension)

	// Define schema for agentic memory
	schema := &entity.Schema{
		CollectionName: m.collectionName,
		Description:    "Agentic Memory storage for cross-session context",
		AutoID:         false,
		Fields: []*entity.Field{
			{
				Name:       "id",
				DataType:   entity.FieldTypeVarChar,
				PrimaryKey: true,
				TypeParams: map[string]string{"max_length": "64"},
			},
			{
				Name:           "user_id",
				DataType:       entity.FieldTypeVarChar,
				TypeParams:     map[string]string{"max_length": "256"},
				IsPartitionKey: true, // Enables efficient per-user queries
			},
			{
				Name:       "project_id",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "256"},
			},
			{
				Name:       "memory_type",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "32"},
			},
			{
				Name:       "content",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "65535"},
			},
			{
				Name:       "source",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "256"},
			},
			{
				Name:       "metadata",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "65535"},
			},
			{
				Name:     "embedding",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					"dim": fmt.Sprintf("%d", m.config.Milvus.Dimension),
				},
			},
			{
				Name:     "created_at",
				DataType: entity.FieldTypeInt64,
			},
			{
				Name:     "updated_at",
				DataType: entity.FieldTypeInt64,
			},
			{
				Name:     "access_count",
				DataType: entity.FieldTypeInt64,
			},
			{
				Name:     "importance",
				DataType: entity.FieldTypeFloat,
			},
		},
	}

	// Create collection with partition key support
	// NumPartitions determines how partition key values are distributed (default: 16)
	numPartitions := int64(16) // Milvus default
	if m.config.Milvus.NumPartitions > 0 {
		numPartitions = int64(m.config.Milvus.NumPartitions)
	}
	logging.Infof("MilvusStore: creating collection with %d partitions (partition key: user_id)", numPartitions)

	if createErr := m.client.CreateCollection(ctx, schema, 1, client.WithPartitionNum(numPartitions)); createErr != nil {
		return fmt.Errorf("failed to create collection: %w", createErr)
	}

	// Create HNSW index for vector search
	index, err := entity.NewIndexHNSW(entity.COSINE, 16, 256) // M=16, efConstruction=256
	if err != nil {
		return fmt.Errorf("failed to create HNSW index: %w", err)
	}
	if err := m.client.CreateIndex(ctx, m.collectionName, "embedding", index, false); err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}

	// Load collection to make it queryable
	if err := m.client.LoadCollection(ctx, m.collectionName, false); err != nil {
		return fmt.Errorf("failed to load collection: %w", err)
	}

	logging.Infof("MilvusStore: collection '%s' created and loaded successfully", m.collectionName)
	return nil
}

// Retrieve searches for memories in Milvus with similarity threshold filtering
func (m *MilvusStore) Retrieve(ctx context.Context, opts RetrieveOptions) ([]*RetrieveResult, error) {
	startTime := time.Now()
	backend := "milvus"
	operation := "retrieve"
	status := "success"
	resultCount := 0

	// Defer metrics recording
	defer func() {
		duration := time.Since(startTime).Seconds()
		RecordMemoryRetrieval(backend, operation, status, opts.UserID, duration, resultCount)
	}()

	if !m.enabled {
		status = "error"
		return nil, fmt.Errorf("milvus store is not enabled")
	}

	// Apply defaults
	limit := opts.Limit
	if limit <= 0 {
		limit = m.config.DefaultRetrievalLimit
	}

	threshold := opts.Threshold
	if threshold <= 0 {
		threshold = m.config.DefaultSimilarityThreshold
	}

	if opts.Query == "" {
		status = "error"
		return nil, fmt.Errorf("query is required")
	}

	if opts.UserID == "" {
		status = "error"
		return nil, fmt.Errorf("user id is required")
	}

	logging.Debugf("MilvusStore.Retrieve: query='%s', user_id='%s', limit=%d, threshold=%.4f, hybrid=%v (mode=%s)",
		opts.Query, opts.UserID, limit, threshold, opts.HybridSearch, opts.HybridMode)

	// Generate embedding for the query
	embedding, err := GenerateEmbedding(opts.Query, m.embeddingConfig)
	if err != nil {
		status = "error"
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	logging.Debugf("MilvusStore.Retrieve: generated embedding with model=%s, dimension=%d",
		m.embeddingConfig.Model, len(embedding))

	// Build filter expression for user_id
	filterExpr := fmt.Sprintf("user_id == \"%s\"", opts.UserID)

	// Add memory type filter if specified
	if len(opts.Types) > 0 {
		typeFilter := "("
		for i, memType := range opts.Types {
			if i > 0 {
				typeFilter += " || "
			}
			typeFilter += fmt.Sprintf("memory_type == \"%s\"", string(memType))
		}
		typeFilter += ")"
		filterExpr = fmt.Sprintf("%s && %s", filterExpr, typeFilter)
	}

	logging.Debugf("MilvusStore.Retrieve: filter expression: %s", filterExpr)

	// Create search parameters
	// Using HNSW index with ef parameter (adjust based on your index configuration)
	searchParam, err := entity.NewIndexHNSWSearchParam(64)
	if err != nil {
		status = "error"
		return nil, fmt.Errorf("failed to create search parameters: %w", err)
	}

	// When hybrid search is enabled, expand the candidate pool so text-based
	// re-ranking has a richer set to work with.
	searchTopK := limit * 4
	if opts.HybridSearch {
		searchTopK = limit * 8
	}
	if searchTopK < 20 {
		searchTopK = 20
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
		status = "error"
		return nil, fmt.Errorf("milvus search failed after retries: %w", err)
	}

	if len(searchResult) == 0 || searchResult[0].ResultCount == 0 {
		logging.Debugf("MilvusStore.Retrieve: no results found")
		status = "miss"
		resultCount = 0
		return []*RetrieveResult{}, nil
	}

	// Parse Milvus columns into candidate structs.
	candidates := m.parseCandidates(searchResult[0], opts.UserID)

	logging.Debugf("MilvusStore.Retrieve: parsed %d candidates from Milvus", len(candidates))

	// Apply hybrid re-ranking when enabled.
	if opts.HybridSearch && len(candidates) > 1 {
		candidates = m.hybridRerank(candidates, opts)
		logging.Debugf("MilvusStore.Retrieve: hybrid re-ranked %d candidates (mode=%s)", len(candidates), opts.HybridMode)
	}

	// Apply threshold + limit.
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

	logging.Debugf("MilvusStore.Retrieve: returning %d results (filtered from %d candidates)",
		len(results), len(candidates))

	// Update access tracking in background (reinforcement: S += 1, t = 0).
	if len(results) > 0 {
		ids := make([]string, len(results))
		for i, r := range results {
			ids[i] = r.Memory.ID
		}
		go m.recordRetrievalBatch(ids)
	}

	logging.Debugf("MilvusStore.Retrieve: %d results found", len(results))

	resultCount = len(results)
	if resultCount > 0 {
		status = "hit"
	} else {
		status = "miss"
	}

	return results, nil
}

// parseCandidates extracts RetrieveResult entries from a Milvus SearchResult.
func (m *MilvusStore) parseCandidates(sr client.SearchResult, defaultUserID string) []*RetrieveResult {
	scores := sr.Scores
	fields := sr.Fields

	idIdx, contentIdx, typeIdx, metadataIdx := -1, -1, -1, -1
	for i, field := range fields {
		switch field.Name() {
		case "id":
			idIdx = i
		case "content":
			contentIdx = i
		case "memory_type":
			typeIdx = i
		case "metadata":
			metadataIdx = i
		}
	}

	results := make([]*RetrieveResult, 0, len(scores))
	for i := 0; i < len(scores); i++ {
		var id, content, memType string
		metadata := make(map[string]interface{})

		if idIdx >= 0 && idIdx < len(fields) {
			if col, ok := fields[idIdx].(*entity.ColumnVarChar); ok && col.Len() > i {
				if val, err := col.ValueByIdx(i); err == nil {
					id = val
				}
			}
		}
		if contentIdx >= 0 && contentIdx < len(fields) {
			if col, ok := fields[contentIdx].(*entity.ColumnVarChar); ok && col.Len() > i {
				if val, err := col.ValueByIdx(i); err == nil {
					content = val
				}
			}
		}
		if typeIdx >= 0 && typeIdx < len(fields) {
			if col, ok := fields[typeIdx].(*entity.ColumnVarChar); ok && col.Len() > i {
				if val, err := col.ValueByIdx(i); err == nil {
					memType = val
				}
			}
		}
		if metadataIdx >= 0 && metadataIdx < len(fields) {
			if col, ok := fields[metadataIdx].(*entity.ColumnVarChar); ok && col.Len() > i {
				if metadataVal, err := col.ValueByIdx(i); err == nil && metadataVal != "" {
					if err := json.Unmarshal([]byte(metadataVal), &metadata); err != nil {
						metadata["raw"] = metadataVal
					} else {
						metadata["_raw_source"] = metadataVal
					}
				}
			}
		}

		if id == "" || content == "" {
			continue
		}

		mem := &Memory{
			ID:      id,
			Content: content,
			Type:    MemoryType(memType),
		}
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

		results = append(results, &RetrieveResult{Memory: mem, Score: scores[i]})
	}
	return results
}

// hybridRerank applies BM25 + n-gram scoring on top of vector results
// using the shared scoring infrastructure from pkg/vectorstore.
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

// IsEnabled returns whether the store is enabled
func (m *MilvusStore) IsEnabled() bool {
	return m.enabled
}

// CheckConnection verifies the Milvus connection is healthy
func (m *MilvusStore) CheckConnection(ctx context.Context) error {
	if !m.enabled {
		return nil
	}

	if m.client == nil {
		return fmt.Errorf("milvus client is not initialized")
	}

	// Check if collection exists
	hasCollection, err := m.client.HasCollection(ctx, m.collectionName)
	if err != nil {
		return fmt.Errorf("failed to check collection existence: %w", err)
	}

	if !hasCollection {
		return fmt.Errorf("collection '%s' does not exist", m.collectionName)
	}

	return nil
}

// Close releases resources held by the store
func (m *MilvusStore) Close() error {
	// Note: We don't close the client here as it might be shared
	// The caller is responsible for managing the client lifecycle
	return nil
}

// Store saves a new memory to Milvus.
// Generates embedding for the content and inserts into the collection.
func (m *MilvusStore) Store(ctx context.Context, memory *Memory) error {
	startTime := time.Now()
	backend := "milvus"
	operation := "store"
	status := "success"

	// Defer metrics recording
	defer func() {
		duration := time.Since(startTime).Seconds()
		RecordMemoryStoreOperation(backend, operation, status, duration)
	}()

	if !m.enabled {
		status = "error"
		return fmt.Errorf("milvus store is not enabled")
	}

	if memory.ID == "" {
		status = "error"
		return fmt.Errorf("memory ID is required")
	}
	if memory.Content == "" {
		status = "error"
		return fmt.Errorf("memory content is required")
	}
	if memory.UserID == "" {
		status = "error"
		return fmt.Errorf("user ID is required")
	}

	logging.Debugf("MilvusStore.Store: id=%s, user=%s, type=%s, content_len=%d",
		memory.ID, memory.UserID, memory.Type, len(memory.Content))

	// Generate embedding for content if not already set
	var embedding []float32
	if len(memory.Embedding) > 0 {
		embedding = memory.Embedding
	} else {
		var err error
		embedding, err = GenerateEmbedding(memory.Content, m.embeddingConfig)
		if err != nil {
			status = "error"
			return fmt.Errorf("failed to generate embedding: %w", err)
		}
	}

	// Set timestamps; on first save LastAccessed = now (t=0 for retention score)
	now := time.Now()
	if memory.CreatedAt.IsZero() {
		memory.CreatedAt = now
	}
	memory.UpdatedAt = now
	if memory.LastAccessed.IsZero() {
		memory.LastAccessed = now
	}

	// Build metadata JSON (last_accessed and access_count used for retention scoring)
	metadata := map[string]interface{}{
		"user_id":       memory.UserID,
		"project_id":    memory.ProjectID,
		"source":        memory.Source,
		"importance":    memory.Importance,
		"access_count":  memory.AccessCount,
		"last_accessed": memory.LastAccessed.Unix(),
	}
	metadataJSON, err := json.Marshal(metadata)
	if err != nil {
		status = "error"
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	// Create columns for insert
	// Use defaults for optional fields if not provided (all fields required by schema)
	projectID := memory.ProjectID
	if projectID == "" {
		projectID = "default"
	}
	source := memory.Source
	if source == "" {
		source = "extraction" // Default source for extracted memories
	}

	idCol := entity.NewColumnVarChar("id", []string{memory.ID})
	contentCol := entity.NewColumnVarChar("content", []string{memory.Content})
	userIDCol := entity.NewColumnVarChar("user_id", []string{memory.UserID})
	projectIDCol := entity.NewColumnVarChar("project_id", []string{projectID})
	memTypeCol := entity.NewColumnVarChar("memory_type", []string{string(memory.Type)})
	sourceCol := entity.NewColumnVarChar("source", []string{source})
	metadataCol := entity.NewColumnVarChar("metadata", []string{string(metadataJSON)})
	embeddingCol := entity.NewColumnFloatVector("embedding", len(embedding), [][]float32{embedding})
	createdAtCol := entity.NewColumnInt64("created_at", []int64{memory.CreatedAt.Unix()})
	updatedAtCol := entity.NewColumnInt64("updated_at", []int64{memory.UpdatedAt.Unix()})
	accessCountCol := entity.NewColumnInt64("access_count", []int64{int64(memory.AccessCount)})
	importanceCol := entity.NewColumnFloat("importance", []float32{float32(memory.Importance)})

	// Insert with retry logic
	err = m.retryWithBackoff(ctx, func() error {
		_, insertErr := m.client.Insert(
			ctx,
			m.collectionName,
			"", // Default partition
			idCol,
			contentCol,
			userIDCol,
			projectIDCol,
			memTypeCol,
			sourceCol,
			metadataCol,
			embeddingCol,
			createdAtCol,
			updatedAtCol,
			accessCountCol,
			importanceCol,
		)
		return insertErr
	})
	if err != nil {
		status = "error"
		return fmt.Errorf("milvus insert failed: %w", err)
	}

	logging.Debugf("MilvusStore.Store: successfully stored memory id=%s", memory.ID)
	return nil
}

// upsert atomically replaces a row in Milvus by primary key.
// The memory must be fully populated (including Embedding, timestamps, etc.).
// Used by Update to avoid the delete+insert data-loss window.
func (m *MilvusStore) upsert(ctx context.Context, memory *Memory) error {
	// Build metadata JSON
	metadata := map[string]interface{}{
		"user_id":       memory.UserID,
		"project_id":    memory.ProjectID,
		"source":        memory.Source,
		"importance":    memory.Importance,
		"access_count":  memory.AccessCount,
		"last_accessed": memory.LastAccessed.Unix(),
	}
	metadataJSON, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	projectID := memory.ProjectID
	if projectID == "" {
		projectID = "default"
	}
	source := memory.Source
	if source == "" {
		source = "extraction"
	}

	embedding := memory.Embedding
	if len(embedding) == 0 {
		return fmt.Errorf("embedding is required for upsert")
	}

	idCol := entity.NewColumnVarChar("id", []string{memory.ID})
	contentCol := entity.NewColumnVarChar("content", []string{memory.Content})
	userIDCol := entity.NewColumnVarChar("user_id", []string{memory.UserID})
	projectIDCol := entity.NewColumnVarChar("project_id", []string{projectID})
	memTypeCol := entity.NewColumnVarChar("memory_type", []string{string(memory.Type)})
	sourceCol := entity.NewColumnVarChar("source", []string{source})
	metadataCol := entity.NewColumnVarChar("metadata", []string{string(metadataJSON)})
	embeddingCol := entity.NewColumnFloatVector("embedding", len(embedding), [][]float32{embedding})
	createdAtCol := entity.NewColumnInt64("created_at", []int64{memory.CreatedAt.Unix()})
	updatedAtCol := entity.NewColumnInt64("updated_at", []int64{memory.UpdatedAt.Unix()})
	accessCountCol := entity.NewColumnInt64("access_count", []int64{int64(memory.AccessCount)})
	importanceCol := entity.NewColumnFloat("importance", []float32{float32(memory.Importance)})

	err = m.retryWithBackoff(ctx, func() error {
		_, upsertErr := m.client.Upsert(
			ctx,
			m.collectionName,
			"",
			idCol,
			contentCol,
			userIDCol,
			projectIDCol,
			memTypeCol,
			sourceCol,
			metadataCol,
			embeddingCol,
			createdAtCol,
			updatedAtCol,
			accessCountCol,
			importanceCol,
		)
		return upsertErr
	})
	if err != nil {
		return fmt.Errorf("milvus upsert failed: %w", err)
	}

	logging.Debugf("MilvusStore.upsert: successfully upserted memory id=%s", memory.ID)
	return nil
}

// Get retrieves a memory by ID from Milvus.
func (m *MilvusStore) Get(ctx context.Context, id string) (*Memory, error) {
	if !m.enabled {
		return nil, fmt.Errorf("milvus store is not enabled")
	}

	if id == "" {
		return nil, fmt.Errorf("memory ID is required")
	}

	logging.Debugf("MilvusStore.Get: retrieving memory id=%s", id)

	// Query by ID (includes embedding so the caller can Upsert without re-generating it)
	filterExpr := fmt.Sprintf("id == \"%s\"", id)
	outputFields := []string{"id", "content", "user_id", "memory_type", "metadata", "created_at", "updated_at", "embedding"}

	var queryResult []entity.Column
	err := m.retryWithBackoff(ctx, func() error {
		var retryErr error
		queryResult, retryErr = m.client.Query(
			ctx,
			m.collectionName,
			[]string{}, // All partitions
			filterExpr,
			outputFields,
		)
		return retryErr
	})
	if err != nil {
		return nil, fmt.Errorf("milvus query failed: %w", err)
	}

	if len(queryResult) == 0 {
		return nil, fmt.Errorf("memory not found: %s", id)
	}

	// Check if any column has data
	hasData := false
	for _, col := range queryResult {
		if col.Len() > 0 {
			hasData = true
			break
		}
	}
	if !hasData {
		return nil, fmt.Errorf("memory not found: %s", id)
	}

	memory := &Memory{}

	// Extract fields from columns
	for _, col := range queryResult {
		switch col.Name() {
		case "id":
			if c, ok := col.(*entity.ColumnVarChar); ok && c.Len() > 0 {
				val, _ := c.ValueByIdx(0)
				memory.ID = val
			}
		case "content":
			if c, ok := col.(*entity.ColumnVarChar); ok && c.Len() > 0 {
				val, _ := c.ValueByIdx(0)
				memory.Content = val
			}
		case "user_id":
			if c, ok := col.(*entity.ColumnVarChar); ok && c.Len() > 0 {
				val, _ := c.ValueByIdx(0)
				memory.UserID = val
			}
		case "memory_type":
			if c, ok := col.(*entity.ColumnVarChar); ok && c.Len() > 0 {
				val, _ := c.ValueByIdx(0)
				memory.Type = MemoryType(val)
			}
		case "metadata":
			if c, ok := col.(*entity.ColumnVarChar); ok && c.Len() > 0 {
				val, _ := c.ValueByIdx(0)
				if val != "" {
					var metadata map[string]interface{}
					if err := json.Unmarshal([]byte(val), &metadata); err == nil {
						if projectID, ok := metadata["project_id"].(string); ok {
							memory.ProjectID = projectID
						}
						if source, ok := metadata["source"].(string); ok {
							memory.Source = source
						}
						if importance, ok := metadata["importance"].(float64); ok {
							memory.Importance = float32(importance)
						}
						if accessCount, ok := metadata["access_count"].(float64); ok {
							memory.AccessCount = int(accessCount)
						}
						if lastAccessed, ok := metadata["last_accessed"].(float64); ok {
							memory.LastAccessed = time.Unix(int64(lastAccessed), 0)
						}
					}
				}
			}
		case "created_at":
			if c, ok := col.(*entity.ColumnInt64); ok && c.Len() > 0 {
				val, _ := c.ValueByIdx(0)
				memory.CreatedAt = time.Unix(val, 0)
			}
		case "updated_at":
			if c, ok := col.(*entity.ColumnInt64); ok && c.Len() > 0 {
				val, _ := c.ValueByIdx(0)
				memory.UpdatedAt = time.Unix(val, 0)
			}
		case "embedding":
			if c, ok := col.(*entity.ColumnFloatVector); ok && c.Len() > 0 {
				memory.Embedding = c.Data()[0]
			}
		}
	}

	if memory.ID == "" {
		return nil, fmt.Errorf("memory not found: %s", id)
	}

	logging.Debugf("MilvusStore.Get: found memory id=%s, user_id=%s", memory.ID, memory.UserID)
	return memory, nil
}

// List returns memories matching the filter criteria with pagination.
// Queries Milvus with scalar filtering (no vector search) and returns paginated results.
func (m *MilvusStore) List(ctx context.Context, opts ListOptions) (*ListResult, error) {
	if !m.enabled {
		return nil, fmt.Errorf("milvus store is not enabled")
	}

	if opts.UserID == "" {
		return nil, fmt.Errorf("user ID is required for listing memories")
	}

	logging.Debugf("MilvusStore.List: user_id=%s, types=%v, limit=%d",
		opts.UserID, opts.Types, opts.Limit)

	// Build filter expression
	filterExpr := fmt.Sprintf("user_id == \"%s\"", opts.UserID)

	if len(opts.Types) > 0 {
		typeFilter := "("
		for i, memType := range opts.Types {
			if i > 0 {
				typeFilter += " || "
			}
			typeFilter += fmt.Sprintf("memory_type == \"%s\"", string(memType))
		}
		typeFilter += ")"
		filterExpr = fmt.Sprintf("%s && %s", filterExpr, typeFilter)
	}

	outputFields := []string{"id", "content", "user_id", "memory_type", "metadata", "created_at", "updated_at"}

	// Query all matching records to get total count and apply pagination
	var queryResult []entity.Column
	err := m.retryWithBackoff(ctx, func() error {
		var retryErr error
		queryResult, retryErr = m.client.Query(
			ctx,
			m.collectionName,
			[]string{}, // All partitions
			filterExpr,
			outputFields,
		)
		return retryErr
	})
	if err != nil {
		return nil, fmt.Errorf("milvus query failed: %w", err)
	}

	// Parse results into Memory objects (no project_id filtering — field is not populated)
	memories := m.parseListResults(queryResult, "")

	// Sort by created_at descending for deterministic results.
	// Milvus Query does not support server-side ORDER BY, so sorting is done client-side.
	sort.Slice(memories, func(i, j int) bool {
		return memories[i].CreatedAt.After(memories[j].CreatedAt)
	})

	total := len(memories)

	// Apply limit
	limit := opts.Limit
	if limit <= 0 {
		limit = 20
	}
	if limit > 100 {
		limit = 100
	}

	if limit < len(memories) {
		memories = memories[:limit]
	}

	logging.Debugf("MilvusStore.List: found %d total, returning %d (limit=%d)",
		total, len(memories), limit)

	return &ListResult{
		Memories: memories,
		Total:    total,
		Limit:    limit,
	}, nil
}

// parseListResults converts Milvus query columns into Memory objects for List operations.
// Optionally filters by project_id (stored in metadata JSON).
func (m *MilvusStore) parseListResults(queryResult []entity.Column, projectIDFilter string) []*Memory {
	if len(queryResult) == 0 {
		return []*Memory{}
	}

	// Determine the number of rows from the first column
	rowCount := 0
	for _, col := range queryResult {
		if col.Len() > rowCount {
			rowCount = col.Len()
		}
	}

	// Build column maps for fast lookup
	var idCol *entity.ColumnVarChar
	var contentCol *entity.ColumnVarChar
	var userIDCol *entity.ColumnVarChar
	var typeCol *entity.ColumnVarChar
	var metadataCol *entity.ColumnVarChar
	var createdAtCol *entity.ColumnInt64
	var updatedAtCol *entity.ColumnInt64

	for _, col := range queryResult {
		switch col.Name() {
		case "id":
			idCol, _ = col.(*entity.ColumnVarChar)
		case "content":
			contentCol, _ = col.(*entity.ColumnVarChar)
		case "user_id":
			userIDCol, _ = col.(*entity.ColumnVarChar)
		case "memory_type":
			typeCol, _ = col.(*entity.ColumnVarChar)
		case "metadata":
			metadataCol, _ = col.(*entity.ColumnVarChar)
		case "created_at":
			createdAtCol, _ = col.(*entity.ColumnInt64)
		case "updated_at":
			updatedAtCol, _ = col.(*entity.ColumnInt64)
		}
	}

	memories := make([]*Memory, 0, rowCount)
	for i := 0; i < rowCount; i++ {
		mem := &Memory{}

		if idCol != nil && idCol.Len() > i {
			mem.ID, _ = idCol.ValueByIdx(i)
		}
		if contentCol != nil && contentCol.Len() > i {
			mem.Content, _ = contentCol.ValueByIdx(i)
		}
		if userIDCol != nil && userIDCol.Len() > i {
			mem.UserID, _ = userIDCol.ValueByIdx(i)
		}
		if typeCol != nil && typeCol.Len() > i {
			val, _ := typeCol.ValueByIdx(i)
			mem.Type = MemoryType(val)
		}
		if metadataCol != nil && metadataCol.Len() > i {
			val, _ := metadataCol.ValueByIdx(i)
			if val != "" {
				var metadata map[string]interface{}
				if err := json.Unmarshal([]byte(val), &metadata); err == nil {
					if pid, ok := metadata["project_id"].(string); ok {
						mem.ProjectID = pid
					}
					if src, ok := metadata["source"].(string); ok {
						mem.Source = src
					}
					if imp, ok := metadata["importance"].(float64); ok {
						mem.Importance = float32(imp)
					}
					if ac, ok := metadata["access_count"].(float64); ok {
						mem.AccessCount = int(ac)
					}
				}
			}
		}
		if createdAtCol != nil && createdAtCol.Len() > i {
			val, _ := createdAtCol.ValueByIdx(i)
			mem.CreatedAt = time.Unix(val, 0)
		}
		if updatedAtCol != nil && updatedAtCol.Len() > i {
			val, _ := updatedAtCol.ValueByIdx(i)
			mem.UpdatedAt = time.Unix(val, 0)
		}

		// Skip if memory ID is empty (corrupt/missing data)
		if mem.ID == "" {
			continue
		}

		// Apply project_id filter (stored in metadata JSON, not a direct Milvus column)
		if projectIDFilter != "" && mem.ProjectID != projectIDFilter {
			continue
		}

		memories = append(memories, mem)
	}

	return memories
}

// Update modifies an existing memory in Milvus using Upsert (atomic replace by primary key).
// The caller must provide a fully populated Memory (including Embedding); Update preserves CreatedAt
// from the existing row and sets UpdatedAt to now.
func (m *MilvusStore) Update(ctx context.Context, id string, memory *Memory) error {
	startTime := time.Now()
	backend := "milvus"
	operation := "update"
	status := "success"

	// Defer metrics recording
	defer func() {
		duration := time.Since(startTime).Seconds()
		RecordMemoryStoreOperation(backend, operation, status, duration)
	}()

	if !m.enabled {
		status = "error"
		return fmt.Errorf("milvus store is not enabled")
	}

	if id == "" {
		status = "error"
		return fmt.Errorf("memory ID is required")
	}

	logging.Debugf("MilvusStore.Update: upserting memory id=%s", id)

	memory.ID = id
	memory.UpdatedAt = time.Now()

	// If CreatedAt or Embedding are missing, fetch from the existing row so we don't lose data
	if memory.CreatedAt.IsZero() || len(memory.Embedding) == 0 {
		existing, err := m.Get(ctx, id)
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

	err := m.upsert(ctx, memory)
	if err != nil {
		status = "error"
		return err
	}
	return nil
}

// recordRetrievalBatch updates LastAccessed and AccessCount for each retrieved memory in the background.
// Uses a detached context with a timeout so request cancellation does not abort the writes.
func (m *MilvusStore) recordRetrievalBatch(ids []string) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	for _, id := range ids {
		if err := m.recordRetrieval(ctx, id); err != nil {
			logging.Warnf("MilvusStore.recordRetrievalBatch: id=%s: %v", id, err)
		}
	}
}

// recordRetrieval updates LastAccessed and AccessCount for a single memory (reinforcement: S += 1, t = 0).
func (m *MilvusStore) recordRetrieval(ctx context.Context, id string) error {
	existing, err := m.Get(ctx, id)
	if err != nil {
		return err
	}
	existing.AccessCount++
	existing.LastAccessed = time.Now()
	existing.UpdatedAt = existing.LastAccessed
	return m.Update(ctx, id, existing)
}

// Forget deletes a memory by ID from Milvus.
func (m *MilvusStore) Forget(ctx context.Context, id string) error {
	startTime := time.Now()
	backend := "milvus"
	operation := "forget"
	status := "success"

	// Defer metrics recording
	defer func() {
		duration := time.Since(startTime).Seconds()
		RecordMemoryStoreOperation(backend, operation, status, duration)
	}()

	if !m.enabled {
		status = "error"
		return fmt.Errorf("milvus store is not enabled")
	}

	if id == "" {
		status = "error"
		return fmt.Errorf("memory ID is required")
	}

	logging.Debugf("MilvusStore.Forget: deleting memory id=%s", id)

	// Build delete expression
	// NOTE: IDs are system-generated UUIDs, so injection risk is minimal.
	// For production with user-controlled IDs, consider escaping quotes or using parameterized queries.
	deleteExpr := fmt.Sprintf("id == \"%s\"", id)

	err := m.retryWithBackoff(ctx, func() error {
		return m.client.Delete(
			ctx,
			m.collectionName,
			"", // Default partition
			deleteExpr,
		)
	})
	if err != nil {
		status = "error"
		return fmt.Errorf("milvus delete failed: %w", err)
	}

	logging.Debugf("MilvusStore.Forget: successfully deleted memory id=%s", id)
	return nil
}

// ForgetByScope deletes all memories matching the scope from Milvus.
// Scope includes UserID (required), ProjectID (optional), Types (optional).
func (m *MilvusStore) ForgetByScope(ctx context.Context, scope MemoryScope) error {
	startTime := time.Now()
	backend := "milvus"
	operation := "forget_by_scope"
	status := "success"

	// Defer metrics recording
	defer func() {
		duration := time.Since(startTime).Seconds()
		RecordMemoryStoreOperation(backend, operation, status, duration)
	}()

	if !m.enabled {
		status = "error"
		return fmt.Errorf("milvus store is not enabled")
	}

	if scope.UserID == "" {
		status = "error"
		return fmt.Errorf("user ID is required for scope deletion")
	}

	logging.Debugf("MilvusStore.ForgetByScope: deleting memories for user_id=%s, project_id=%s, types=%v",
		scope.UserID, scope.ProjectID, scope.Types)

	// Build filter expression
	filterExpr := fmt.Sprintf("user_id == \"%s\"", scope.UserID)

	// Add project filter if specified
	if scope.ProjectID != "" {
		// Note: project_id is in metadata JSON, so we need to query first then delete by ID
		// For simplicity, we'll query matching IDs first, then delete them
		err := m.forgetByScopeWithQuery(ctx, scope)
		if err != nil {
			status = "error"
		}
		return err
	}

	// Add type filter if specified
	if len(scope.Types) > 0 {
		typeFilter := "("
		for i, memType := range scope.Types {
			if i > 0 {
				typeFilter += " || "
			}
			typeFilter += fmt.Sprintf("memory_type == \"%s\"", string(memType))
		}
		typeFilter += ")"
		filterExpr = fmt.Sprintf("%s && %s", filterExpr, typeFilter)
	}

	logging.Debugf("MilvusStore.ForgetByScope: delete expression: %s", filterExpr)

	err := m.retryWithBackoff(ctx, func() error {
		return m.client.Delete(
			ctx,
			m.collectionName,
			"", // Default partition
			filterExpr,
		)
	})
	if err != nil {
		status = "error"
		return fmt.Errorf("milvus delete by scope failed: %w", err)
	}

	logging.Debugf("MilvusStore.ForgetByScope: successfully deleted memories for user_id=%s", scope.UserID)
	return nil
}

// forgetByScopeWithQuery handles complex scope deletion that requires querying first.
// Used when project_id filter is specified (since it's in metadata JSON).
func (m *MilvusStore) forgetByScopeWithQuery(ctx context.Context, scope MemoryScope) error {
	// Query all memories for the user
	filterExpr := fmt.Sprintf("user_id == \"%s\"", scope.UserID)

	// Add type filter if specified
	if len(scope.Types) > 0 {
		typeFilter := "("
		for i, memType := range scope.Types {
			if i > 0 {
				typeFilter += " || "
			}
			typeFilter += fmt.Sprintf("memory_type == \"%s\"", string(memType))
		}
		typeFilter += ")"
		filterExpr = fmt.Sprintf("%s && %s", filterExpr, typeFilter)
	}

	outputFields := []string{"id", "metadata"}

	var queryResult []entity.Column
	err := m.retryWithBackoff(ctx, func() error {
		var retryErr error
		queryResult, retryErr = m.client.Query(
			ctx,
			m.collectionName,
			[]string{},
			filterExpr,
			outputFields,
		)
		return retryErr
	})
	if err != nil {
		return fmt.Errorf("milvus query failed: %w", err)
	}

	// Collect IDs to delete
	var idsToDelete []string

	// Find ID and metadata columns
	var idCol, metadataCol *entity.ColumnVarChar
	for _, col := range queryResult {
		switch col.Name() {
		case "id":
			if c, ok := col.(*entity.ColumnVarChar); ok {
				idCol = c
			}
		case "metadata":
			if c, ok := col.(*entity.ColumnVarChar); ok {
				metadataCol = c
			}
		}
	}

	if idCol == nil {
		logging.Debugf("MilvusStore.ForgetByScope: no IDs found")
		return nil
	}

	// Iterate through all IDs and check project_id in metadata if needed
	for i := 0; i < idCol.Len(); i++ {
		memID, _ := idCol.ValueByIdx(i)

		if scope.ProjectID != "" && metadataCol != nil && metadataCol.Len() > i {
			metadataStr, _ := metadataCol.ValueByIdx(i)
			var metadata map[string]interface{}
			if err := json.Unmarshal([]byte(metadataStr), &metadata); err == nil {
				if projectID, ok := metadata["project_id"].(string); ok {
					if projectID == scope.ProjectID {
						idsToDelete = append(idsToDelete, memID)
					}
				}
			}
		} else if scope.ProjectID == "" {
			idsToDelete = append(idsToDelete, memID)
		}
	}

	// Delete each matching memory
	// NOTE: Deletes one-by-one for simplicity. For production at scale,
	// consider batch deletion using "id in [...]" expression for efficiency.
	for _, memID := range idsToDelete {
		if err := m.Forget(ctx, memID); err != nil {
			logging.Warnf("MilvusStore.ForgetByScope: failed to delete memory id=%s: %v", memID, err)
		}
	}

	logging.Debugf("MilvusStore.ForgetByScope: deleted %d memories", len(idsToDelete))
	return nil
}

// isTransientError checks if an error is transient and should be retried
func isTransientError(err error) bool {
	if err == nil {
		return false
	}

	errStr := strings.ToLower(err.Error())

	// Check for common transient error patterns
	transientPatterns := []string{
		"connection",
		"timeout",
		"deadline exceeded",
		"context deadline exceeded",
		"unavailable",
		"temporary",
		"retry",
		"rate limit",
		"too many requests",
		"server error",
		"internal error",
		"service unavailable",
		"network",
		"broken pipe",
		"connection reset",
		"no connection",
		"connection refused",
	}

	for _, pattern := range transientPatterns {
		if strings.Contains(errStr, pattern) {
			return true
		}
	}

	return false
}

// retryWithBackoff retries an operation with exponential backoff for transient errors
func (m *MilvusStore) retryWithBackoff(ctx context.Context, operation func() error) error {
	var lastErr error

	for attempt := 0; attempt < m.maxRetries; attempt++ {
		lastErr = operation()

		// If no error or non-transient error, return immediately
		if lastErr == nil || !isTransientError(lastErr) {
			return lastErr
		}

		// If this is the last attempt, return the error
		if attempt == m.maxRetries-1 {
			logging.Warnf("MilvusStore: operation failed after %d retries: %v", m.maxRetries, lastErr)
			return lastErr
		}

		// Calculate exponential backoff delay
		// Cap the exponent to avoid overflow (max 30 for safety)
		exponent := attempt
		if exponent < 0 {
			exponent = 0
		} else if exponent > 30 {
			exponent = 30
		}
		delay := m.retryBaseDelay * time.Duration(1<<exponent) // 2^attempt * baseDelay

		logging.Debugf("MilvusStore: transient error on attempt %d/%d, retrying in %v: %v",
			attempt+1, m.maxRetries, delay, lastErr)

		// Wait with context cancellation support
		select {
		case <-ctx.Done():
			return fmt.Errorf("context cancelled during retry: %w", ctx.Err())
		case <-time.After(delay):
			// Continue to next retry
		}
	}

	return lastErr
}
