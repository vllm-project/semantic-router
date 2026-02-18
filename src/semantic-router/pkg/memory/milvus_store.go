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
	if !m.enabled {
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
		return nil, fmt.Errorf("query is required")
	}

	if opts.UserID == "" {
		return nil, fmt.Errorf("user id is required")
	}

	// TODO: Remove demo logging after POC
	logging.Infof("╔══════════════════════════════════════════════════════════════════╗")
	logging.Infof("║                 MEMORY RETRIEVE (MILVUS)                         ║")
	logging.Infof("╠══════════════════════════════════════════════════════════════════╣")
	logging.Infof("║ Query:     %s", truncateForLog(opts.Query, 100))
	logging.Infof("║ UserID:    %s", opts.UserID)
	logging.Infof("║ Limit:     %d", limit)
	logging.Infof("║ Threshold: %.4f", threshold)
	logging.Infof("╚══════════════════════════════════════════════════════════════════╝")
	logging.Debugf("MilvusStore.Retrieve: query='%s', user_id='%s', limit=%d, threshold=%.4f",
		opts.Query, opts.UserID, limit, threshold)

	// Generate embedding for the query
	embedding, err := GenerateEmbedding(opts.Query, m.embeddingConfig)
	if err != nil {
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
		return nil, fmt.Errorf("failed to create search parameters: %w", err)
	}

	// Perform vector search in Milvus with retry logic
	// We search for top-k results, then filter by threshold
	searchTopK := limit * 4 // Search for more results to account for threshold filtering
	if searchTopK < 20 {
		searchTopK = 20 // Minimum search size
	}

	var searchResult []client.SearchResult
	err = m.retryWithBackoff(ctx, func() error {
		var retryErr error
		searchResult, retryErr = m.client.Search(
			ctx,
			m.collectionName,
			[]string{}, // Empty partitions means search all
			filterExpr,
			[]string{"id", "content", "memory_type", "metadata"},
			[]entity.Vector{entity.FloatVector(embedding)},
			"embedding",   // Vector field name
			entity.COSINE, // Metric type
			searchTopK,
			searchParam,
		)
		return retryErr
	})
	if err != nil {
		return nil, fmt.Errorf("milvus search failed after retries: %w", err)
	}

	if len(searchResult) == 0 || searchResult[0].ResultCount == 0 {
		logging.Debugf("MilvusStore.Retrieve: no results found")
		return []*RetrieveResult{}, nil
	}

	// Extract results and filter by threshold
	results := make([]*RetrieveResult, 0, limit)
	scores := searchResult[0].Scores
	fields := searchResult[0].Fields

	// Find field indices
	idIdx, contentIdx, typeIdx, metadataIdx := -1, -1, -1, -1
	for i, field := range fields {
		fieldName := field.Name()
		switch fieldName {
		case "id":
			idIdx = i
		case "content":
			contentIdx = i
		case "memory_type":
			typeIdx = i
		case "metadata":
			metadataIdx = i
		}
		logging.Debugf("MilvusStore.Retrieve: field[%d] name='%s'", i, fieldName)
	}
	logging.Debugf("MilvusStore.Retrieve: field indices - id=%d, content=%d, type=%d, metadata=%d",
		idIdx, contentIdx, typeIdx, metadataIdx)

	// Process results and filter by threshold
	for i := 0; i < len(scores) && len(results) < limit; i++ {
		// Filter by similarity threshold
		score := scores[i]
		if score < threshold {
			logging.Debugf("MilvusStore.Retrieve: skipping result %d with score %.4f < threshold %.4f",
				i, score, threshold)
			continue
		}

		// Extract fields to build MemoryHit
		var id, content, memType string
		metadata := make(map[string]interface{})

		// Extract ID
		if idIdx >= 0 && idIdx < len(fields) {
			if col, ok := fields[idIdx].(*entity.ColumnVarChar); ok && col.Len() > i {
				if val, err := col.ValueByIdx(i); err == nil {
					id = val
				}
			}
		}

		// Extract content
		if contentIdx >= 0 && contentIdx < len(fields) {
			if col, ok := fields[contentIdx].(*entity.ColumnVarChar); ok && col.Len() > i {
				if val, err := col.ValueByIdx(i); err == nil {
					content = val
				}
			}
		}

		// Extract memory type
		if typeIdx >= 0 && typeIdx < len(fields) {
			if col, ok := fields[typeIdx].(*entity.ColumnVarChar); ok && col.Len() > i {
				if val, err := col.ValueByIdx(i); err == nil {
					memType = val
				}
			}
		}

		// Extract metadata (if available as JSON string)
		if metadataIdx >= 0 && metadataIdx < len(fields) {
			if col, ok := fields[metadataIdx].(*entity.ColumnVarChar); ok && col.Len() > i {
				if metadataVal, err := col.ValueByIdx(i); err == nil && metadataVal != "" {
					// Inflate JSON string into the map for downstream code accessibility
					if err := json.Unmarshal([]byte(metadataVal), &metadata); err != nil {
						// Fallback if JSON is malformed
						metadata["raw"] = metadataVal
					} else {
						// Reference for debugging/audit
						metadata["_raw_source"] = metadataVal
					}
				}
			}
		}

		// Only add if we have at least ID and content
		if id == "" || content == "" {
			continue
		}

		// Build MemoryHit as intermediate structure
		hit := MemoryHit{
			ID:         id,
			Content:    content,
			Type:       MemoryType(memType),
			Similarity: score,
			Metadata:   metadata,
		}

		// Convert MemoryHit to Memory object
		memory := &Memory{
			ID:      hit.ID,
			Content: hit.Content,
			Type:    hit.Type,
		}

		// Extract user_id from metadata if available
		if userID, ok := hit.Metadata["user_id"].(string); ok {
			memory.UserID = userID
		} else if opts.UserID != "" {
			memory.UserID = opts.UserID
		}

		// Extract project_id from metadata if available
		if projectID, ok := hit.Metadata["project_id"].(string); ok {
			memory.ProjectID = projectID
		}

		// Extract source from metadata if available
		if source, ok := hit.Metadata["source"].(string); ok {
			memory.Source = source
		}

		// Extract importance from metadata if available
		// Handle both float64 (from JSON) and string (like "high" - skip non-numeric)
		if importance, ok := hit.Metadata["importance"].(float64); ok {
			memory.Importance = float32(importance)
		} else if importance, ok := hit.Metadata["importance"].(float32); ok {
			memory.Importance = importance
		}
		// Access tracking for retention scoring
		if accessCount, ok := hit.Metadata["access_count"].(float64); ok {
			memory.AccessCount = int(accessCount)
		}
		if lastAccessed, ok := hit.Metadata["last_accessed"].(float64); ok {
			memory.LastAccessed = time.Unix(int64(lastAccessed), 0)
		}

		// Create RetrieveResult with Memory and Score
		result := &RetrieveResult{
			Memory: memory,
			Score:  hit.Similarity,
		}

		results = append(results, result)
	}

	logging.Debugf("MilvusStore.Retrieve: returning %d results (filtered from %d candidates)",
		len(results), len(scores))

	// Update access tracking in background (reinforcement: S += 1, t = 0).
	// Always active when memory is enabled; runs async so Retrieve latency is unaffected.
	if len(results) > 0 {
		ids := make([]string, len(results))
		for i, r := range results {
			ids[i] = r.Memory.ID
		}
		go m.recordRetrievalBatch(ids)
	}

	// TODO: Remove demo logging after POC
	logging.Infof("╔══════════════════════════════════════════════════════════════════╗")
	logging.Infof("║ MEMORY RETRIEVE RESULTS: %d memories found                       ║", len(results))
	logging.Infof("╚══════════════════════════════════════════════════════════════════╝")
	for i, r := range results {
		logging.Infof("  %d. [%.3f] %s: %s", i+1, r.Score, r.Memory.Type, r.Memory.Content) // Full content for demo
	}

	return results, nil
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
	if !m.enabled {
		return fmt.Errorf("milvus store is not enabled")
	}

	if memory.ID == "" {
		return fmt.Errorf("memory ID is required")
	}
	if memory.Content == "" {
		return fmt.Errorf("memory content is required")
	}
	if memory.UserID == "" {
		return fmt.Errorf("user ID is required")
	}

	// TODO: Remove demo logs after POC
	logging.Infof("╔══════════════════════════════════════════════════════════════════╗")
	logging.Infof("║                    MEMORY STORE (MILVUS)                         ║")
	logging.Infof("╠══════════════════════════════════════════════════════════════════╣")
	logging.Infof("║ ID:      %s", memory.ID)
	logging.Infof("║ User:    %s", memory.UserID)
	logging.Infof("║ Type:    %s", memory.Type)
	logging.Infof("║ Content: %s", memory.Content) // Full content for demo
	logging.Infof("╚══════════════════════════════════════════════════════════════════╝")

	// Generate embedding for content if not already set
	var embedding []float32
	if len(memory.Embedding) > 0 {
		embedding = memory.Embedding
	} else {
		var err error
		embedding, err = GenerateEmbedding(memory.Content, m.embeddingConfig)
		if err != nil {
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
	if !m.enabled {
		return fmt.Errorf("milvus store is not enabled")
	}

	if id == "" {
		return fmt.Errorf("memory ID is required")
	}

	logging.Debugf("MilvusStore.Update: upserting memory id=%s", id)

	memory.ID = id
	memory.UpdatedAt = time.Now()

	// If CreatedAt or Embedding are missing, fetch from the existing row so we don't lose data
	if memory.CreatedAt.IsZero() || len(memory.Embedding) == 0 {
		existing, err := m.Get(ctx, id)
		if err != nil {
			return fmt.Errorf("memory not found: %s", id)
		}
		if memory.CreatedAt.IsZero() {
			memory.CreatedAt = existing.CreatedAt
		}
		if len(memory.Embedding) == 0 {
			memory.Embedding = existing.Embedding
		}
	}

	return m.upsert(ctx, memory)
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
	if !m.enabled {
		return fmt.Errorf("milvus store is not enabled")
	}

	if id == "" {
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
		return fmt.Errorf("milvus delete failed: %w", err)
	}

	logging.Debugf("MilvusStore.Forget: successfully deleted memory id=%s", id)
	return nil
}

// ForgetByScope deletes all memories matching the scope from Milvus.
// Scope includes UserID (required), ProjectID (optional), Types (optional).
func (m *MilvusStore) ForgetByScope(ctx context.Context, scope MemoryScope) error {
	if !m.enabled {
		return fmt.Errorf("milvus store is not enabled")
	}

	if scope.UserID == "" {
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
		return m.forgetByScopeWithQuery(ctx, scope)
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

// MemoryPruneEntry holds minimal fields needed to compute retention score for pruning.
type MemoryPruneEntry struct {
	ID           string
	LastAccessed time.Time
	AccessCount  int
}

// ListForPrune returns all memories for a user with id, last_accessed, access_count for pruning.
func (m *MilvusStore) ListForPrune(ctx context.Context, userID string) ([]MemoryPruneEntry, error) {
	if !m.enabled {
		return nil, fmt.Errorf("milvus store is not enabled")
	}
	if userID == "" {
		return nil, fmt.Errorf("user ID is required")
	}
	filterExpr := fmt.Sprintf("user_id == \"%s\"", userID)
	outputFields := []string{"id", "metadata", "created_at"}

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
		return nil, fmt.Errorf("milvus query failed: %w", err)
	}

	var idCol, metadataCol *entity.ColumnVarChar
	var createdAtCol *entity.ColumnInt64
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
		case "created_at":
			if c, ok := col.(*entity.ColumnInt64); ok {
				createdAtCol = c
			}
		}
	}
	if idCol == nil {
		return []MemoryPruneEntry{}, nil
	}

	now := time.Now()
	var entries []MemoryPruneEntry
	for i := 0; i < idCol.Len(); i++ {
		id, _ := idCol.ValueByIdx(i)
		entry := MemoryPruneEntry{ID: id}
		if metadataCol != nil && metadataCol.Len() > i {
			metadataStr, _ := metadataCol.ValueByIdx(i)
			var meta map[string]interface{}
			if json.Unmarshal([]byte(metadataStr), &meta) == nil {
				if la, ok := meta["last_accessed"].(float64); ok {
					entry.LastAccessed = time.Unix(int64(la), 0)
				}
				if ac, ok := meta["access_count"].(float64); ok {
					entry.AccessCount = int(ac)
				}
			}
		}
		// Pre-existing memories stored before access tracking may have zero LastAccessed.
		// Fall back to created_at so they don't get R ≈ 0 and get pruned immediately.
		if entry.LastAccessed.IsZero() {
			if createdAtCol != nil && createdAtCol.Len() > i {
				val, _ := createdAtCol.ValueByIdx(i)
				if val > 0 {
					entry.LastAccessed = time.Unix(val, 0)
				}
			}
			// If still zero (no created_at either), treat as "just now" to protect the memory
			if entry.LastAccessed.IsZero() {
				entry.LastAccessed = now
			}
		}
		entries = append(entries, entry)
	}
	return entries, nil
}

// PruneUser deletes memories for userID that have R < PruneThreshold, then if over MaxMemoriesPerUser
// deletes lowest-R memories until at cap. No-op if the store is disabled.
func (m *MilvusStore) PruneUser(ctx context.Context, userID string) (deleted int, err error) {
	if !m.enabled {
		return 0, nil
	}
	cfg := m.config.QualityScoring

	initialStrength := cfg.InitialStrengthDays
	if initialStrength <= 0 {
		initialStrength = DefaultInitialStrengthDays
	}
	delta := cfg.PruneThreshold
	if delta <= 0 {
		delta = DefaultPruneThreshold
	}
	maxPerUser := cfg.MaxMemoriesPerUser

	entries, err := m.ListForPrune(ctx, userID)
	if err != nil {
		return 0, err
	}

	toDelete := PruneCandidates(entries, time.Now(), initialStrength, delta, maxPerUser)
	for _, id := range toDelete {
		if err := m.Forget(ctx, id); err != nil {
			logging.Warnf("MilvusStore.PruneUser: Forget id=%s: %v", id, err)
			continue
		}
		deleted++
	}
	logging.Debugf("MilvusStore.PruneUser: user_id=%s deleted %d memories", userID, deleted)
	return deleted, nil
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
