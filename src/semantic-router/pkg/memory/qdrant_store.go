package memory

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/qdrant/go-client/qdrant"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type QdrantStore struct {
	client          *qdrant.Client
	collectionName  string
	config          config.MemoryConfig
	qdrantConfig    *config.MemoryQdrantConfig
	enabled         bool
	embeddingConfig EmbeddingConfig
}

type QdrantStoreOptions struct {
	Client          *qdrant.Client
	Config          config.MemoryConfig
	QdrantConfig    *config.MemoryQdrantConfig
	Enabled         bool
	EmbeddingConfig *EmbeddingConfig
}

func NewQdrantStore(opts QdrantStoreOptions) (*QdrantStore, error) {
	if !opts.Enabled {
		return &QdrantStore{enabled: false}, nil
	}
	if opts.Client == nil {
		return nil, fmt.Errorf("qdrant client is required")
	}
	if opts.QdrantConfig == nil {
		return nil, fmt.Errorf("qdrant config is required")
	}

	collectionName := opts.QdrantConfig.Collection
	if collectionName == "" {
		collectionName = "agentic_memory"
	}

	embCfg := EmbeddingConfig{Model: EmbeddingModelBERT}
	if opts.EmbeddingConfig != nil {
		embCfg = *opts.EmbeddingConfig
	}

	s := &QdrantStore{
		client:          opts.Client,
		collectionName:  collectionName,
		config:          opts.Config,
		qdrantConfig:    opts.QdrantConfig,
		enabled:         true,
		embeddingConfig: embCfg,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := s.ensureCollection(ctx); err != nil {
		return nil, fmt.Errorf("failed to initialise qdrant memory collection: %w", err)
	}

	logging.ComponentEvent("memory", "qdrant_store_initialized", map[string]interface{}{
		"collection": collectionName,
		"embedding":  embCfg.Model,
	})
	return s, nil
}

func (s *QdrantStore) ensureCollection(ctx context.Context) error {
	exists, err := s.client.CollectionExists(ctx, s.collectionName)
	if err != nil {
		return fmt.Errorf("failed to check qdrant collection: %w", err)
	}
	if exists {
		return nil
	}

	dim := s.qdrantConfig.Dimension
	if dim <= 0 {
		dim = 384
	}

	if err := s.client.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: s.collectionName,
		VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
			Size:     uint64(dim), //nolint:gosec
			Distance: qdrant.Distance_Cosine,
		}),
		HnswConfig: &qdrant.HnswConfigDiff{
			M:           qdrant.PtrOf(uint64(16)), //nolint:gosec
			EfConstruct: qdrant.PtrOf(uint64(64)), //nolint:gosec
		},
	}); err != nil {
		return fmt.Errorf("failed to create qdrant memory collection: %w", err)
	}

	for _, field := range []string{"user_id", "project_id", "memory_type"} {
		if _, err := s.client.CreateFieldIndex(ctx, &qdrant.CreateFieldIndexCollection{
			CollectionName: s.collectionName,
			FieldName:      field,
			FieldType:      qdrant.FieldType_FieldTypeKeyword.Enum(),
		}); err != nil {
			return fmt.Errorf("failed to create index on %q: %w", field, err)
		}
	}

	return nil
}

// Qdrant only allows UUIDs and +ve integers.
// Ref: https://qdrant.tech/documentation/concepts/points/#point-ids
// So we create a deterministic UUID based on the original ID.
func arbitraryIDToUUID(id string) *qdrant.PointId {
	// If already a valid UUID, use it directly
	if _, err := uuid.Parse(id); err == nil {
		return qdrant.NewIDUUID(id)
	}

	// Otherwise create a deterministic UUID based on the ID
	deterministicUUID := uuid.NewSHA1(uuid.NameSpaceURL, []byte(id))
	return qdrant.NewIDUUID(deterministicUUID.String())
}

func payloadToMemory(payload map[string]*qdrant.Value) *Memory {
	str := func(key string) string {
		if v, ok := payload[key]; ok {
			return v.GetStringValue()
		}
		return ""
	}
	i64 := func(key string) int64 {
		if v, ok := payload[key]; ok {
			return v.GetIntegerValue()
		}
		return 0
	}
	unix := func(key string) time.Time {
		if t := i64(key); t != 0 {
			return time.Unix(t, 0)
		}
		return time.Time{}
	}
	var importance float32
	if v, ok := payload["importance"]; ok {
		importance = float32(v.GetDoubleValue())
	}
	var accessCount int
	if v, ok := payload["access_count"]; ok {
		accessCount = int(v.GetIntegerValue())
	}
	return &Memory{
		ID:              str("id"),
		Content:         str("content"),
		UserID:          str("user_id"),
		ProjectID:       str("project_id"),
		Type:            MemoryType(str("memory_type")),
		Source:          str("source"),
		Importance:      importance,
		AccessCount:     accessCount,
		CreatedAt:       unix("created_at"),
		UpdatedAt:       unix("updated_at"),
		LastAccessed:    unix("last_accessed"),
		CreatedByUserID: str("created_by_user_id"),
		ConversationID:  str("conversation_id"),
		CreatedVia:      str("created_via"),
	}
}

func memoryToPayload(mem *Memory) map[string]any {
	return map[string]any{
		"id":                 mem.ID,
		"content":            mem.Content,
		"user_id":            mem.UserID,
		"project_id":         mem.ProjectID,
		"memory_type":        string(mem.Type),
		"source":             mem.Source,
		"importance":         float64(mem.Importance),
		"access_count":       int64(mem.AccessCount),
		"created_at":         mem.CreatedAt.Unix(),
		"updated_at":         mem.UpdatedAt.Unix(),
		"last_accessed":      mem.LastAccessed.Unix(),
		"created_by_user_id": mem.CreatedByUserID,
		"conversation_id":    mem.ConversationID,
		"created_via":        mem.CreatedVia,
	}
}

func (s *QdrantStore) Store(ctx context.Context, mem *Memory) error {
	if !s.enabled {
		return fmt.Errorf("qdrant store is not enabled")
	}
	if mem.ID == "" {
		return fmt.Errorf("memory ID is required")
	}
	if mem.Content == "" {
		return fmt.Errorf("memory content is required")
	}
	if mem.UserID == "" {
		return fmt.Errorf("user ID is required")
	}

	emb := mem.Embedding
	if len(emb) == 0 {
		var err error
		emb, err = GenerateEmbedding(mem.Content, s.embeddingConfig)
		if err != nil {
			return fmt.Errorf("failed to generate embedding: %w", err)
		}
	}

	now := time.Now()
	if mem.CreatedAt.IsZero() {
		mem.CreatedAt = now
	}
	mem.UpdatedAt = now
	if mem.LastAccessed.IsZero() {
		mem.LastAccessed = now
	}

	wait := true
	_, err := s.client.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: s.collectionName,
		Wait:           &wait,
		Points: []*qdrant.PointStruct{{
			Id:      arbitraryIDToUUID(mem.ID),
			Vectors: qdrant.NewVectorsDense(emb),
			Payload: qdrant.NewValueMap(memoryToPayload(mem)),
		}},
	})
	if err != nil {
		return fmt.Errorf("qdrant upsert failed: %w", err)
	}

	RecordMemoryStoreOperation("qdrant", "store", "success", 0)
	return nil
}

func (s *QdrantStore) Retrieve(ctx context.Context, opts RetrieveOptions) ([]*RetrieveResult, error) {
	if !s.enabled {
		return nil, nil
	}

	emb, err := GenerateEmbedding(opts.Query, s.embeddingConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	limit := opts.Limit
	if limit <= 0 {
		limit = 5
	}
	threshold := opts.Threshold
	if threshold <= 0 {
		threshold = float32(s.config.DefaultSimilarityThreshold)
	}

	must := []*qdrant.Condition{
		qdrant.NewMatchKeyword("user_id", opts.UserID),
	}
	if opts.ProjectID != "" {
		must = append(must, qdrant.NewMatchKeyword("project_id", opts.ProjectID))
	}
	if len(opts.Types) > 0 {
		typeVals := make([]string, len(opts.Types))
		for i, t := range opts.Types {
			typeVals[i] = string(t)
		}
		must = append(must, qdrant.NewMatchKeywords("memory_type", typeVals...))
	}

	scored, err := s.client.Query(ctx, &qdrant.QueryPoints{
		CollectionName: s.collectionName,
		Query:          qdrant.NewQueryDense(emb),
		Limit:          qdrant.PtrOf(uint64(limit)), //nolint:gosec
		ScoreThreshold: &threshold,
		WithPayload:    qdrant.NewWithPayload(true),
		Filter:         &qdrant.Filter{Must: must},
	})
	if err != nil {
		return nil, fmt.Errorf("qdrant query failed: %w", err)
	}

	results := make([]*RetrieveResult, 0, len(scored))
	for _, sp := range scored {
		m := payloadToMemory(sp.Payload)
		results = append(results, &RetrieveResult{Memory: m, Score: sp.Score})
	}

	RecordMemoryStoreOperation("qdrant", "retrieve", "success", 0)
	return results, nil
}

func (s *QdrantStore) Get(ctx context.Context, id string) (*Memory, error) {
	if !s.enabled {
		return nil, fmt.Errorf("qdrant store is not enabled")
	}

	pts, err := s.client.Get(ctx, &qdrant.GetPoints{
		CollectionName: s.collectionName,
		Ids:            []*qdrant.PointId{arbitraryIDToUUID(id)},
		WithPayload:    qdrant.NewWithPayload(true),
	})
	if err != nil {
		return nil, fmt.Errorf("qdrant get failed: %w", err)
	}
	if len(pts) == 0 {
		return nil, fmt.Errorf("memory %q not found", id)
	}

	return payloadToMemory(pts[0].Payload), nil
}

func (s *QdrantStore) Update(ctx context.Context, id string, mem *Memory) error {
	if !s.enabled {
		return fmt.Errorf("qdrant store is not enabled")
	}

	if _, err := s.Get(ctx, id); err != nil {
		return err
	}

	mem.ID = id
	emb := mem.Embedding
	if len(emb) == 0 {
		var err error
		emb, err = GenerateEmbedding(mem.Content, s.embeddingConfig)
		if err != nil {
			return fmt.Errorf("failed to generate embedding: %w", err)
		}
	}
	mem.UpdatedAt = time.Now()

	wait := true
	_, err := s.client.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: s.collectionName,
		Wait:           &wait,
		Points: []*qdrant.PointStruct{{
			Id:      arbitraryIDToUUID(id),
			Vectors: qdrant.NewVectorsDense(emb),
			Payload: qdrant.NewValueMap(memoryToPayload(mem)),
		}},
	})
	if err != nil {
		return fmt.Errorf("qdrant update failed: %w", err)
	}

	RecordMemoryStoreOperation("qdrant", "update", "success", 0)
	return nil
}

func (s *QdrantStore) List(ctx context.Context, opts ListOptions) (*ListResult, error) {
	if !s.enabled {
		return &ListResult{}, nil
	}

	limit := opts.Limit
	if limit <= 0 {
		limit = 20
	}
	if limit > 100 {
		limit = 100
	}

	must := []*qdrant.Condition{
		qdrant.NewMatchKeyword("user_id", opts.UserID),
	}
	if len(opts.Types) > 0 {
		typeVals := make([]string, len(opts.Types))
		for i, t := range opts.Types {
			typeVals[i] = string(t)
		}
		must = append(must, qdrant.NewMatchKeywords("memory_type", typeVals...))
	}

	points, _, err := s.client.ScrollAndOffset(ctx, &qdrant.ScrollPoints{
		CollectionName: s.collectionName,
		Filter:         &qdrant.Filter{Must: must},
		Limit:          qdrant.PtrOf(uint32(limit)), //nolint:gosec
		WithPayload:    qdrant.NewWithPayload(true),
	})
	if err != nil {
		return nil, fmt.Errorf("qdrant scroll failed: %w", err)
	}

	memories := make([]*Memory, 0, len(points))
	for _, pt := range points {
		memories = append(memories, payloadToMemory(pt.Payload))
	}

	return &ListResult{
		Memories: memories,
		Total:    len(memories),
		Limit:    limit,
	}, nil
}

func (s *QdrantStore) Forget(ctx context.Context, id string) error {
	if !s.enabled {
		return fmt.Errorf("qdrant store is not enabled")
	}

	if _, err := s.Get(ctx, id); err != nil {
		return err
	}

	wait := true
	_, err := s.client.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: s.collectionName,
		Wait:           &wait,
		Points:         qdrant.NewPointsSelectorIDs([]*qdrant.PointId{arbitraryIDToUUID(id)}),
	})
	if err != nil {
		return fmt.Errorf("qdrant delete failed: %w", err)
	}

	RecordMemoryStoreOperation("qdrant", "forget", "success", 0)
	return nil
}

func (s *QdrantStore) ForgetByScope(ctx context.Context, scope MemoryScope) error {
	if !s.enabled {
		return fmt.Errorf("qdrant store is not enabled")
	}
	if scope.UserID == "" {
		return fmt.Errorf("user_id is required for ForgetByScope")
	}

	must := []*qdrant.Condition{
		qdrant.NewMatchKeyword("user_id", scope.UserID),
	}
	if scope.ProjectID != "" {
		must = append(must, qdrant.NewMatchKeyword("project_id", scope.ProjectID))
	}
	if len(scope.Types) > 0 {
		typeVals := make([]string, len(scope.Types))
		for i, t := range scope.Types {
			typeVals[i] = string(t)
		}
		must = append(must, qdrant.NewMatchKeywords("memory_type", typeVals...))
	}

	wait := true
	_, err := s.client.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: s.collectionName,
		Wait:           &wait,
		Points: qdrant.NewPointsSelectorFilter(&qdrant.Filter{
			Must: must,
		}),
	})
	if err != nil {
		return fmt.Errorf("qdrant ForgetByScope failed: %w", err)
	}

	RecordMemoryStoreOperation("qdrant", "forget_by_scope", "success", 0)
	return nil
}

func (s *QdrantStore) IsEnabled() bool { return s.enabled }

func (s *QdrantStore) CheckConnection(ctx context.Context) error {
	_, err := s.client.ListCollections(ctx)
	if err != nil {
		return fmt.Errorf("qdrant connection check failed: %w", err)
	}
	return nil
}

func (s *QdrantStore) Close() error {
	if s.client != nil {
		return s.client.Close()
	}
	return nil
}
