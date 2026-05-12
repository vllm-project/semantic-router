package cache

import (
	"context"
	"crypto/md5" //nolint:gosec
	"fmt"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/qdrant/go-client/qdrant"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

const pendingResponseMarker = "__pending__"

type QdrantCache struct {
	SimilarityTracker
	client              *qdrant.Client
	cfg                 *config.QdrantConfig
	collectionName      string
	similarityThreshold float32
	ttlSeconds          int
	enabled             bool
	hitCount            int64
	missCount           int64
	embeddingModel      string
}

type QdrantCacheOptions struct {
	SimilarityThreshold float32
	TTLSeconds          int
	Enabled             bool
	Config              *config.QdrantConfig
	EmbeddingModel      string
}

func NewQdrantCache(opts QdrantCacheOptions) (*QdrantCache, error) {
	if !opts.Enabled {
		return &QdrantCache{enabled: false}, nil
	}

	if opts.Config == nil {
		return nil, fmt.Errorf("qdrant configuration is required")
	}

	port := opts.Config.Port
	if port <= 0 {
		port = 6334
	}
	collectionName := opts.Config.CollectionName
	if collectionName == "" {
		collectionName = "semantic_cache"
	}
	embeddingModel := opts.EmbeddingModel
	if embeddingModel == "" {
		embeddingModel = "bert"
	}

	client, err := qdrant.NewClient(&qdrant.Config{
		Host:   opts.Config.Host,
		Port:   port,
		APIKey: opts.Config.APIKey,
		UseTLS: opts.Config.UseTLS,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create qdrant client: %w", err)
	}

	c := &QdrantCache{
		client:              client,
		cfg:                 opts.Config,
		collectionName:      collectionName,
		similarityThreshold: opts.SimilarityThreshold,
		ttlSeconds:          opts.TTLSeconds,
		enabled:             true,
		embeddingModel:      embeddingModel,
	}

	if err := c.CheckConnection(); err != nil {
		_ = client.Close()
		return nil, err
	}

	if err := c.ensureCollection(); err != nil {
		_ = client.Close()
		return nil, fmt.Errorf("failed to initialise qdrant cache collection: %w", err)
	}

	return c, nil
}

func (c *QdrantCache) connTimeout() time.Duration {
	if c.cfg != nil && c.cfg.ConnectTimeout > 0 {
		return time.Duration(c.cfg.ConnectTimeout) * time.Second
	}
	return 30 * time.Second
}

func (c *QdrantCache) ensureCollection() error {
	ctx, cancel := context.WithTimeout(context.Background(), c.connTimeout())
	defer cancel()

	exists, err := c.client.CollectionExists(ctx, c.collectionName)
	if err != nil {
		return fmt.Errorf("failed to check qdrant collection: %w", err)
	}
	if exists {
		return nil
	}

	testEmb, err := c.getEmbedding("test")
	if err != nil {
		return fmt.Errorf("failed to detect embedding dimension: %w", err)
	}
	dim := len(testEmb)

	if err := c.client.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: c.collectionName,
		VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
			Size:     uint64(dim), //nolint:gosec
			Distance: qdrant.Distance_Cosine,
		}),
		HnswConfig: &qdrant.HnswConfigDiff{
			M:           qdrant.PtrOf(uint64(16)), //nolint:gosec
			EfConstruct: qdrant.PtrOf(uint64(64)), //nolint:gosec
		},
	}); err != nil {
		return fmt.Errorf("failed to create qdrant cache collection: %w", err)
	}

	if _, err := c.client.CreateFieldIndex(ctx, &qdrant.CreateFieldIndexCollection{
		CollectionName: c.collectionName,
		FieldName:      "request_id",
		FieldType:      qdrant.FieldType_FieldTypeKeyword.Enum(),
	}); err != nil {
		return fmt.Errorf("failed to create request_id index: %w", err)
	}

	logging.Debugf("QdrantCache: created collection %q (dim=%d)", c.collectionName, dim)
	return nil
}

func (c *QdrantCache) getEmbedding(text string) ([]float32, error) {
	switch c.embeddingModel {
	case "qwen3":
		out, err := candle_binding.GetEmbeddingBatched(text, "qwen3", 0)
		if err != nil {
			return nil, err
		}
		return out.Embedding, nil
	case "gemma":
		out, err := candle_binding.GetEmbeddingWithModelType(text, "gemma", 0)
		if err != nil {
			return nil, err
		}
		return out.Embedding, nil
	case "mmbert":
		out, err := candle_binding.GetEmbedding2DMatryoshka(text, "mmbert", 0, 0)
		if err != nil {
			return nil, err
		}
		return out.Embedding, nil
	case "multimodal":
		out, err := candle_binding.GetEmbeddingWithModelType(text, "multimodal", 384)
		if err != nil {
			return nil, err
		}
		return out.Embedding, nil
	default: // "bert" or ""
		return candle_binding.GetEmbedding(text, 0)
	}
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

func (c *QdrantCache) expiresAt(ttlSeconds int) int64 {
	effective := ttlSeconds
	if effective < 0 {
		effective = c.ttlSeconds
	}
	if effective <= 0 {
		return 0
	}
	return time.Now().Add(time.Duration(effective) * time.Second).Unix()
}

func (c *QdrantCache) IsEnabled() bool { return c.enabled }

func (c *QdrantCache) CheckConnection() error {
	ctx, cancel := context.WithTimeout(context.Background(), c.connTimeout())
	defer cancel()
	_, err := c.client.ListCollections(ctx)
	if err != nil {
		return fmt.Errorf("qdrant connection check failed: %w", err)
	}
	return nil
}

func (c *QdrantCache) AddPendingRequest(requestID, model, query string, requestBody []byte, ttlSeconds int) error {
	if !c.enabled || ttlSeconds == 0 {
		return nil
	}

	emb, err := c.getEmbedding(query)
	if err != nil {
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	id := fmt.Sprintf("%x", md5.Sum([]byte(requestID))) //nolint:gosec
	wait := true
	_, err = c.client.Upsert(context.Background(), &qdrant.UpsertPoints{
		CollectionName: c.collectionName,
		Wait:           &wait,
		Points: []*qdrant.PointStruct{{
			Id:      arbitraryIDToUUID(id),
			Vectors: qdrant.NewVectorsDense(emb),
			Payload: qdrant.NewValueMap(map[string]any{
				"request_id":    requestID,
				"model":         model,
				"query":         query,
				"request_body":  string(requestBody),
				"response_body": pendingResponseMarker,
				"timestamp":     time.Now().Unix(),
				"expires_at":    int64(0),
			}),
		}},
	})
	if err != nil {
		metrics.RecordCacheOperation("qdrant", "add_pending", "error", 0)
		return fmt.Errorf("failed to store pending request: %w", err)
	}
	metrics.RecordCacheOperation("qdrant", "add_pending", "success", 0)
	return nil
}

func (c *QdrantCache) UpdateWithResponse(requestID string, responseBody []byte, ttlSeconds int) error {
	if !c.enabled {
		return nil
	}

	ctx := context.Background()

	scrollResult, _, err := c.client.ScrollAndOffset(ctx, &qdrant.ScrollPoints{
		CollectionName: c.collectionName,
		Filter: &qdrant.Filter{
			Must: []*qdrant.Condition{
				qdrant.NewMatchKeyword("request_id", requestID),
			},
		},
		Limit:       qdrant.PtrOf(uint32(1)),
		WithPayload: qdrant.NewWithPayload(true),
		WithVectors: qdrant.NewWithVectors(true),
	})
	if err != nil {
		return fmt.Errorf("failed to find pending entry for request %q: %w", requestID, err)
	}
	if len(scrollResult) == 0 {
		return fmt.Errorf("no pending entry found for request_id %q", requestID)
	}

	pt := scrollResult[0]
	var emb []float32
	if pt.Vectors != nil {
		if dv := pt.Vectors.GetVector(); dv != nil {
			emb = dv.GetDenseVector().GetData()
		}
	}
	if len(emb) == 0 {
		return fmt.Errorf("pending entry for %q has no vector", requestID)
	}

	wait := true
	_, err = c.client.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: c.collectionName,
		Wait:           &wait,
		Points: []*qdrant.PointStruct{{
			Id:      pt.Id,
			Vectors: qdrant.NewVectorsDense(emb),
			Payload: qdrant.NewValueMap(map[string]any{
				"request_id":    requestID,
				"model":         pt.Payload["model"].GetStringValue(),
				"query":         pt.Payload["query"].GetStringValue(),
				"request_body":  pt.Payload["request_body"].GetStringValue(),
				"response_body": string(responseBody),
				"timestamp":     time.Now().Unix(),
				"expires_at":    c.expiresAt(ttlSeconds),
			}),
		}},
	})
	if err != nil {
		metrics.RecordCacheOperation("qdrant", "update_response", "error", 0)
		return fmt.Errorf("failed to update entry with response: %w", err)
	}
	metrics.RecordCacheOperation("qdrant", "update_response", "success", 0)
	return nil
}

func (c *QdrantCache) AddEntry(requestID, model, query string, requestBody, responseBody []byte, ttlSeconds int) error {
	if !c.enabled || ttlSeconds == 0 {
		return nil
	}

	emb, err := c.getEmbedding(query)
	if err != nil {
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	id := fmt.Sprintf("%x", md5.Sum([]byte(requestID))) //nolint:gosec
	wait := true
	_, err = c.client.Upsert(context.Background(), &qdrant.UpsertPoints{
		CollectionName: c.collectionName,
		Wait:           &wait,
		Points: []*qdrant.PointStruct{{
			Id:      arbitraryIDToUUID(id),
			Vectors: qdrant.NewVectorsDense(emb),
			Payload: qdrant.NewValueMap(map[string]any{
				"request_id":    requestID,
				"model":         model,
				"query":         query,
				"request_body":  string(requestBody),
				"response_body": string(responseBody),
				"timestamp":     time.Now().Unix(),
				"expires_at":    c.expiresAt(ttlSeconds),
			}),
		}},
	})
	if err != nil {
		metrics.RecordCacheOperation("qdrant", "add_entry", "error", 0)
		return fmt.Errorf("failed to store cache entry: %w", err)
	}
	metrics.RecordCacheOperation("qdrant", "add_entry", "success", 0)
	return nil
}

func (c *QdrantCache) FindSimilar(model, query string) ([]byte, bool, error) {
	return c.FindSimilarWithThreshold(model, query, c.similarityThreshold)
}

func (c *QdrantCache) FindSimilarWithThreshold(model, query string, threshold float32) ([]byte, bool, error) {
	start := time.Now()

	if !c.enabled {
		return nil, false, nil
	}

	emb, err := c.getEmbedding(query)
	if err != nil {
		metrics.RecordCacheOperation("qdrant", "find_similar", "error", time.Since(start).Seconds())
		return nil, false, fmt.Errorf("failed to generate embedding: %w", err)
	}

	now := time.Now().Unix()
	filter := &qdrant.Filter{
		Must: []*qdrant.Condition{
			{
				ConditionOneOf: &qdrant.Condition_Filter{
					Filter: &qdrant.Filter{
						MustNot: []*qdrant.Condition{
							qdrant.NewMatchKeyword("response_body", pendingResponseMarker),
						},
					},
				},
			},
		},
		Should: []*qdrant.Condition{
			// expires_at == 0 (no TTL) OR expires_at > now
			qdrant.NewMatchInt("expires_at", 0),
			qdrant.NewRange("expires_at", &qdrant.Range{Gt: qdrant.PtrOf(float64(now))}),
		},
	}

	scored, err := c.client.Query(context.Background(), &qdrant.QueryPoints{
		CollectionName: c.collectionName,
		Query:          qdrant.NewQueryDense(emb),
		Limit:          qdrant.PtrOf(uint64(1)), //nolint:gosec
		ScoreThreshold: &threshold,
		WithPayload:    qdrant.NewWithPayload(true),
		Filter:         filter,
	})
	if err != nil {
		atomic.AddInt64(&c.missCount, 1)
		metrics.RecordCacheOperation("qdrant", "find_similar", "error", time.Since(start).Seconds())
		return nil, false, nil
	}

	if len(scored) == 0 {
		atomic.AddInt64(&c.missCount, 1)
		metrics.RecordCacheOperation("qdrant", "find_similar", "miss", time.Since(start).Seconds())
		return nil, false, nil
	}

	best := scored[0]
	c.StoreSimilarity(best.Score)

	responseBody := best.Payload["response_body"].GetStringValue()
	if responseBody == "" || responseBody == pendingResponseMarker {
		atomic.AddInt64(&c.missCount, 1)
		metrics.RecordCacheOperation("qdrant", "find_similar", "miss", time.Since(start).Seconds())
		return nil, false, nil
	}

	atomic.AddInt64(&c.hitCount, 1)
	logging.Debugf("QdrantCache: CACHE HIT similarity=%.4f threshold=%.4f response_size=%d",
		best.Score, threshold, len(responseBody))
	metrics.RecordCacheOperation("qdrant", "find_similar", "hit", time.Since(start).Seconds())
	return []byte(responseBody), true, nil
}

func (c *QdrantCache) GetStats() CacheStats {
	hits := atomic.LoadInt64(&c.hitCount)
	misses := atomic.LoadInt64(&c.missCount)
	total := hits + misses
	ratio := float64(0)
	if total > 0 {
		ratio = float64(hits) / float64(total)
	}
	return CacheStats{
		HitCount:  hits,
		MissCount: misses,
		HitRatio:  ratio,
	}
}

func (c *QdrantCache) SearchCollection(
	ctx context.Context,
	collectionName string,
	embedding []float32,
	threshold float32,
	topK int,
	contentField string,
) ([]string, []float32, error) {
	if !c.enabled {
		return nil, nil, nil
	}
	if collectionName == "" {
		return nil, nil, fmt.Errorf("collection name is required")
	}
	if contentField == "" {
		contentField = "content"
	}

	scored, err := c.client.Query(ctx, &qdrant.QueryPoints{
		CollectionName: collectionName,
		Query:          qdrant.NewQueryDense(embedding),
		Limit:          qdrant.PtrOf(uint64(topK)), //nolint:gosec
		ScoreThreshold: &threshold,
		WithPayload:    qdrant.NewWithPayload(true),
	})
	if err != nil {
		return nil, nil, fmt.Errorf("qdrant search collection %q failed: %w", collectionName, err)
	}

	texts := make([]string, 0, len(scored))
	scores := make([]float32, 0, len(scored))
	for _, sp := range scored {
		if v, ok := sp.Payload[contentField]; ok {
			texts = append(texts, v.GetStringValue())
		}
		scores = append(scores, sp.Score)
	}
	return texts, scores, nil
}

func (c *QdrantCache) Close() error {
	if c.client != nil {
		return c.client.Close()
	}
	return nil
}
