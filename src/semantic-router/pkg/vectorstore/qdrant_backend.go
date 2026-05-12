package vectorstore

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/qdrant/go-client/qdrant"
)

type QdrantBackendConfig struct {
	Host             string // hostname
	Port             int    // gRPC port (default 6334)
	APIKey           string // optional API key
	UseTLS           bool
	CollectionPrefix string // prefix for collection names (default "vsr_vs_")
	ConnectTimeout   int    // connection timeout in seconds (default 10)
}

type QdrantBackend struct {
	client           *qdrant.Client
	collectionPrefix string
}

func NewQdrantBackend(cfg QdrantBackendConfig) (*QdrantBackend, error) {
	if cfg.Host == "" {
		return nil, fmt.Errorf("qdrant host is required")
	}

	prefix := cfg.CollectionPrefix
	if prefix == "" {
		prefix = "vsr_vs_"
	}
	port := cfg.Port
	if port <= 0 {
		port = 6334
	}
	timeout := cfg.ConnectTimeout
	if timeout <= 0 {
		timeout = 10
	}

	client, err := qdrant.NewClient(&qdrant.Config{
		Host:   cfg.Host,
		Port:   port,
		APIKey: cfg.APIKey,
		UseTLS: cfg.UseTLS,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create qdrant client: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
	defer cancel()
	if _, connErr := client.ListCollections(ctx); connErr != nil {
		_ = client.Close()
		return nil, fmt.Errorf("qdrant connection check failed: %w", connErr)
	}

	return &QdrantBackend{
		client:           client,
		collectionPrefix: prefix,
	}, nil
}

func (q *QdrantBackend) collectionName(vectorStoreID string) string {
	return q.collectionPrefix + vectorStoreID
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

func (q *QdrantBackend) CreateCollection(ctx context.Context, vectorStoreID string, dimension int) error {
	colName := q.collectionName(vectorStoreID)

	exists, err := q.client.CollectionExists(ctx, colName)
	if err != nil {
		return fmt.Errorf("failed to check existence of collection %q: %w", colName, err)
	}
	if exists {
		return fmt.Errorf("collection %q already exists", colName)
	}

	if err := q.client.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: colName,
		VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
			Size:     uint64(dimension), //nolint:gosec
			Distance: qdrant.Distance_Cosine,
		}),
	}); err != nil {
		return fmt.Errorf("failed to create qdrant collection %q: %w", colName, err)
	}

	// Create a keyword payload index on file_id for efficient filter-based deletes.
	if _, err := q.client.CreateFieldIndex(ctx, &qdrant.CreateFieldIndexCollection{
		CollectionName: colName,
		FieldName:      "file_id",
		FieldType:      qdrant.FieldType_FieldTypeKeyword.Enum(),
	}); err != nil {
		return fmt.Errorf("failed to create file_id index on %q: %w", colName, err)
	}

	return nil
}

func (q *QdrantBackend) DeleteCollection(ctx context.Context, vectorStoreID string) error {
	colName := q.collectionName(vectorStoreID)
	if err := q.client.DeleteCollection(ctx, colName); err != nil {
		return fmt.Errorf("failed to drop qdrant collection %q: %w", colName, err)
	}
	return nil
}

func (q *QdrantBackend) CollectionExists(ctx context.Context, vectorStoreID string) (bool, error) {
	colName := q.collectionName(vectorStoreID)
	exists, err := q.client.CollectionExists(ctx, colName)
	if err != nil {
		return false, fmt.Errorf("failed to check collection %q: %w", colName, err)
	}
	return exists, nil
}

func (q *QdrantBackend) InsertChunks(ctx context.Context, vectorStoreID string, chunks []EmbeddedChunk) error {
	if len(chunks) == 0 {
		return nil
	}

	colName := q.collectionName(vectorStoreID)
	now := time.Now().Unix()

	points := make([]*qdrant.PointStruct, len(chunks))
	for i, c := range chunks {
		points[i] = &qdrant.PointStruct{
			Id:      arbitraryIDToUUID(c.ID),
			Vectors: qdrant.NewVectorsDense(c.Embedding),
			Payload: qdrant.NewValueMap(map[string]any{
				"doc_id":      c.ID,
				"file_id":     c.FileID,
				"filename":    c.Filename,
				"content":     c.Content,
				"chunk_index": int64(c.ChunkIndex),
				"created_at":  now,
			}),
		}
	}

	wait := true
	if _, err := q.client.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: colName,
		Wait:           &wait,
		Points:         points,
	}); err != nil {
		return fmt.Errorf("failed to upsert chunks into %q: %w", colName, err)
	}

	return nil
}

func (q *QdrantBackend) DeleteByFileID(ctx context.Context, vectorStoreID string, fileID string) error {
	if !safeIdentifierPattern.MatchString(fileID) {
		return fmt.Errorf("invalid file ID: contains disallowed characters")
	}
	colName := q.collectionName(vectorStoreID)
	wait := true
	if _, err := q.client.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: colName,
		Wait:           &wait,
		Points: qdrant.NewPointsSelectorFilter(&qdrant.Filter{
			Must: []*qdrant.Condition{
				qdrant.NewMatchKeyword("file_id", fileID),
			},
		}),
	}); err != nil {
		return fmt.Errorf("failed to delete chunks for file %q from %q: %w", fileID, colName, err)
	}
	return nil
}

func (q *QdrantBackend) Search(
	ctx context.Context, vectorStoreID string, queryEmbedding []float32,
	topK int, threshold float32, filter map[string]interface{},
) ([]SearchResult, error) {
	colName := q.collectionName(vectorStoreID)

	var qdrantFilter *qdrant.Filter
	if fid, ok := filter["file_id"].(string); ok && fid != "" {
		if !safeIdentifierPattern.MatchString(fid) {
			return nil, fmt.Errorf("invalid file_id filter: contains disallowed characters")
		}
		qdrantFilter = &qdrant.Filter{
			Must: []*qdrant.Condition{
				qdrant.NewMatchKeyword("file_id", fid),
			},
		}
	}

	scored, err := q.client.Query(ctx, &qdrant.QueryPoints{
		CollectionName: colName,
		Query:          qdrant.NewQueryDense(queryEmbedding),
		Limit:          qdrant.PtrOf(uint64(topK)), //nolint:gosec
		ScoreThreshold: &threshold,
		WithPayload:    qdrant.NewWithPayload(true),
		Filter:         qdrantFilter,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to search %q: %w", colName, err)
	}

	results := make([]SearchResult, 0, len(scored))
	for _, sp := range scored {
		r := SearchResult{Score: float64(sp.Score)}

		if v, ok := sp.Payload["file_id"]; ok {
			r.FileID = v.GetStringValue()
		}
		if v, ok := sp.Payload["filename"]; ok {
			r.Filename = v.GetStringValue()
		}
		if v, ok := sp.Payload["content"]; ok {
			r.Content = v.GetStringValue()
		}
		if v, ok := sp.Payload["chunk_index"]; ok {
			r.ChunkIndex = int(v.GetIntegerValue())
		}

		results = append(results, r)
	}

	return results, nil
}

func (q *QdrantBackend) Close() error {
	if q.client != nil {
		return q.client.Close()
	}
	return nil
}
