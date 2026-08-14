package cache

import (
	"context"
	"fmt"
	"time"

	"github.com/qdrant/go-client/qdrant"
)

var _ ExactCacheBackend = (*QdrantCache)(nil)

// FindExact returns a deterministic Qdrant exact-response entry.
func (c *QdrantCache) FindExact(
	partition string,
	fingerprint string,
) (LookupResult, error) {
	if !c.enabled || fingerprint == "" {
		return LookupResult{}, nil
	}
	recordID := exactCacheRecordID(partition, fingerprint)
	points, err := c.client.Get(context.Background(), &qdrant.GetPoints{
		CollectionName: c.collectionName,
		Ids: []*qdrant.PointId{
			arbitraryIDToUUID(recordID),
		},
		WithPayload: qdrant.NewWithPayload(true),
	})
	if err != nil {
		return LookupResult{}, fmt.Errorf("qdrant exact lookup failed: %w", err)
	}
	if len(points) == 0 {
		return LookupResult{}, nil
	}
	payload := points[0].Payload
	if payload["model"].GetStringValue() != partition ||
		payload["query"].GetStringValue() != exactCacheQueryMarker {
		return LookupResult{}, nil
	}
	expiresAt := payload["expires_at"].GetIntegerValue()
	if expiresAt > 0 && expiresAt <= time.Now().Unix() {
		return LookupResult{}, nil
	}
	responseBody := payload["response_body"].GetStringValue()
	if responseBody == "" {
		return LookupResult{}, nil
	}
	return LookupResult{
		ResponseBody: []byte(responseBody),
		Found:        true,
		Similarity:   1,
	}, nil
}

// AddExact writes a deterministic Qdrant exact-response entry.
func (c *QdrantCache) AddExact(
	partition string,
	fingerprint string,
	responseBody []byte,
	ttlSeconds int,
) error {
	if !c.enabled || fingerprint == "" || ttlSeconds == 0 {
		return nil
	}
	recordID := exactCacheRecordID(partition, fingerprint)
	wait := true
	_, err := c.client.Upsert(context.Background(), &qdrant.UpsertPoints{
		CollectionName: c.collectionName,
		Wait:           &wait,
		Points: []*qdrant.PointStruct{{
			Id: arbitraryIDToUUID(recordID),
			Vectors: qdrant.NewVectorsDense(
				exactCacheSentinelVector(c.embeddingDimension()),
			),
			Payload: qdrant.NewValueMap(map[string]any{
				"request_id":    recordID,
				"model":         partition,
				"query":         exactCacheQueryMarker,
				"request_body":  "",
				"response_body": string(responseBody),
				"timestamp":     time.Now().Unix(),
				"expires_at":    c.expiresAt(ttlSeconds),
			}),
		}},
	})
	if err != nil {
		return fmt.Errorf("qdrant exact write failed: %w", err)
	}
	return nil
}
