package cache

import (
	"context"
	"fmt"
	"time"

	"github.com/qdrant/go-client/qdrant"
)

var _ ExactCacheBackend = (*QdrantCache)(nil)

// parseQdrantPayloadTiming extracts storedAt and expiresAt from a Qdrant point payload.
func parseQdrantPayloadTiming(payload map[string]*qdrant.Value) (time.Time, time.Time) {
	var storedAt, expiresAt time.Time
	if tsVal, ok := payload["timestamp"]; ok {
		if ts := tsVal.GetIntegerValue(); ts > 0 {
			storedAt = time.Unix(ts, 0)
		}
	}
	if expVal, ok := payload["expires_at"]; ok {
		if exp := expVal.GetIntegerValue(); exp > 0 {
			expiresAt = time.Unix(exp, 0)
		}
	}
	return storedAt, expiresAt
}

func isQdrantExactPayloadValid(payload map[string]*qdrant.Value, partition string) bool {
	if payload["model"].GetStringValue() != partition ||
		payload["query"].GetStringValue() != exactCacheQueryMarker {
		return false
	}
	expiresAt := payload["expires_at"].GetIntegerValue()
	return expiresAt == 0 || expiresAt > time.Now().Unix()
}

// FindExact returns a deterministic Qdrant exact-response entry.
func (c *QdrantCache) FindExact(
	ctx context.Context,
	partition string,
	fingerprint string,
) (LookupResult, error) {
	if !c.enabled || fingerprint == "" {
		return LookupResult{}, nil
	}
	recordID := exactCacheRecordID(partition, fingerprint)
	points, err := c.client.Get(ctx, &qdrant.GetPoints{
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
	if !isQdrantExactPayloadValid(payload, partition) {
		return LookupResult{}, nil
	}
	responseBody := payload["response_body"].GetStringValue()
	if responseBody == "" {
		return LookupResult{}, nil
	}
	storedAt, expiresAt := parseQdrantPayloadTiming(payload)
	return lookupResultFromTimestamps([]byte(responseBody), 1, storedAt, expiresAt), nil
}

// AddExact writes a deterministic Qdrant exact-response entry.
func (c *QdrantCache) AddExact(
	ctx context.Context,
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
	_, err := c.client.Upsert(ctx, &qdrant.UpsertPoints{
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
