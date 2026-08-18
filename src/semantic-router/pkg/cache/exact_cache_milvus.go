package cache

import (
	"context"
	"fmt"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

var _ ExactCacheBackend = (*MilvusCache)(nil)

// FindExact returns a deterministic Milvus exact-response entry.
func (c *MilvusCache) FindExact(
	partition string,
	fingerprint string,
) (LookupResult, error) {
	if !c.enabled || fingerprint == "" {
		return LookupResult{}, nil
	}
	recordID := exactCacheRecordID(partition, fingerprint)
	expr := fmt.Sprintf(
		`id == %s && model == %s && query == %s && (expires_at == 0 || expires_at > %d)`,
		milvusStringLiteral(recordID),
		milvusStringLiteral(partition),
		milvusStringLiteral(exactCacheQueryMarker),
		time.Now().Unix(),
	)
	results, err := c.client.Query(
		context.Background(),
		c.collectionName,
		nil,
		expr,
		[]string{"response_body"},
		c.searchQueryOptions()...,
	)
	if err != nil {
		return LookupResult{}, fmt.Errorf("milvus exact lookup failed: %w", err)
	}
	for _, result := range results {
		column, ok := result.(*entity.ColumnVarChar)
		if !ok || column.Name() != "response_body" || column.Len() == 0 {
			continue
		}
		responseBody, valueErr := column.ValueByIdx(0)
		if valueErr != nil {
			return LookupResult{}, fmt.Errorf(
				"milvus exact response decode failed: %w",
				valueErr,
			)
		}
		if responseBody == "" {
			return LookupResult{}, nil
		}
		return LookupResult{
			ResponseBody: []byte(responseBody),
			Found:        true,
			Similarity:   1,
		}, nil
	}
	return LookupResult{}, nil
}

// AddExact writes a deterministic Milvus exact-response entry.
func (c *MilvusCache) AddExact(
	partition string,
	fingerprint string,
	responseBody []byte,
	ttlSeconds int,
) error {
	if !c.enabled || fingerprint == "" || ttlSeconds == 0 {
		return nil
	}
	recordID := exactCacheRecordID(partition, fingerprint)
	effectiveTTL := effectiveExactTTL(ttlSeconds, c.ttlSeconds)
	now := time.Now()
	expiresAt := int64(0)
	if effectiveTTL > 0 {
		expiresAt = now.Add(time.Duration(effectiveTTL) * time.Second).Unix()
	}
	dimension := semanticCacheEmbeddingDimension(
		c.config.Collection.VectorField.Dimension,
		c.embeddingModel,
	)
	_, err := c.client.Upsert(
		context.Background(),
		c.collectionName,
		"",
		entity.NewColumnVarChar("id", []string{recordID}),
		entity.NewColumnVarChar("request_id", []string{recordID}),
		entity.NewColumnVarChar("model", []string{partition}),
		entity.NewColumnVarChar("query", []string{exactCacheQueryMarker}),
		entity.NewColumnVarChar("request_body", []string{""}),
		entity.NewColumnVarChar("response_body", []string{string(responseBody)}),
		entity.NewColumnFloatVector(
			c.config.Collection.VectorField.Name,
			dimension,
			[][]float32{exactCacheSentinelVector(dimension)},
		),
		entity.NewColumnInt64("timestamp", []int64{now.Unix()}),
		entity.NewColumnInt64("ttl_seconds", []int64{int64(effectiveTTL)}),
		entity.NewColumnInt64("expires_at", []int64{expiresAt}),
	)
	if err != nil {
		return fmt.Errorf("milvus exact write failed: %w", err)
	}
	if err := c.client.Flush(context.Background(), c.collectionName, false); err != nil {
		return fmt.Errorf("milvus exact flush failed: %w", err)
	}
	return nil
}
