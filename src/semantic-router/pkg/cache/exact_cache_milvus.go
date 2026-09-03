package cache

import (
	"context"
	"fmt"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

var _ ExactCacheBackend = (*MilvusCache)(nil)

func parseMilvusColumnVarChar(col *entity.ColumnVarChar) (string, error) {
	if col.Name() != "response_body" || col.Len() == 0 {
		return "", nil
	}
	val, err := col.ValueByIdx(0)
	if err != nil {
		return "", fmt.Errorf("milvus exact response decode failed: %w", err)
	}
	return val, nil
}

func parseMilvusColumnInt64(col *entity.ColumnInt64, storedAt, expiresAt *time.Time) {
	if col.Len() == 0 {
		return
	}
	val, _ := col.ValueByIdx(0)
	if val <= 0 {
		return
	}
	if col.Name() == "timestamp" {
		*storedAt = time.Unix(val, 0)
	} else if col.Name() == "expires_at" {
		*expiresAt = time.Unix(val, 0)
	}
}

func parseMilvusExactColumns(results []entity.Column) (string, time.Time, time.Time, error) {
	var responseBody string
	var storedAt, expiresAt time.Time
	for _, result := range results {
		switch column := result.(type) {
		case *entity.ColumnVarChar:
			val, err := parseMilvusColumnVarChar(column)
			if err != nil {
				return "", time.Time{}, time.Time{}, err
			}
			if val != "" {
				responseBody = val
			}
		case *entity.ColumnInt64:
			parseMilvusColumnInt64(column, &storedAt, &expiresAt)
		}
	}
	return responseBody, storedAt, expiresAt, nil
}

// FindExact returns a deterministic Milvus exact-response entry.
func (c *MilvusCache) FindExact(
	ctx context.Context,
	partition string,
	fingerprint string,
) (LookupResult, error) {
	if !c.enabled || fingerprint == "" {
		return LookupResult{}, nil
	}
	if ctx == nil {
		ctx = context.Background()
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
		ctx,
		c.collectionName,
		nil,
		expr,
		[]string{"response_body", "timestamp", "expires_at"},
		c.searchQueryOptions()...,
	)
	if err != nil {
		return LookupResult{}, fmt.Errorf("milvus exact lookup failed: %w", err)
	}
	responseBody, storedAt, expiresAt, err := parseMilvusExactColumns(results)
	if err != nil {
		return LookupResult{}, err
	}
	if responseBody == "" {
		return LookupResult{}, nil
	}
	return lookupResultFromTimestamps([]byte(responseBody), 1, storedAt, expiresAt), nil
}

// AddExact writes a deterministic Milvus exact-response entry.
func (c *MilvusCache) AddExact(
	ctx context.Context,
	partition string,
	fingerprint string,
	responseBody []byte,
	ttlSeconds int,
) error {
	if !c.enabled || fingerprint == "" || ttlSeconds == 0 {
		return nil
	}
	if ctx == nil {
		ctx = context.Background()
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
		ctx,
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
	if err := c.client.Flush(ctx, c.collectionName, false); err != nil {
		return fmt.Errorf("milvus exact flush failed: %w", err)
	}
	return nil
}
