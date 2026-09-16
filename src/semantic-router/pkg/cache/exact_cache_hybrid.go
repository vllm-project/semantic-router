//go:build !windows && cgo

package cache

import "context"

var _ ExactCacheBackend = (*HybridCache)(nil)

// FindExact delegates persistent exact lookup to the hybrid Milvus backend.
func (h *HybridCache) FindExact(
	ctx context.Context,
	partition string,
	fingerprint string,
) (LookupResult, error) {
	if !h.enabled || h.milvusCache == nil {
		return LookupResult{}, nil
	}
	return h.milvusCache.FindExact(ctx, partition, fingerprint)
}

// AddExact delegates persistent exact writes to the hybrid Milvus backend.
func (h *HybridCache) AddExact(
	ctx context.Context,
	partition string,
	fingerprint string,
	responseBody []byte,
	ttlSeconds int,
) error {
	if !h.enabled || h.milvusCache == nil {
		return nil
	}
	return h.milvusCache.AddExact(
		ctx,
		partition,
		fingerprint,
		responseBody,
		ttlSeconds,
	)
}
