package cache

import (
	"crypto/sha256"
	"encoding/hex"
)

const (
	exactCacheKeyPrefix   = "vsr:response-cache:exact:v1:"
	exactCacheQueryMarker = "__vsr_exact__"
)

func exactCacheStorageKey(partition string, fingerprint string) string {
	return exactCacheKeyPrefix + exactCacheRecordID(partition, fingerprint)
}

func exactCacheRecordID(partition string, fingerprint string) string {
	sum := sha256.Sum256([]byte(partition + "\x00" + fingerprint))
	return hex.EncodeToString(sum[:])
}

func effectiveExactTTL(requestTTL int, defaultTTL int) int {
	if requestTTL == -1 {
		return defaultTTL
	}
	return requestTTL
}

func exactCacheSentinelVector(dimension int) []float32 {
	vector := make([]float32, dimension)
	if dimension > 0 {
		vector[0] = 1
	}
	return vector
}
