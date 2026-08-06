package cache

import (
	"crypto/sha256"
	"encoding/hex"
)

const exactCacheKeyPrefix = "vsr:response-cache:exact:v1:"

func exactCacheStorageKey(partition string, fingerprint string) string {
	sum := sha256.Sum256([]byte(partition + "\x00" + fingerprint))
	return exactCacheKeyPrefix + hex.EncodeToString(sum[:])
}

func effectiveExactTTL(requestTTL int, defaultTTL int) int {
	if requestTTL == -1 {
		return defaultTTL
	}
	return requestTTL
}
