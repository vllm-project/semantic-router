package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"
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

func parseExactTimingFields(fields map[string]string) (time.Time, time.Time) {
	var storedAt, expiresAt time.Time
	if tsStr, ok := fields["timestamp"]; ok {
		var ts int64
		if _, err := fmt.Sscanf(tsStr, "%d", &ts); err == nil && ts > 0 {
			storedAt = time.Unix(ts, 0)
		}
	}
	if expStr, ok := fields["expires_at"]; ok {
		var exp int64
		if _, err := fmt.Sscanf(expStr, "%d", &exp); err == nil && exp > 0 {
			expiresAt = time.Unix(exp, 0)
		}
	}
	if expiresAt.IsZero() && !storedAt.IsZero() {
		if ttlStr, ok := fields["ttl_seconds"]; ok {
			var ttlSec int64
			if _, err := fmt.Sscanf(ttlStr, "%d", &ttlSec); err == nil && ttlSec > 0 {
				expiresAt = storedAt.Add(time.Duration(ttlSec) * time.Second)
			}
		}
	}
	return storedAt, expiresAt
}
