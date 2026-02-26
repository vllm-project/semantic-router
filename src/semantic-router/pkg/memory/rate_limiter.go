package memory

import (
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// StoreRateLimiter enforces per-user memory creation rate limits to counter the
// Progressive Shortening Strategy (PSS) described in MINJA (arXiv:2503.03704,
// Section 4.2). PSS requires the attacker to submit many sequential queries that
// each generate a malicious record. Rate limiting caps how many memories a single
// user can create per time window, making PSS impractical.
type StoreRateLimiter struct {
	mu       sync.Mutex
	windows  map[string]*rateBucket
	limit    int
	windowNs int64
}

type rateBucket struct {
	count       int
	windowStart time.Time
}

// NewStoreRateLimiter creates a rate limiter with the given limit and window.
func NewStoreRateLimiter(limit int, windowSeconds int) *StoreRateLimiter {
	if limit <= 0 {
		limit = 5
	}
	if windowSeconds <= 0 {
		windowSeconds = 60
	}
	return &StoreRateLimiter{
		windows:  make(map[string]*rateBucket),
		limit:    limit,
		windowNs: int64(windowSeconds) * int64(time.Second),
	}
}

// Allow returns true if the user is within their rate limit for memory creation.
// If the window has expired, it resets. Thread-safe.
func (rl *StoreRateLimiter) Allow(userID string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	bucket, exists := rl.windows[userID]

	if !exists || now.Sub(bucket.windowStart) >= time.Duration(rl.windowNs) {
		rl.windows[userID] = &rateBucket{count: 1, windowStart: now}
		return true
	}

	if bucket.count >= rl.limit {
		logging.Warnf("StoreRateLimiter: user %s exceeded rate limit (%d/%ds), blocking memory creation",
			userID, rl.limit, rl.windowNs/int64(time.Second))
		RecordRateLimitBlocked()
		return false
	}

	bucket.count++
	return true
}

// Cleanup removes expired entries to prevent unbounded growth. Call periodically.
func (rl *StoreRateLimiter) Cleanup() {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	for userID, bucket := range rl.windows {
		if now.Sub(bucket.windowStart) >= time.Duration(rl.windowNs) {
			delete(rl.windows, userID)
		}
	}
}
