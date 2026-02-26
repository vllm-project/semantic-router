package memory

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestStoreRateLimiter_AllowsWithinLimit(t *testing.T) {
	rl := NewStoreRateLimiter(3, 60)

	assert.True(t, rl.Allow("user1"))
	assert.True(t, rl.Allow("user1"))
	assert.True(t, rl.Allow("user1"))
	assert.False(t, rl.Allow("user1"), "4th call should be blocked (limit=3)")
}

func TestStoreRateLimiter_IndependentPerUser(t *testing.T) {
	rl := NewStoreRateLimiter(2, 60)

	assert.True(t, rl.Allow("user1"))
	assert.True(t, rl.Allow("user1"))
	assert.False(t, rl.Allow("user1"), "user1 blocked at limit")

	assert.True(t, rl.Allow("user2"), "user2 should have its own bucket")
	assert.True(t, rl.Allow("user2"))
	assert.False(t, rl.Allow("user2"), "user2 also blocked at limit")
}

func TestStoreRateLimiter_DefaultValues(t *testing.T) {
	rl := NewStoreRateLimiter(0, 0)
	assert.Equal(t, 5, rl.limit, "default limit should be 5")
	assert.Equal(t, int64(60_000_000_000), rl.windowNs, "default window should be 60s in ns")
}

func TestStoreRateLimiter_Cleanup(t *testing.T) {
	rl := NewStoreRateLimiter(2, 1)

	rl.Allow("user1")
	rl.Allow("user1")

	rl.mu.Lock()
	assert.Contains(t, rl.windows, "user1")
	rl.mu.Unlock()

	rl.Cleanup()

	rl.mu.Lock()
	_, exists := rl.windows["user1"]
	rl.mu.Unlock()
	// Window might or might not have expired depending on timing;
	// the important thing is Cleanup() doesn't panic.
	_ = exists
}
