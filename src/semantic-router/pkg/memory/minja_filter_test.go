package memory

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func makeMemory(id, userID, content string, score float32) *RetrieveResult {
	return &RetrieveResult{
		Memory: &Memory{ID: id, UserID: userID, Content: content},
		Score:  score,
	}
}

func defaultMinjaConfig() config.MinjaDefenseConfig {
	return config.MinjaDefenseConfig{}
}

func TestMinjaFilter_NilFilter_PassesAll(t *testing.T) {
	disabled := false
	cfg := config.MinjaDefenseConfig{Enabled: &disabled}
	f := NewMinjaFilter(cfg, "user1")
	assert.Nil(t, f)

	memories := []*RetrieveResult{makeMemory("1", "user1", "hello", 0.9)}
	var nilFilter *MinjaFilter
	result := nilFilter.Filter(memories)
	assert.Equal(t, memories, result)
}

func TestMinjaFilter_OwnedMemories_PassThrough(t *testing.T) {
	f := NewMinjaFilter(defaultMinjaConfig(), "user1")
	require.NotNil(t, f)

	memories := []*RetrieveResult{
		makeMemory("1", "user1", "My preference is dark mode", 0.95),
		makeMemory("2", "user1", "I use Go for backend", 0.88),
		makeMemory("3", "user1", "Budget is $5000", 0.80),
	}

	result := f.Filter(memories)
	assert.Len(t, result, 3, "all owned memories should pass")
}

func TestMinjaFilter_SharedMemory_HighSimilarity_Passes(t *testing.T) {
	f := NewMinjaFilter(defaultMinjaConfig(), "user1")

	memories := []*RetrieveResult{
		makeMemory("1", "user1", "own fact", 0.95),
		makeMemory("2", "user2", "shared team fact", 0.90),
	}

	result := f.Filter(memories)
	assert.Len(t, result, 2, "shared memory with score 0.90 >= 0.85 should pass")
}

func TestMinjaFilter_SharedMemory_LowSimilarity_Filtered(t *testing.T) {
	f := NewMinjaFilter(defaultMinjaConfig(), "user1")

	memories := []*RetrieveResult{
		makeMemory("1", "user1", "own fact", 0.95),
		makeMemory("2", "user2", "attacker fact", 0.75),
	}

	result := f.Filter(memories)
	assert.Len(t, result, 1, "shared memory with score 0.75 < 0.85 should be filtered")
	assert.Equal(t, "1", result[0].Memory.ID)
}

func TestMinjaFilter_SharedMemory_CapEnforced(t *testing.T) {
	cfg := defaultMinjaConfig()
	cfg.MaxSharedMemoriesPerRequest = 2
	cfg.MaxSharedPerCreator = 5
	f := NewMinjaFilter(cfg, "user1")

	memories := []*RetrieveResult{
		makeMemory("1", "user2", "shared fact 1", 0.95),
		makeMemory("2", "user3", "shared fact 2", 0.93),
		makeMemory("3", "user4", "shared fact 3", 0.91),
		makeMemory("4", "user5", "shared fact 4", 0.89),
	}

	result := f.Filter(memories)
	assert.Len(t, result, 2, "only 2 shared memories should pass (cap=2)")
}

func TestMinjaFilter_PerCreatorDiversity(t *testing.T) {
	cfg := defaultMinjaConfig()
	cfg.MaxSharedMemoriesPerRequest = 10
	cfg.MaxSharedPerCreator = 1
	f := NewMinjaFilter(cfg, "user1")

	memories := []*RetrieveResult{
		makeMemory("1", "attacker", "poison fact 1", 0.95),
		makeMemory("2", "attacker", "poison fact 2", 0.93),
		makeMemory("3", "attacker", "poison fact 3", 0.91),
		makeMemory("4", "legit_user", "team fact", 0.90),
	}

	result := f.Filter(memories)
	assert.Len(t, result, 2, "1 from attacker (per-creator=1) + 1 from legit_user")

	creators := map[string]int{}
	for _, r := range result {
		creators[r.Memory.UserID]++
	}
	assert.Equal(t, 1, creators["attacker"], "attacker capped at 1")
	assert.Equal(t, 1, creators["legit_user"], "legit user gets 1")
}

func TestMinjaFilter_MixedOwnedAndShared(t *testing.T) {
	cfg := defaultMinjaConfig()
	cfg.MaxSharedMemoriesPerRequest = 2
	f := NewMinjaFilter(cfg, "user1")

	memories := []*RetrieveResult{
		makeMemory("1", "user1", "own fact 1", 0.98),
		makeMemory("2", "user2", "shared fact 1", 0.92),
		makeMemory("3", "user1", "own fact 2", 0.90),
		makeMemory("4", "user3", "shared fact 2", 0.88),
		makeMemory("5", "user4", "shared fact 3", 0.86),
	}

	result := f.Filter(memories)
	assert.Len(t, result, 4, "2 owned + 2 shared (cap=2); 3rd shared filtered")

	ownCount := 0
	sharedCount := 0
	for _, r := range result {
		if r.Memory.UserID == "user1" {
			ownCount++
		} else {
			sharedCount++
		}
	}
	assert.Equal(t, 2, ownCount)
	assert.Equal(t, 2, sharedCount)
}

func TestMinjaFilter_EmptyInput(t *testing.T) {
	f := NewMinjaFilter(defaultMinjaConfig(), "user1")
	result := f.Filter(nil)
	assert.Empty(t, result)

	result = f.Filter([]*RetrieveResult{})
	assert.Empty(t, result)
}

func TestMinjaFilter_NoUserID_AllPassThrough(t *testing.T) {
	f := NewMinjaFilter(defaultMinjaConfig(), "")

	memories := []*RetrieveResult{
		makeMemory("1", "user1", "fact 1", 0.95),
		makeMemory("2", "user2", "fact 2", 0.70),
	}

	result := f.Filter(memories)
	assert.Len(t, result, 2, "without requesting user ID, all memories treated as owned")
}

func TestMinjaFilter_CustomConfig(t *testing.T) {
	cfg := config.MinjaDefenseConfig{
		SharedMemoryMinSimilarity:   0.90,
		MaxSharedMemoriesPerRequest: 1,
		MaxSharedPerCreator:         1,
	}
	f := NewMinjaFilter(cfg, "user1")

	memories := []*RetrieveResult{
		makeMemory("1", "user2", "high sim fact", 0.92),
		makeMemory("2", "user3", "also high sim", 0.91),
	}

	result := f.Filter(memories)
	assert.Len(t, result, 1, "only 1 shared memory allowed (cap=1)")
}
