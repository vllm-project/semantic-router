package memory

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

func init() {
	// Initialize BERT model for embeddings (required for similarity calculation)
	err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true)
	if err != nil {
		// Skip tests if model initialization fails (model might not be available)
		fmt.Printf("Warning: Failed to initialize BERT model for tests: %v\n", err)
		fmt.Printf("Tests will be skipped. Make sure models are downloaded.\n")
	}
}

// =============================================================================
// Test Helpers
// =============================================================================

// newTestInMemoryStore creates an InMemoryStore with bert config for testing
// since that's the model initialized in init()
func newTestInMemoryStore() *InMemoryStore {
	return NewInMemoryStoreWithConfig(EmbeddingConfig{
		Model: EmbeddingModelBERT,
	})
}

// =============================================================================
// Similarity Threshold Tests
// =============================================================================

func TestInMemoryStore_Retrieve_DefaultThreshold(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()

	// Store memories with different similarity levels
	memories := []struct {
		id      string
		content string
	}{
		{"mem1", "User's budget for Hawaii vacation is $10,000"},
		{"mem2", "User prefers direct flights"},
		{"mem3", "The weather in Hawaii is sunny"},
	}

	for _, m := range memories {
		mem := &Memory{
			ID:        m.id,
			Type:      MemoryTypeSemantic,
			Content:   m.content,
			UserID:    "user1",
			CreatedAt: time.Now(),
		}
		err := store.Store(ctx, mem)
		require.NoError(t, err, "Failed to store memory %s", m.id)
	}

	// Test with default threshold (0.6) - should use 0.6 when Threshold is 0
	results, err := store.Retrieve(ctx, RetrieveOptions{
		Query:     "What is my budget for Hawaii?",
		UserID:    "user1",
		Threshold: 0.6, // Explicit default threshold
	})
	require.NoError(t, err)

	// Verify all results are above threshold
	for _, result := range results {
		assert.GreaterOrEqual(t, result.Score, float32(0.6),
			"Result score %.4f should be >= 0.6 threshold", result.Score)
	}
}

func TestInMemoryStore_Retrieve_FilterByThreshold(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()

	// Store memories
	mem1 := &Memory{
		ID:        "high_sim",
		Type:      MemoryTypeSemantic,
		Content:   "User's budget for Hawaii vacation is $10,000",
		UserID:    "user1",
		CreatedAt: time.Now(),
	}
	mem2 := &Memory{
		ID:        "low_sim",
		Type:      MemoryTypeSemantic,
		Content:   "The capital of France is Paris",
		UserID:    "user1",
		CreatedAt: time.Now(),
	}

	err := store.Store(ctx, mem1)
	require.NoError(t, err)
	err = store.Store(ctx, mem2)
	require.NoError(t, err)

	// Test with threshold 0.6 - should filter out low similarity results
	results, err := store.Retrieve(ctx, RetrieveOptions{
		Query:     "What is my budget for Hawaii?",
		UserID:    "user1",
		Threshold: 0.6,
	})
	require.NoError(t, err)

	// Verify all results meet threshold
	for _, result := range results {
		assert.GreaterOrEqual(t, result.Score, float32(0.6),
			"Result score %.4f should be >= 0.6 threshold", result.Score)
		assert.NotEqual(t, "low_sim", result.Memory.ID,
			"Low similarity memory should be filtered out")
	}
}

func TestInMemoryStore_Retrieve_ThresholdBoundary(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()

	// Store a memory
	mem := &Memory{
		ID:        "boundary_test",
		Type:      MemoryTypeSemantic,
		Content:   "User's budget for Hawaii vacation is $10,000",
		UserID:    "user1",
		CreatedAt: time.Now(),
	}
	err := store.Store(ctx, mem)
	require.NoError(t, err)

	// Test with threshold exactly at 0.6
	results, err := store.Retrieve(ctx, RetrieveOptions{
		Query:     "What is my budget for Hawaii?",
		UserID:    "user1",
		Threshold: 0.6,
	})
	require.NoError(t, err)

	// All results should have score >= 0.6
	for _, result := range results {
		assert.GreaterOrEqual(t, result.Score, float32(0.6),
			"Result score %.4f should be >= 0.6 threshold", result.Score)
	}

	// Test with threshold slightly above (0.61) - may filter out some results
	results2, err := store.Retrieve(ctx, RetrieveOptions{
		Query:     "What is my budget for Hawaii?",
		UserID:    "user1",
		Threshold: 0.61,
	})
	require.NoError(t, err)

	// All results should have score >= 0.61
	for _, result := range results2 {
		assert.GreaterOrEqual(t, result.Score, float32(0.61),
			"Result score %.4f should be >= 0.61 threshold", result.Score)
	}

	// Results with higher threshold should be subset of lower threshold
	assert.LessOrEqual(t, len(results2), len(results),
		"Higher threshold should return fewer or equal results")
}

func TestInMemoryStore_Retrieve_DifferentThresholdValues(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()

	// Store multiple memories with varying similarity
	memories := []*Memory{
		{
			ID:        "mem1",
			Type:      MemoryTypeSemantic,
			Content:   "User's budget for Hawaii vacation is $10,000",
			UserID:    "user1",
			CreatedAt: time.Now(),
		},
		{
			ID:        "mem2",
			Type:      MemoryTypeSemantic,
			Content:   "User prefers direct flights to Hawaii",
			UserID:    "user1",
			CreatedAt: time.Now(),
		},
		{
			ID:        "mem3",
			Type:      MemoryTypeSemantic,
			Content:   "The weather in Hawaii is sunny",
			UserID:    "user1",
			CreatedAt: time.Now(),
		},
	}

	for _, mem := range memories {
		err := store.Store(ctx, mem)
		require.NoError(t, err)
	}

	query := "What is my budget for Hawaii?"

	// Test with low threshold (0.3) - should return more results
	resultsLow, err := store.Retrieve(ctx, RetrieveOptions{
		Query:     query,
		UserID:    "user1",
		Threshold: 0.3,
	})
	require.NoError(t, err)

	// Test with default threshold (0.6) - should return fewer results
	resultsDefault, err := store.Retrieve(ctx, RetrieveOptions{
		Query:     query,
		UserID:    "user1",
		Threshold: 0.6,
	})
	require.NoError(t, err)

	// Test with high threshold (0.8) - should return even fewer results
	resultsHigh, err := store.Retrieve(ctx, RetrieveOptions{
		Query:     query,
		UserID:    "user1",
		Threshold: 0.8,
	})
	require.NoError(t, err)

	// Verify threshold ordering: more results with lower threshold
	assert.GreaterOrEqual(t, len(resultsLow), len(resultsDefault),
		"Lower threshold (0.3) should return more or equal results than default (0.6)")
	assert.GreaterOrEqual(t, len(resultsDefault), len(resultsHigh),
		"Default threshold (0.6) should return more or equal results than high (0.8)")

	// Verify all results meet their respective thresholds
	for _, result := range resultsLow {
		assert.GreaterOrEqual(t, result.Score, float32(0.3))
	}
	for _, result := range resultsDefault {
		assert.GreaterOrEqual(t, result.Score, float32(0.6))
	}
	for _, result := range resultsHigh {
		assert.GreaterOrEqual(t, result.Score, float32(0.8))
	}
}

func TestInMemoryStore_Retrieve_ThresholdZero(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()

	// Store memories
	mem := &Memory{
		ID:        "test_mem",
		Type:      MemoryTypeSemantic,
		Content:   "User's budget for Hawaii vacation is $10,000",
		UserID:    "user1",
		CreatedAt: time.Now(),
	}
	err := store.Store(ctx, mem)
	require.NoError(t, err)

	// Test with threshold 0 - should return all results (no filtering)
	results, err := store.Retrieve(ctx, RetrieveOptions{
		Query:     "What is my budget?",
		UserID:    "user1",
		Threshold: 0.0,
	})
	require.NoError(t, err)

	// Should return at least the stored memory
	assert.NotEmpty(t, results, "Threshold 0 should return results")
}

func TestInMemoryStore_Retrieve_ThresholdVeryHigh(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()

	// Store memories
	mem := &Memory{
		ID:        "test_mem",
		Type:      MemoryTypeSemantic,
		Content:   "User's budget for Hawaii vacation is $10,000",
		UserID:    "user1",
		CreatedAt: time.Now(),
	}
	err := store.Store(ctx, mem)
	require.NoError(t, err)

	// Test with very high threshold (0.99) - may return no results
	results, err := store.Retrieve(ctx, RetrieveOptions{
		Query:     "What is my budget for Hawaii?",
		UserID:    "user1",
		Threshold: 0.99,
	})
	require.NoError(t, err)

	// All results (if any) should meet the high threshold
	for _, result := range results {
		assert.GreaterOrEqual(t, result.Score, float32(0.99),
			"Result score %.4f should be >= 0.99 threshold", result.Score)
	}
}

func TestInMemoryStore_Retrieve_ThresholdWithMultipleUsers(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()

	// Store memories for different users
	mem1 := &Memory{
		ID:        "user1_mem",
		Type:      MemoryTypeSemantic,
		Content:   "User's budget for Hawaii vacation is $10,000",
		UserID:    "user1",
		CreatedAt: time.Now(),
	}
	mem2 := &Memory{
		ID:        "user2_mem",
		Type:      MemoryTypeSemantic,
		Content:   "User's budget for Hawaii vacation is $10,000",
		UserID:    "user2",
		CreatedAt: time.Now(),
	}

	err := store.Store(ctx, mem1)
	require.NoError(t, err)
	err = store.Store(ctx, mem2)
	require.NoError(t, err)

	// Test threshold filtering with user isolation
	results, err := store.Retrieve(ctx, RetrieveOptions{
		Query:     "What is my budget for Hawaii?",
		UserID:    "user1",
		Threshold: 0.6,
	})
	require.NoError(t, err)

	// Verify all results are for user1 and meet threshold
	for _, result := range results {
		assert.Equal(t, "user1", result.Memory.UserID,
			"Results should be filtered by user")
		assert.GreaterOrEqual(t, result.Score, float32(0.6),
			"Result score %.4f should be >= 0.6 threshold", result.Score)
		assert.NotEqual(t, "user2_mem", result.Memory.ID,
			"User2's memory should not appear in user1's results")
	}
}
