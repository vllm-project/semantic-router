package memory

import (
	"context"
	"fmt"
	"testing"
	"time"

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

// newTestStore creates an InMemoryStore with bert config for testing
// since that's the model initialized in init()
func newTestStore() *InMemoryStore {
	return NewInMemoryStoreWithConfig(EmbeddingConfig{
		Model: EmbeddingModelBERT,
	})
}

func TestDeduplicationLogic(t *testing.T) {
	// Create in-memory store
	store := newTestStore()
	ctx := context.Background()
	config := DefaultDeduplicationConfig()

	// Test cases with different similarity levels
	testCases := []struct {
		name     string
		content1 string
		content2 string
		expected string // "update" or "create"
	}{
		{
			name:     "Exact duplicate - Should UPDATE",
			content1: "User's budget for Hawaii vacation is $10,000",
			content2: "User's budget for Hawaii vacation is $10,000",
			expected: "update",
		},
		{
			name:     "Very similar wording - Should UPDATE",
			content1: "My budget for the Hawaii trip is $10,000",
			content2: "My budget for Hawaii vacation is $10,000",
			expected: "update",
		},
		{
			name:     "Similar with value change - Should UPDATE",
			content1: "User's budget for Hawaii vacation is $10,000",
			content2: "User's budget for Hawaii trip is now $15,000",
			expected: "update",
		},
		{
			name:     "Different topic - Should CREATE",
			content1: "User's budget for Hawaii vacation is $10,000",
			content2: "User likes chocolate ice cream",
			expected: "create",
		},
		{
			name:     "Related but different - Should CREATE (gray zone)",
			content1: "User's budget for Hawaii vacation is $10,000",
			content2: "User prefers direct flights to Hawaii",
			expected: "create",
		},
	}

	for i, tc := range testCases {
		t.Run(fmt.Sprintf("Test_%d_%s", i+1, tc.name), func(t *testing.T) {
			// Store first memory
			mem1 := &Memory{
				ID:        fmt.Sprintf("mem_%d_1", i),
				Type:      MemoryTypeSemantic,
				Content:   tc.content1,
				UserID:    "test_user",
				CreatedAt: time.Now(),
			}

			if err := store.Store(ctx, mem1); err != nil {
				t.Fatalf("Failed to store first memory: %v", err)
			}

			// Check deduplication for second content
			result := CheckDeduplication(ctx, store, "test_user", tc.content2, MemoryTypeSemantic, config)

			t.Logf("Content 1: %s", tc.content1)
			t.Logf("Content 2: %s", tc.content2)
			t.Logf("Similarity: %.4f", result.Similarity)
			t.Logf("Action: %s (expected: %s)", result.Action, tc.expected)

			if result.Action != tc.expected {
				t.Errorf("Expected action %s, got %s (similarity=%.4f)", tc.expected, result.Action, result.Similarity)
			}

			if result.Action == "update" && result.ExistingMemory == nil {
				t.Error("Update action but no existing memory found")
			}
		})
	}
}

func TestDeduplicationMultipleMemories(t *testing.T) {
	store := newTestStore()
	ctx := context.Background()
	config := DefaultDeduplicationConfig()

	// Store multiple memories for same user
	memories := []string{
		"User's budget for Hawaii vacation is $10,000",
		"User prefers direct flights",
		"User likes beach hotels",
	}

	for i, content := range memories {
		mem := &Memory{
			ID:        fmt.Sprintf("multi_mem_%d", i),
			Type:      MemoryTypeSemantic,
			Content:   content,
			UserID:    "test_user_2",
			CreatedAt: time.Now(),
		}
		if err := store.Store(ctx, mem); err != nil {
			t.Fatalf("Failed to store memory: %v", err)
		}
	}

	// Test deduplication with similar content
	testContent := "User's budget for Hawaii trip is $10,000"
	result := CheckDeduplication(ctx, store, "test_user_2", testContent, MemoryTypeSemantic, config)

	t.Logf("Testing: %s", testContent)
	t.Logf("Similarity: %.4f", result.Similarity)
	t.Logf("Action: %s", result.Action)
	if result.ExistingMemory != nil {
		t.Logf("Matched with: %s", result.ExistingMemory.Content)
	}

	// Should find the first memory (budget) and update it
	if result.Action != "update" {
		t.Errorf("Expected update action, got %s", result.Action)
	}
	if result.ExistingMemory == nil {
		t.Error("Expected to find existing memory")
	}
	if result.ExistingMemory.Content != memories[0] {
		t.Errorf("Expected to match first memory, got: %s", result.ExistingMemory.Content)
	}
}

func TestDeduplicationUserIsolation(t *testing.T) {
	store := newTestStore()
	ctx := context.Background()
	config := DefaultDeduplicationConfig()

	// Store memory for user1
	mem1 := &Memory{
		ID:        "user1_mem",
		Type:      MemoryTypeSemantic,
		Content:   "User's budget for Hawaii vacation is $10,000",
		UserID:    "user1",
		CreatedAt: time.Now(),
	}
	if err := store.Store(ctx, mem1); err != nil {
		t.Fatalf("Failed to store memory: %v", err)
	}

	// Try to find similar for user2 (should not find)
	result := CheckDeduplication(ctx, store, "user2", "User's budget for Hawaii vacation is $10,000", MemoryTypeSemantic, config)

	if result.Action != "create" {
		t.Errorf("Expected create action (user isolation), got %s", result.Action)
	}
	if result.ExistingMemory != nil {
		t.Error("Should not find memory from different user")
	}
}

func TestDeduplicationTypeIsolation(t *testing.T) {
	store := newTestStore()
	ctx := context.Background()
	config := DefaultDeduplicationConfig()

	// Store semantic memory
	mem1 := &Memory{
		ID:        "semantic_mem",
		Type:      MemoryTypeSemantic,
		Content:   "User's budget for Hawaii vacation is $10,000",
		UserID:    "test_user",
		CreatedAt: time.Now(),
	}
	if err := store.Store(ctx, mem1); err != nil {
		t.Fatalf("Failed to store memory: %v", err)
	}

	// Try to find similar for procedural type (should not find)
	result := CheckDeduplication(ctx, store, "test_user", "User's budget for Hawaii vacation is $10,000", MemoryTypeProcedural, config)

	if result.Action != "create" {
		t.Errorf("Expected create action (type isolation), got %s", result.Action)
	}
	if result.ExistingMemory != nil {
		t.Error("Should not find memory of different type")
	}
}

// =============================================================================
// Threshold Value Tests - Explicit verification of 0.7 and 0.9 thresholds
// =============================================================================

func TestDeduplicationDefaultThresholdValues(t *testing.T) {
	store := newTestStore()
	ctx := context.Background()
	config := DefaultDeduplicationConfig()

	// Verify default threshold values
	if config.SearchThreshold != 0.7 {
		t.Errorf("Expected default SearchThreshold to be 0.7, got %.2f", config.SearchThreshold)
	}
	if config.UpdateThreshold != 0.9 {
		t.Errorf("Expected default UpdateThreshold to be 0.9, got %.2f", config.UpdateThreshold)
	}

	// Store a memory for testing
	mem := &Memory{
		ID:        "threshold_test_mem",
		Type:      MemoryTypeSemantic,
		Content:   "User's budget for Hawaii vacation is $10,000",
		UserID:    "threshold_user",
		CreatedAt: time.Now(),
	}
	if err := store.Store(ctx, mem); err != nil {
		t.Fatalf("Failed to store memory: %v", err)
	}

	// Test with exact duplicate (should have high similarity > 0.9)
	exactDuplicate := "User's budget for Hawaii vacation is $10,000"
	result := CheckDeduplication(ctx, store, "threshold_user", exactDuplicate, MemoryTypeSemantic, config)

	t.Logf("Exact duplicate test - Similarity: %.4f, Action: %s", result.Similarity, result.Action)

	// Should UPDATE if similarity > 0.9
	if result.Similarity > config.UpdateThreshold {
		if result.Action != "update" {
			t.Errorf("Expected UPDATE action for similarity %.4f > %.2f, got %s",
				result.Similarity, config.UpdateThreshold, result.Action)
		}
		if result.ExistingMemory == nil {
			t.Error("Expected existing memory for UPDATE action")
		}
	}
}

func TestDeduplicationSearchThresholdBehavior(t *testing.T) {
	store := newTestStore()
	ctx := context.Background()

	// Test with custom SearchThreshold to verify filtering behavior
	config := DeduplicationConfig{
		SearchThreshold: 0.7,
		UpdateThreshold: 0.9,
	}

	// Store a memory
	mem := &Memory{
		ID:        "search_threshold_mem",
		Type:      MemoryTypeSemantic,
		Content:   "User's budget for Hawaii vacation is $10,000",
		UserID:    "search_user",
		CreatedAt: time.Now(),
	}
	if err := store.Store(ctx, mem); err != nil {
		t.Fatalf("Failed to store memory: %v", err)
	}

	// Test with very different content (should be below SearchThreshold)
	veryDifferent := "The capital of France is Paris"
	result := CheckDeduplication(ctx, store, "search_user", veryDifferent, MemoryTypeSemantic, config)

	t.Logf("Very different content - Similarity: %.4f, Action: %s", result.Similarity, result.Action)

	// Should CREATE if similarity < SearchThreshold (0.7)
	if result.Similarity < config.SearchThreshold {
		if result.Action != "create" {
			t.Errorf("Expected CREATE action for similarity %.4f < SearchThreshold %.2f, got %s",
				result.Similarity, config.SearchThreshold, result.Action)
		}
		if result.ExistingMemory != nil {
			t.Error("Should not find existing memory when similarity < SearchThreshold")
		}
	}

	// Verify FindSimilar respects SearchThreshold
	existing, similarity := FindSimilar(ctx, store, "search_user", veryDifferent, MemoryTypeSemantic, config)
	if existing != nil && similarity < config.SearchThreshold {
		t.Errorf("FindSimilar should not return results below SearchThreshold %.2f, got similarity %.4f",
			config.SearchThreshold, similarity)
	}
}

func TestDeduplicationUpdateThresholdBehavior(t *testing.T) {
	store := newTestStore()
	ctx := context.Background()

	config := DeduplicationConfig{
		SearchThreshold: 0.7,
		UpdateThreshold: 0.9,
	}

	// Store a memory
	mem := &Memory{
		ID:        "update_threshold_mem",
		Type:      MemoryTypeSemantic,
		Content:   "User's budget for Hawaii vacation is $10,000",
		UserID:    "update_user",
		CreatedAt: time.Now(),
	}
	if err := store.Store(ctx, mem); err != nil {
		t.Fatalf("Failed to store memory: %v", err)
	}

	// Test with very similar content (should be > UpdateThreshold)
	verySimilar := "My budget for Hawaii vacation is $10,000"
	result := CheckDeduplication(ctx, store, "update_user", verySimilar, MemoryTypeSemantic, config)

	t.Logf("Very similar content - Similarity: %.4f, Action: %s", result.Similarity, result.Action)

	// Should UPDATE if similarity > UpdateThreshold (0.9)
	if result.Similarity > config.UpdateThreshold {
		if result.Action != "update" {
			t.Errorf("Expected UPDATE action for similarity %.4f > UpdateThreshold %.2f, got %s",
				result.Similarity, config.UpdateThreshold, result.Action)
		}
		if result.ExistingMemory == nil {
			t.Error("Expected existing memory for UPDATE action")
		}
	}
}

// =============================================================================
// Boundary Condition Tests - Testing at exact threshold values
// =============================================================================

func TestDeduplicationBoundaryAtSearchThreshold(t *testing.T) {
	store := newTestStore()
	ctx := context.Background()

	// Test with SearchThreshold set to a specific value
	// We'll use a custom config and verify behavior at the boundary
	config := DeduplicationConfig{
		SearchThreshold: 0.7,
		UpdateThreshold: 0.9,
	}

	// Store a memory
	mem := &Memory{
		ID:        "boundary_search_mem",
		Type:      MemoryTypeSemantic,
		Content:   "User's budget for Hawaii vacation is $10,000",
		UserID:    "boundary_user",
		CreatedAt: time.Now(),
	}
	if err := store.Store(ctx, mem); err != nil {
		t.Fatalf("Failed to store memory: %v", err)
	}

	// Test with content that might be at or near SearchThreshold boundary
	// Note: We can't control exact similarity, but we can verify the logic
	relatedContent := "User prefers direct flights to Hawaii"
	result := CheckDeduplication(ctx, store, "boundary_user", relatedContent, MemoryTypeSemantic, config)

	t.Logf("Boundary test at SearchThreshold - Similarity: %.4f, SearchThreshold: %.2f, Action: %s",
		result.Similarity, config.SearchThreshold, result.Action)

	// Verify behavior: if similarity >= SearchThreshold, should find memory
	// If similarity < SearchThreshold, should CREATE with no existing memory
	if result.Similarity < config.SearchThreshold {
		if result.Action != "create" {
			t.Errorf("Expected CREATE for similarity %.4f < SearchThreshold %.2f, got %s",
				result.Similarity, config.SearchThreshold, result.Action)
		}
		if result.ExistingMemory != nil {
			t.Error("Should not have existing memory when similarity < SearchThreshold")
		}
	} else {
		// Similarity >= SearchThreshold, should find memory
		if result.ExistingMemory == nil {
			t.Error("Should find existing memory when similarity >= SearchThreshold")
		}
		// Action depends on UpdateThreshold
		if result.Similarity > config.UpdateThreshold {
			if result.Action != "update" {
				t.Errorf("Expected UPDATE for similarity %.4f > UpdateThreshold %.2f, got %s",
					result.Similarity, config.UpdateThreshold, result.Action)
			}
		} else {
			// In gray zone (SearchThreshold <= similarity <= UpdateThreshold)
			if result.Action != "create" {
				t.Errorf("Expected CREATE for similarity %.4f in gray zone, got %s",
					result.Similarity, result.Action)
			}
		}
	}
}

func TestDeduplicationBoundaryAtUpdateThreshold(t *testing.T) {
	store := newTestStore()
	ctx := context.Background()

	config := DeduplicationConfig{
		SearchThreshold: 0.7,
		UpdateThreshold: 0.9,
	}

	// Store a memory
	mem := &Memory{
		ID:        "boundary_update_mem",
		Type:      MemoryTypeSemantic,
		Content:   "User's budget for Hawaii vacation is $10,000",
		UserID:    "boundary_update_user",
		CreatedAt: time.Now(),
	}
	if err := store.Store(ctx, mem); err != nil {
		t.Fatalf("Failed to store memory: %v", err)
	}

	// Test with very similar content (should be near or above UpdateThreshold)
	verySimilar := "User's budget for Hawaii trip is $10,000"
	result := CheckDeduplication(ctx, store, "boundary_update_user", verySimilar, MemoryTypeSemantic, config)

	t.Logf("Boundary test at UpdateThreshold - Similarity: %.4f, UpdateThreshold: %.2f, Action: %s",
		result.Similarity, config.UpdateThreshold, result.Action)

	// Verify behavior at UpdateThreshold boundary
	if result.Similarity > config.UpdateThreshold {
		// Above UpdateThreshold: should UPDATE
		if result.Action != "update" {
			t.Errorf("Expected UPDATE for similarity %.4f > UpdateThreshold %.2f, got %s",
				result.Similarity, config.UpdateThreshold, result.Action)
		}
		if result.ExistingMemory == nil {
			t.Error("Expected existing memory for UPDATE action")
		}
	} else if result.Similarity >= config.SearchThreshold {
		// In gray zone (SearchThreshold <= similarity <= UpdateThreshold): should CREATE
		if result.Action != "create" {
			t.Errorf("Expected CREATE for similarity %.4f in gray zone [%.2f, %.2f], got %s",
				result.Similarity, config.SearchThreshold, config.UpdateThreshold, result.Action)
		}
		if result.ExistingMemory == nil {
			t.Error("Should find existing memory in gray zone")
		}
	}
}

// =============================================================================
// Custom Threshold Configuration Tests
// =============================================================================

func TestDeduplicationCustomThresholdConfig(t *testing.T) {
	store := newTestStore()
	ctx := context.Background()

	// Test with custom thresholds
	customConfig := DeduplicationConfig{
		SearchThreshold: 0.6,  // Lower than default
		UpdateThreshold: 0.85, // Lower than default
	}

	// Store a memory
	mem := &Memory{
		ID:        "custom_threshold_mem",
		Type:      MemoryTypeSemantic,
		Content:   "User's budget for Hawaii vacation is $10,000",
		UserID:    "custom_user",
		CreatedAt: time.Now(),
	}
	if err := store.Store(ctx, mem); err != nil {
		t.Fatalf("Failed to store memory: %v", err)
	}

	// Test with moderately similar content
	moderatelySimilar := "User's budget for Hawaii trip is $10,000"
	result := CheckDeduplication(ctx, store, "custom_user", moderatelySimilar, MemoryTypeSemantic, customConfig)

	t.Logf("Custom threshold test - Similarity: %.4f, SearchThreshold: %.2f, UpdateThreshold: %.2f, Action: %s",
		result.Similarity, customConfig.SearchThreshold, customConfig.UpdateThreshold, result.Action)

	// Verify custom thresholds are respected
	if result.Similarity < customConfig.SearchThreshold {
		if result.Action != "create" {
			t.Errorf("Expected CREATE for similarity %.4f < custom SearchThreshold %.2f, got %s",
				result.Similarity, customConfig.SearchThreshold, result.Action)
		}
	} else if result.Similarity > customConfig.UpdateThreshold {
		if result.Action != "update" {
			t.Errorf("Expected UPDATE for similarity %.4f > custom UpdateThreshold %.2f, got %s",
				result.Similarity, customConfig.UpdateThreshold, result.Action)
		}
	} else {
		// In gray zone with custom thresholds
		if result.Action != "create" {
			t.Errorf("Expected CREATE for similarity %.4f in custom gray zone, got %s",
				result.Similarity, result.Action)
		}
	}
}

func TestDeduplicationCustomThresholdVeryStrict(t *testing.T) {
	store := newTestStore()
	ctx := context.Background()

	// Test with very strict thresholds
	strictConfig := DeduplicationConfig{
		SearchThreshold: 0.85, // Higher than default
		UpdateThreshold: 0.95, // Higher than default
	}

	// Store a memory
	mem := &Memory{
		ID:        "strict_threshold_mem",
		Type:      MemoryTypeSemantic,
		Content:   "User's budget for Hawaii vacation is $10,000",
		UserID:    "strict_user",
		CreatedAt: time.Now(),
	}
	if err := store.Store(ctx, mem); err != nil {
		t.Fatalf("Failed to store memory: %v", err)
	}

	// Test with similar but not identical content
	similar := "My budget for Hawaii vacation is $10,000"
	result := CheckDeduplication(ctx, store, "strict_user", similar, MemoryTypeSemantic, strictConfig)

	t.Logf("Strict threshold test - Similarity: %.4f, SearchThreshold: %.2f, UpdateThreshold: %.2f, Action: %s",
		result.Similarity, strictConfig.SearchThreshold, strictConfig.UpdateThreshold, result.Action)

	// With strict thresholds, more content will fall below SearchThreshold
	if result.Similarity < strictConfig.SearchThreshold {
		if result.Action != "create" {
			t.Errorf("Expected CREATE for similarity %.4f < strict SearchThreshold %.2f, got %s",
				result.Similarity, strictConfig.SearchThreshold, result.Action)
		}
	}
}

func TestDeduplicationCustomThresholdVeryLenient(t *testing.T) {
	store := newTestStore()
	ctx := context.Background()

	// Test with very lenient thresholds
	lenientConfig := DeduplicationConfig{
		SearchThreshold: 0.5, // Lower than default
		UpdateThreshold: 0.7, // Lower than default
	}

	// Store a memory
	mem := &Memory{
		ID:        "lenient_threshold_mem",
		Type:      MemoryTypeSemantic,
		Content:   "User's budget for Hawaii vacation is $10,000",
		UserID:    "lenient_user",
		CreatedAt: time.Now(),
	}
	if err := store.Store(ctx, mem); err != nil {
		t.Fatalf("Failed to store memory: %v", err)
	}

	// Test with related but different content
	related := "User prefers direct flights to Hawaii"
	result := CheckDeduplication(ctx, store, "lenient_user", related, MemoryTypeSemantic, lenientConfig)

	t.Logf("Lenient threshold test - Similarity: %.4f, SearchThreshold: %.2f, UpdateThreshold: %.2f, Action: %s",
		result.Similarity, lenientConfig.SearchThreshold, lenientConfig.UpdateThreshold, result.Action)

	// With lenient thresholds, more content will be found and potentially updated
	if result.Similarity >= lenientConfig.SearchThreshold {
		if result.ExistingMemory == nil {
			t.Error("Should find existing memory with lenient SearchThreshold")
		}
		if result.Similarity > lenientConfig.UpdateThreshold {
			if result.Action != "update" {
				t.Errorf("Expected UPDATE for similarity %.4f > lenient UpdateThreshold %.2f, got %s",
					result.Similarity, lenientConfig.UpdateThreshold, result.Action)
			}
		}
	}
}

// =============================================================================
// SearchThreshold Minimum Similarity Tests
// =============================================================================

func TestDeduplicationSearchThresholdMinimumSimilarity(t *testing.T) {
	store := newTestStore()
	ctx := context.Background()

	// Store a memory
	mem := &Memory{
		ID:        "min_similarity_mem",
		Type:      MemoryTypeSemantic,
		Content:   "User's budget for Hawaii vacation is $10,000",
		UserID:    "min_sim_user",
		CreatedAt: time.Now(),
	}
	if err := store.Store(ctx, mem); err != nil {
		t.Fatalf("Failed to store memory: %v", err)
	}

	// Test FindSimilar directly with different SearchThreshold values
	testCases := []struct {
		name            string
		content         string
		searchThreshold float32
		shouldFind      bool
	}{
		{
			name:            "High threshold - should not find low similarity",
			content:         "The weather is nice today",
			searchThreshold: 0.8,
			shouldFind:      false,
		},
		{
			name:            "Default threshold - may find if similarity >= 0.7",
			content:         "User prefers direct flights",
			searchThreshold: 0.7,
			shouldFind:      true, // May find if similarity is high enough
		},
		{
			name:            "Low threshold - should find more results",
			content:         "User prefers direct flights",
			searchThreshold: 0.5,
			shouldFind:      true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			testConfig := DeduplicationConfig{
				SearchThreshold: tc.searchThreshold,
				UpdateThreshold: 0.9,
			}

			existing, similarity := FindSimilar(ctx, store, "min_sim_user", tc.content, MemoryTypeSemantic, testConfig)

			t.Logf("SearchThreshold: %.2f, Similarity: %.4f, Found: %v",
				tc.searchThreshold, similarity, existing != nil)

			// Verify that if we found a result, similarity >= SearchThreshold
			if existing != nil {
				if similarity < testConfig.SearchThreshold {
					t.Errorf("Found memory with similarity %.4f < SearchThreshold %.2f",
						similarity, testConfig.SearchThreshold)
				}
			} else {
				// If not found, similarity should be 0 or below threshold
				if similarity > 0 && similarity >= testConfig.SearchThreshold {
					t.Errorf("No memory found but similarity %.4f >= SearchThreshold %.2f",
						similarity, testConfig.SearchThreshold)
				}
			}
		})
	}
}

func TestDeduplicationThresholdRanges(t *testing.T) {
	store := newTestStore()
	ctx := context.Background()

	// Store a memory
	mem := &Memory{
		ID:        "range_test_mem",
		Type:      MemoryTypeSemantic,
		Content:   "User's budget for Hawaii vacation is $10,000",
		UserID:    "range_user",
		CreatedAt: time.Now(),
	}
	if err := store.Store(ctx, mem); err != nil {
		t.Fatalf("Failed to store memory: %v", err)
	}

	// Test different content types to verify threshold ranges
	testCases := []struct {
		name          string
		content       string
		expectedRange string // "below_search", "gray_zone", "above_update"
	}{
		{
			name:          "Exact duplicate",
			content:       "User's budget for Hawaii vacation is $10,000",
			expectedRange: "above_update", // Should be > 0.9
		},
		{
			name:          "Very similar",
			content:       "My budget for Hawaii vacation is $10,000",
			expectedRange: "gray_zone", // May be in gray zone (0.7-0.9) depending on exact wording
		},
		{
			name:          "Related content",
			content:       "User prefers direct flights to Hawaii",
			expectedRange: "gray_zone", // Should be between 0.7-0.9
		},
		{
			name:          "Different topic",
			content:       "User likes chocolate ice cream",
			expectedRange: "below_search", // Should be < 0.7
		},
	}

	config := DefaultDeduplicationConfig()

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := CheckDeduplication(ctx, store, "range_user", tc.content, MemoryTypeSemantic, config)

			t.Logf("Content: %s", tc.content)
			t.Logf("Similarity: %.4f, Expected range: %s", result.Similarity, tc.expectedRange)

			switch tc.expectedRange {
			case "below_search":
				if result.Similarity >= config.SearchThreshold {
					t.Errorf("Expected similarity < %.2f, got %.4f",
						config.SearchThreshold, result.Similarity)
				}
				if result.Action != "create" {
					t.Errorf("Expected CREATE action, got %s", result.Action)
				}
				if result.ExistingMemory != nil {
					t.Error("Should not find existing memory below SearchThreshold")
				}

			case "gray_zone":
				if result.Similarity < config.SearchThreshold || result.Similarity > config.UpdateThreshold {
					t.Errorf("Expected similarity in gray zone [%.2f, %.2f], got %.4f",
						config.SearchThreshold, config.UpdateThreshold, result.Similarity)
				}
				if result.Action != "create" {
					t.Errorf("Expected CREATE action in gray zone, got %s", result.Action)
				}
				if result.ExistingMemory == nil {
					t.Error("Should find existing memory in gray zone")
				}

			case "above_update":
				if result.Similarity <= config.UpdateThreshold {
					t.Errorf("Expected similarity > %.2f, got %.4f",
						config.UpdateThreshold, result.Similarity)
				}
				if result.Action != "update" {
					t.Errorf("Expected UPDATE action, got %s", result.Action)
				}
				if result.ExistingMemory == nil {
					t.Error("Should find existing memory above UpdateThreshold")
				}
			}
		})
	}
}
