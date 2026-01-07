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

func TestDeduplicationLogic(t *testing.T) {
	// Create in-memory store
	store := NewInMemoryStore()
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
	store := NewInMemoryStore()
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
	store := NewInMemoryStore()
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
	store := NewInMemoryStore()
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
