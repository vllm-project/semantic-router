package cache

import (
	"fmt"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// TestInMemoryCacheIntegration tests the in-memory cache integration
func TestInMemoryCacheIntegration(t *testing.T) {
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		t.Skipf("Failed to initialize BERT model: %v", err)
	}

	cache := NewInMemoryCache(InMemoryCacheOptions{
		Enabled:             true,
		MaxEntries:          2,
		SimilarityThreshold: 0.9,
		EvictionPolicy:      "lfu",
		TTLSeconds:          0,
	})

	t.Run("InMemoryCacheIntegration", func(t *testing.T) {
		// Step 1: Add first entry
		err := cache.AddEntry("req1", "test-model", "Hello world",
			[]byte("request1"), []byte("response1"))
		if err != nil {
			t.Fatalf("Failed to add first entry: %v", err)
		}

		// Step 2: Add second entry (cache at capacity)
		err = cache.AddEntry("req2", "test-model", "Good morning",
			[]byte("request2"), []byte("response2"))
		if err != nil {
			t.Fatalf("Failed to add second entry: %v", err)
		}

		// Verify
		if len(cache.entries) != 2 {
			t.Errorf("Expected 2 entries, got %d", len(cache.entries))
		}
		if cache.entries[1].RequestID != "req2" {
			t.Errorf("Expected req2 to be the second entry, got %s", cache.entries[1].RequestID)
		}

		// Step 3: Access first entry multiple times to increase its frequency
		for range 2 {
			responseBody, found, err := cache.FindSimilar("test-model", "Hello world")
			if err != nil {
				t.Logf("FindSimilar failed (expected due to high threshold): %v", err)
			}
			if !found {
				t.Errorf("Expected to find similar entry for first query")
			}
			if string(responseBody) != "response1" {
				t.Errorf("Expected response1, got %s", string(responseBody))
			}
		}

		// Step 4: Access second entry once
		responseBody, found, err := cache.FindSimilar("test-model", "Good morning")
		if err != nil {
			t.Logf("FindSimilar failed (expected due to high threshold): %v", err)
		}
		if !found {
			t.Errorf("Expected to find similar entry for second query")
		}
		if string(responseBody) != "response2" {
			t.Errorf("Expected response2, got %s", string(responseBody))
		}

		// Step 5: Add third entry - should trigger LFU eviction
		err = cache.AddEntry("req3", "test-model", "Bye",
			[]byte("request3"), []byte("response3"))
		if err != nil {
			t.Fatalf("Failed to add third entry: %v", err)
		}

		// Verify
		if len(cache.entries) != 2 {
			t.Errorf("Expected 2 entries after eviction, got %d", len(cache.entries))
		}
		if cache.entries[0].RequestID != "req1" {
			t.Errorf("Expected req1 to be the first entry, got %s", cache.entries[0].RequestID)
		}
		if cache.entries[1].RequestID != "req3" {
			t.Errorf("Expected req3 to be the second entry, got %s", cache.entries[1].RequestID)
		}
		if cache.entries[0].HitCount != 2 {
			t.Errorf("Expected HitCount to be 2, got %d", cache.entries[0].HitCount)
		}
		if cache.entries[1].HitCount != 0 {
			t.Errorf("Expected HitCount to be 0, got %d", cache.entries[1].HitCount)
		}
	})
}

// TestInMemoryCachePendingRequestWorkflow tests the in-memory cache pending request workflow
func TestInMemoryCachePendingRequestWorkflow(t *testing.T) {
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		t.Skipf("Failed to initialize BERT model: %v", err)
	}

	cache := NewInMemoryCache(InMemoryCacheOptions{
		Enabled:        true,
		MaxEntries:     2,
		EvictionPolicy: "lru",
	})

	t.Run("PendingRequestFlow", func(t *testing.T) {
		// Step 1: Add pending request
		err := cache.AddPendingRequest("req1", "test-model", "test query", []byte("request"))
		if err != nil {
			t.Fatalf("Failed to add pending request: %v", err)
		}

		// Verify
		if len(cache.entries) != 1 {
			t.Errorf("Expected 1 entry after AddPendingRequest, got %d", len(cache.entries))
		}

		if string(cache.entries[0].ResponseBody) != "" {
			t.Error("Expected ResponseBody to be empty for pending request")
		}

		// Step 2: Update with response
		err = cache.UpdateWithResponse("req1", []byte("response1"))
		if err != nil {
			t.Fatalf("Failed to update with response: %v", err)
		}

		// Step 3: Try to find similar
		response, found, err := cache.FindSimilar("test-model", "test query")
		if err != nil {
			t.Logf("FindSimilar error (may be due to embedding): %v", err)
		}

		if !found {
			t.Errorf("Expected to find completed entry after UpdateWithResponse")
		}
		if string(response) != "response1" {
			t.Errorf("Expected response1, got %s", string(response))
		}
	})
}

// TestEvictionPolicySelection tests that the correct policy is selected
func TestEvictionPolicySelection(t *testing.T) {
	testCases := []struct {
		policy   string
		expected string
	}{
		{"lru", "*cache.LRUPolicy"},
		{"lfu", "*cache.LFUPolicy"},
		{"fifo", "*cache.FIFOPolicy"},
		{"", "*cache.FIFOPolicy"},        // Default
		{"invalid", "*cache.FIFOPolicy"}, // Default fallback
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("Policy_%s", tc.policy), func(t *testing.T) {
			cache := NewInMemoryCache(InMemoryCacheOptions{
				EvictionPolicy: EvictionPolicyType(tc.policy),
			})

			policyType := fmt.Sprintf("%T", cache.evictionPolicy)
			if policyType != tc.expected {
				t.Errorf("Expected policy type %s, got %s", tc.expected, policyType)
			}
		})
	}
}
