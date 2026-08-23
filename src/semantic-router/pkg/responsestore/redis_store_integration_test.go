//go:build integration

package responsestore

import (
	"context"
	"errors"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

var redisIntegrationOwner = responseapi.ResponseOwner{
	Mode:        responseapi.ResponseOwnerAuthenticated,
	NamespaceID: "namespace-integration", APIKeyID: "key-integration", UserID: "user-integration",
}

// These tests require a running Redis instance on localhost:6379
// Run with: go test -tags=integration ./pkg/responsestore/...
//
// To inspect Redis data after tests (comment out cleanup in setupRedisStore):
//   redis-cli -n 15
//   KEYS sr:*
//   GET sr:response:<owner-hash>:resp_abc123
//
// To manually clean up test data:
//   redis-cli -n 15
//   KEYS sr:* | xargs redis-cli -n 15 DEL

func skipRedisIntegrationTests(t *testing.T) {
	t.Helper()
	if os.Getenv("SKIP_REDIS_TESTS") == "true" {
		t.Skip("Redis integration tests skipped due to SKIP_REDIS_TESTS=true")
	}
}

func setupRedisStore(t *testing.T) *RedisStore {
	skipRedisIntegrationTests(t)

	cfg := StoreConfig{
		Enabled:     true,
		TTLSeconds:  300, // 5 minutes
		BackendType: RedisStoreType,
		Redis: RedisStoreConfig{
			Address:   "localhost:6379",
			DB:        15, // Use DB 15 for testing to avoid conflicts
			KeyPrefix: "sr:",
		},
	}

	store, err := NewRedisStore(cfg)
	require.NoError(t, err, "Failed to create Redis store. Make sure Redis is running on localhost:6379")

	// Clean up any existing test data before running tests
	// Comment out this block to preserve data for manual inspection
	ctx := context.Background()
	pattern := store.buildKey("*")
	iter := store.client.Scan(ctx, 0, pattern, 0).Iterator()
	for iter.Next(ctx) {
		store.client.Del(ctx, iter.Val())
	}

	return store
}

func TestRedisStoreIntegration_BasicCRUD(t *testing.T) {
	store := setupRedisStore(t)
	defer store.Close()

	ctx := context.Background()

	// Create a test response
	response := &responseapi.StoredResponse{
		Owner:              redisIntegrationOwner,
		ID:                 "resp_abc123",
		ConversationID:     "conv_xyz789",
		Status:             "completed",
		CreatedAt:          time.Now().Unix(),
		Output:             []responseapi.OutputItem{{ID: "item_1"}},
		PreviousResponseID: "",
	}

	t.Run("Store response", func(t *testing.T) {
		err := store.StoreResponse(ctx, redisIntegrationOwner, response)
		assert.NoError(t, err)
	})

	t.Run("Get response", func(t *testing.T) {
		retrieved, err := store.GetResponse(ctx, redisIntegrationOwner, response.ID)
		assert.NoError(t, err)
		require.NotNil(t, retrieved)
		assert.Equal(t, response.ID, retrieved.ID)
		assert.Equal(t, response.ConversationID, retrieved.ConversationID)
		assert.Equal(t, response.Status, retrieved.Status)
	})

	t.Run("Update response", func(t *testing.T) {
		response.Status = "updated"
		err := store.UpdateResponse(ctx, redisIntegrationOwner, response)
		assert.NoError(t, err)

		retrieved, err := store.GetResponse(ctx, redisIntegrationOwner, response.ID)
		assert.NoError(t, err)
		assert.Equal(t, "updated", retrieved.Status)
	})

	t.Run("Delete response", func(t *testing.T) {
		err := store.DeleteResponse(ctx, redisIntegrationOwner, response.ID)
		assert.NoError(t, err)

		_, err = store.GetResponse(ctx, redisIntegrationOwner, response.ID)
		assert.Equal(t, ErrNotFound, err)
	})

	t.Run("Get non-existent response", func(t *testing.T) {
		_, err := store.GetResponse(ctx, redisIntegrationOwner, "resp_nonexistent")
		assert.Equal(t, ErrNotFound, err)
	})

	t.Run("Update non-existent response", func(t *testing.T) {
		nonExistent := &responseapi.StoredResponse{
			Owner: redisIntegrationOwner, ID: "resp_nonexistent",
		}
		err := store.UpdateResponse(ctx, redisIntegrationOwner, nonExistent)
		assert.Equal(t, ErrNotFound, err)
	})

	t.Run("Delete non-existent response", func(t *testing.T) {
		err := store.DeleteResponse(ctx, redisIntegrationOwner, "resp_nonexistent")
		assert.Equal(t, ErrNotFound, err)
	})
}

func TestRedisStoreIntegration_ResponseOwnershipIsPartitioned(t *testing.T) {
	store := setupRedisStore(t)
	defer store.Close()

	ctx := context.Background()
	otherOwner := responseapi.ResponseOwner{
		Mode:        responseapi.ResponseOwnerAuthenticated,
		NamespaceID: "namespace-other", APIKeyID: "key-other", UserID: "user-other",
	}
	responseID := "resp_cross_owner"
	for _, owner := range []responseapi.ResponseOwner{redisIntegrationOwner, otherOwner} {
		response := &responseapi.StoredResponse{
			Owner: owner, ID: responseID, Status: responseapi.StatusCompleted,
			Metadata: map[string]string{"owner": owner.UserID},
		}
		if err := store.StoreResponse(ctx, owner, response); err != nil {
			t.Fatalf("store owner %s: %v", owner.UserID, err)
		}
	}

	for _, owner := range []responseapi.ResponseOwner{redisIntegrationOwner, otherOwner} {
		stored, err := store.GetResponse(ctx, owner, responseID)
		if err != nil || stored.Owner != owner || stored.Metadata["owner"] != owner.UserID {
			t.Fatalf("get owner %s = (%v, %v)", owner.UserID, stored, err)
		}
	}
	missingOwner := otherOwner
	missingOwner.APIKeyID = "missing-key"
	if _, err := store.GetResponse(ctx, missingOwner, responseID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("cross-owner get error = %v, want ErrNotFound", err)
	}
	if err := store.DeleteResponse(ctx, missingOwner, responseID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("cross-owner delete error = %v, want ErrNotFound", err)
	}
	for _, owner := range []responseapi.ResponseOwner{redisIntegrationOwner, otherOwner} {
		if _, err := store.GetResponse(ctx, owner, responseID); err != nil {
			t.Fatalf("cross-owner delete affected owner %s: %v", owner.UserID, err)
		}
	}
}

func TestRedisStoreIntegration_ConversationChain(t *testing.T) {
	store := setupRedisStore(t)
	defer store.Close()

	ctx := context.Background()

	// Create a conversation chain: resp1 -> resp2 -> resp3
	responses := []*responseapi.StoredResponse{
		{
			Owner:              redisIntegrationOwner,
			ID:                 "resp_1",
			ConversationID:     "conv_chain",
			PreviousResponseID: "", // First in chain
			Status:             "completed",
			CreatedAt:          time.Now().Add(-2 * time.Minute).Unix(),
			Output:             []responseapi.OutputItem{{ID: "item_1"}},
		},
		{
			Owner:              redisIntegrationOwner,
			ID:                 "resp_2",
			ConversationID:     "conv_chain",
			PreviousResponseID: "resp_1",
			Status:             "completed",
			CreatedAt:          time.Now().Add(-1 * time.Minute).Unix(),
			Output:             []responseapi.OutputItem{{ID: "item_2"}},
		},
		{
			Owner:              redisIntegrationOwner,
			ID:                 "resp_3",
			ConversationID:     "conv_chain",
			PreviousResponseID: "resp_2",
			Status:             "completed",
			CreatedAt:          time.Now().Unix(),
			Output:             []responseapi.OutputItem{{ID: "item_3"}},
		},
	}

	// Store all responses
	for _, resp := range responses {
		err := store.StoreResponse(ctx, redisIntegrationOwner, resp)
		require.NoError(t, err)
	}

	t.Run("Get full chain from latest", func(t *testing.T) {
		chain, err := store.GetConversationChain(ctx, redisIntegrationOwner, "resp_3")
		assert.NoError(t, err)
		require.Len(t, chain, 3)

		// Verify chronological order (oldest first)
		assert.Equal(t, "resp_1", chain[0].ID)
		assert.Equal(t, "resp_2", chain[1].ID)
		assert.Equal(t, "resp_3", chain[2].ID)
	})

	t.Run("Get partial chain from middle", func(t *testing.T) {
		chain, err := store.GetConversationChain(ctx, redisIntegrationOwner, "resp_2")
		assert.NoError(t, err)
		require.Len(t, chain, 2)

		assert.Equal(t, "resp_1", chain[0].ID)
		assert.Equal(t, "resp_2", chain[1].ID)
	})

	t.Run("Get chain from first response", func(t *testing.T) {
		chain, err := store.GetConversationChain(ctx, redisIntegrationOwner, "resp_1")
		assert.NoError(t, err)
		require.Len(t, chain, 1)

		assert.Equal(t, "resp_1", chain[0].ID)
	})

	t.Run("Get chain for non-existent response", func(t *testing.T) {
		chain, err := store.GetConversationChain(ctx, redisIntegrationOwner, "resp_nonexistent")
		assert.Error(t, err)
		assert.Nil(t, chain)
	})
}

func TestRedisStoreIntegration_TTL(t *testing.T) {
	skipRedisIntegrationTests(t)

	// Create store with very short TTL for testing
	cfg := StoreConfig{
		Enabled:     true,
		TTLSeconds:  2, // 2 seconds
		BackendType: RedisStoreType,
		Redis: RedisStoreConfig{
			Address:   "localhost:6379",
			DB:        15,
			KeyPrefix: "sr:",
		},
	}

	store, err := NewRedisStore(cfg)
	require.NoError(t, err)
	defer store.Close()

	ctx := context.Background()

	response := &responseapi.StoredResponse{
		Owner:  redisIntegrationOwner,
		ID:     "resp_ttl",
		Status: "completed",
		Output: []responseapi.OutputItem{{ID: "item_1"}},
	}

	t.Run("Response expires after TTL", func(t *testing.T) {
		// Store response
		err := store.StoreResponse(ctx, redisIntegrationOwner, response)
		require.NoError(t, err)

		// Verify it exists
		_, err = store.GetResponse(ctx, redisIntegrationOwner, response.ID)
		assert.NoError(t, err)

		// Wait for TTL to expire
		time.Sleep(3 * time.Second)

		// Verify it's gone
		_, err = store.GetResponse(ctx, redisIntegrationOwner, response.ID)
		assert.Equal(t, ErrNotFound, err)
	})
}

func TestRedisStoreIntegration_ConversationOperations(t *testing.T) {
	store := setupRedisStore(t)
	defer store.Close()

	ctx := context.Background()

	conversation := &responseapi.StoredConversation{
		Owner:     redisIntegrationOwner,
		ID:        "conv_abc123",
		CreatedAt: time.Now().Unix(),
		UpdatedAt: time.Now().Unix(),
	}

	t.Run("Create conversation", func(t *testing.T) {
		err := store.CreateConversation(ctx, redisIntegrationOwner, conversation)
		assert.NoError(t, err)
	})

	t.Run("Get conversation", func(t *testing.T) {
		retrieved, err := store.GetConversation(ctx, redisIntegrationOwner, conversation.ID)
		assert.NoError(t, err)
		require.NotNil(t, retrieved)
		assert.Equal(t, conversation.ID, retrieved.ID)
	})

	t.Run("Update conversation", func(t *testing.T) {
		conversation.UpdatedAt = time.Now().Unix()
		err := store.UpdateConversation(ctx, redisIntegrationOwner, conversation)
		assert.NoError(t, err)

		retrieved, err := store.GetConversation(ctx, redisIntegrationOwner, conversation.ID)
		assert.NoError(t, err)
		assert.Equal(t, conversation.UpdatedAt, retrieved.UpdatedAt)
	})

	t.Run("Delete conversation", func(t *testing.T) {
		err := store.DeleteConversation(ctx, redisIntegrationOwner, conversation.ID, false)
		assert.NoError(t, err)

		_, err = store.GetConversation(ctx, redisIntegrationOwner, conversation.ID)
		assert.Equal(t, ErrNotFound, err)
	})
}

func TestRedisStoreIntegration_ListOperations(t *testing.T) {
	store := setupRedisStore(t)
	defer store.Close()

	ctx := context.Background()

	convID := "conv_list"

	// Create multiple responses for same conversation
	for i := 0; i < 5; i++ {
		response := &responseapi.StoredResponse{
			Owner:          redisIntegrationOwner,
			ID:             fmt.Sprintf("resp_list_%d", i),
			ConversationID: convID,
			Status:         "completed",
			CreatedAt:      time.Now().Unix(),
			Output:         []responseapi.OutputItem{{ID: fmt.Sprintf("item_%d", i)}},
		}
		err := store.StoreResponse(ctx, redisIntegrationOwner, response)
		require.NoError(t, err)
	}

	t.Run("List responses by conversation", func(t *testing.T) {
		responses, err := store.ListResponsesByConversation(ctx, redisIntegrationOwner, convID, ListOptions{})
		assert.NoError(t, err)
		assert.Len(t, responses, 5)
	})

	t.Run("List responses with limit", func(t *testing.T) {
		responses, err := store.ListResponsesByConversation(ctx, redisIntegrationOwner, convID, ListOptions{Limit: 3})
		assert.NoError(t, err)
		assert.LessOrEqual(t, len(responses), 3)
	})
}

func TestRedisStoreIntegration_ConcurrentAccess(t *testing.T) {
	store := setupRedisStore(t)
	defer store.Close()

	ctx := context.Background()

	t.Run("Concurrent writes", func(t *testing.T) {
		const numGoroutines = 10
		done := make(chan bool, numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			go func(index int) {
				response := &responseapi.StoredResponse{
					Owner:  redisIntegrationOwner,
					ID:     fmt.Sprintf("resp_concurrent_%d", index),
					Status: "completed",
					Output: []responseapi.OutputItem{{ID: fmt.Sprintf("item_%d", index)}},
				}
				err := store.StoreResponse(ctx, redisIntegrationOwner, response)
				assert.NoError(t, err)
				done <- true
			}(i)
		}

		// Wait for all goroutines
		for i := 0; i < numGoroutines; i++ {
			<-done
		}

		// Verify all responses were stored
		for i := 0; i < numGoroutines; i++ {
			_, err := store.GetResponse(ctx, redisIntegrationOwner, fmt.Sprintf("resp_concurrent_%d", i))
			assert.NoError(t, err)
		}
	})
}

func TestRedisStoreIntegration_CircularReferenceProtection(t *testing.T) {
	store := setupRedisStore(t)
	defer store.Close()

	ctx := context.Background()

	// Create circular reference: resp1 -> resp2 -> resp1
	responses := []*responseapi.StoredResponse{
		{
			Owner:              redisIntegrationOwner,
			ID:                 "resp_circular_1",
			ConversationID:     "conv_circular",
			PreviousResponseID: "resp_circular_2", // Points to resp2
			Status:             "completed",
			Output:             []responseapi.OutputItem{{ID: "item_1"}},
		},
		{
			Owner:              redisIntegrationOwner,
			ID:                 "resp_circular_2",
			ConversationID:     "conv_circular",
			PreviousResponseID: "resp_circular_1", // Points back to resp1 (circular!)
			Status:             "completed",
			Output:             []responseapi.OutputItem{{ID: "item_2"}},
		},
	}

	for _, resp := range responses {
		err := store.StoreResponse(ctx, redisIntegrationOwner, resp)
		require.NoError(t, err)
	}

	t.Run("Circular reference doesn't cause infinite loop", func(t *testing.T) {
		// This should not hang
		done := make(chan bool)
		go func() {
			chain, err := store.GetConversationChain(ctx, redisIntegrationOwner, "resp_circular_1")
			assert.NoError(t, err)
			// Should break out of loop and return partial chain
			assert.NotEmpty(t, chain)
			done <- true
		}()

		select {
		case <-done:
			// Success - function returned
		case <-time.After(5 * time.Second):
			t.Fatal("GetConversationChain hung - circular reference protection failed")
		}
	})
}
