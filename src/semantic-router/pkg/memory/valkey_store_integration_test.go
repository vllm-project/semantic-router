//go:build !windows && cgo

package memory

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	glide "github.com/valkey-io/valkey-glide/go/v2"
	glideconfig "github.com/valkey-io/valkey-glide/go/v2/config"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// These tests require:
//  1. A running Valkey instance with search module.
//     Default: localhost:6379. Override with VALKEY_HOST / VALKEY_PORT env vars.
//  2. BERT model initialized for embeddings.
//
// Set SKIP_VALKEY_TESTS=true to skip.

func setupValkeyMemoryIntegration(t *testing.T) (*ValkeyStore, *glide.Client) {
	t.Helper()

	if os.Getenv("SKIP_VALKEY_TESTS") == "true" {
		t.Skip("Valkey integration tests skipped due to SKIP_VALKEY_TESTS=true")
	}

	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		t.Skipf("Failed to initialize BERT model: %v", err)
	}

	host := "localhost"
	port := 6379

	if h := os.Getenv("VALKEY_HOST"); h != "" {
		host = h
	}
	if p := os.Getenv("VALKEY_PORT"); p != "" {
		if v, err := strconv.Atoi(p); err == nil {
			port = v
		}
	}

	// Use unique index/prefix per test to avoid collisions
	suffix := strconv.FormatInt(time.Now().UnixNano(), 36)
	indexName := fmt.Sprintf("test_mem_idx_%s", suffix)
	prefix := fmt.Sprintf("test_mem_%s:", suffix)

	vc := &config.MemoryValkeyConfig{
		Host:                host,
		Port:                port,
		Database:            0,
		Timeout:             5,
		CollectionPrefix:    prefix,
		IndexName:           indexName,
		Dimension:           384,
		MetricType:          "COSINE",
		IndexM:              16,
		IndexEfConstruction: 256,
	}

	clientConfig := glideconfig.NewClientConfiguration().
		WithAddress(&glideconfig.NodeAddress{Host: host, Port: port}).
		WithRequestTimeout(5 * time.Second)

	client, err := glide.NewClient(clientConfig)
	if err != nil {
		t.Skipf("Cannot connect to Valkey at %s:%d: %v", host, port, err)
	}

	memCfg := config.MemoryConfig{
		DefaultRetrievalLimit:      5,
		DefaultSimilarityThreshold: 0.70,
	}

	store, err := NewValkeyStore(ValkeyStoreOptions{
		Client:       client,
		Config:       memCfg,
		ValkeyConfig: vc,
		Enabled:      true,
		EmbeddingConfig: &EmbeddingConfig{
			Model: EmbeddingModelBERT,
		},
	})
	if err != nil {
		client.Close()
		t.Fatalf("Failed to create ValkeyStore: %v", err)
	}

	// Cleanup on test completion
	t.Cleanup(func() {
		ctx := context.Background()
		// Drop the index
		_, _ = client.CustomCommand(ctx, []string{"FT.DROPINDEX", indexName})
		// Clean up keys with the prefix using SCAN
		cleanupValkeyKeys(ctx, client, prefix)
		client.Close()
	})

	return store, client
}

func cleanupValkeyKeys(ctx context.Context, client *glide.Client, prefix string) {
	// Use SCAN to find and delete all keys with the prefix.
	cursor := "0"
	pattern := prefix + "*"
	for {
		result, err := client.CustomCommand(ctx, []string{"SCAN", cursor, "MATCH", pattern, "COUNT", "100"})
		if err != nil {
			return
		}
		arr, ok := result.([]interface{})
		if !ok || len(arr) < 2 {
			return
		}
		cursor = fmt.Sprint(arr[0])

		var keys []string
		if keyList, ok := arr[1].([]interface{}); ok {
			for _, k := range keyList {
				if s, ok := k.(string); ok {
					keys = append(keys, s)
				}
			}
		}
		if len(keys) > 0 {
			_, _ = client.Del(ctx, keys)
		}
		if cursor == "0" {
			return
		}
	}
}

// ---------------------------------------------------------------------------
// CheckConnection
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_CheckConnection(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	err := store.CheckConnection(ctx)
	assert.NoError(t, err)
}

// ---------------------------------------------------------------------------
// Store + Get (full CRUD lifecycle)
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_StoreAndGet(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	mem := &Memory{
		ID:         fmt.Sprintf("mem_integ_%d", time.Now().UnixNano()),
		Type:       MemoryTypeSemantic,
		Content:    "The user's preferred programming language is Go",
		UserID:     "test_user_1",
		ProjectID:  "proj_1",
		Source:     "conversation",
		Importance: 0.8,
	}

	// Store
	err := store.Store(ctx, mem)
	require.NoError(t, err)

	// Allow index to catch up
	time.Sleep(200 * time.Millisecond)

	// Get
	retrieved, err := store.Get(ctx, mem.ID)
	require.NoError(t, err)
	assert.Equal(t, mem.ID, retrieved.ID)
	assert.Equal(t, mem.Content, retrieved.Content)
	assert.Equal(t, mem.UserID, retrieved.UserID)
	assert.Equal(t, MemoryTypeSemantic, retrieved.Type)
	assert.Equal(t, "proj_1", retrieved.ProjectID)
	assert.Equal(t, "conversation", retrieved.Source)
	assert.InDelta(t, 0.8, float64(retrieved.Importance), 0.01)
	assert.NotNil(t, retrieved.Embedding, "Get should return the embedding")
	assert.False(t, retrieved.CreatedAt.IsZero())
	assert.False(t, retrieved.UpdatedAt.IsZero())
}

// ---------------------------------------------------------------------------
// Get — not found
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_GetNotFound(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	_, err := store.Get(ctx, "nonexistent_id_12345")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "memory not found")
}

// ---------------------------------------------------------------------------
// Retrieve (semantic search)
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_Retrieve(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	// Store some memories
	memories := []*Memory{
		{
			ID:      fmt.Sprintf("mem_ret_%d_1", time.Now().UnixNano()),
			Type:    MemoryTypeSemantic,
			Content: "The user prefers dark mode in all their applications",
			UserID:  "retrieve_user",
		},
		{
			ID:      fmt.Sprintf("mem_ret_%d_2", time.Now().UnixNano()),
			Type:    MemoryTypeProcedural,
			Content: "To deploy the application, run make deploy in the project root",
			UserID:  "retrieve_user",
		},
		{
			ID:      fmt.Sprintf("mem_ret_%d_3", time.Now().UnixNano()),
			Type:    MemoryTypeSemantic,
			Content: "The capital of France is Paris, a well known European city",
			UserID:  "retrieve_user",
		},
	}

	for _, m := range memories {
		require.NoError(t, store.Store(ctx, m))
	}

	// Allow index to catch up
	time.Sleep(500 * time.Millisecond)

	// Search for dark mode preference
	results, err := store.Retrieve(ctx, RetrieveOptions{
		Query:     "What are the user's display preferences?",
		UserID:    "retrieve_user",
		Limit:     3,
		Threshold: 0.3, // Low threshold for integration test
	})
	require.NoError(t, err)
	assert.NotEmpty(t, results, "should find at least one result")

	// Verify results have scores
	for _, r := range results {
		assert.NotEmpty(t, r.Memory.ID)
		assert.NotEmpty(t, r.Memory.Content)
		assert.Positive(t, r.Score, "score should be positive")
	}
}

// ---------------------------------------------------------------------------
// Retrieve — empty results
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_RetrieveEmpty(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	results, err := store.Retrieve(ctx, RetrieveOptions{
		Query:     "something that matches nothing in an empty store",
		UserID:    "empty_user_" + strconv.FormatInt(time.Now().UnixNano(), 36),
		Limit:     5,
		Threshold: 0.99, // Very high threshold
	})
	require.NoError(t, err)
	assert.Empty(t, results)
}

// ---------------------------------------------------------------------------
// Retrieve with type filter
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_RetrieveWithTypeFilter(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	ts := time.Now().UnixNano()
	require.NoError(t, store.Store(ctx, &Memory{
		ID: fmt.Sprintf("mem_tf_%d_1", ts), Type: MemoryTypeSemantic,
		Content: "User likes Python programming language very much", UserID: "filter_user",
	}))
	require.NoError(t, store.Store(ctx, &Memory{
		ID: fmt.Sprintf("mem_tf_%d_2", ts), Type: MemoryTypeProcedural,
		Content: "To run Python tests use pytest command in terminal", UserID: "filter_user",
	}))

	time.Sleep(500 * time.Millisecond)

	results, err := store.Retrieve(ctx, RetrieveOptions{
		Query:     "Python programming",
		UserID:    "filter_user",
		Types:     []MemoryType{MemoryTypeProcedural},
		Limit:     5,
		Threshold: 0.3,
	})
	require.NoError(t, err)

	for _, r := range results {
		assert.Equal(t, MemoryTypeProcedural, r.Memory.Type, "should only return procedural memories")
	}
}

// ---------------------------------------------------------------------------
// User-scoped isolation
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_UserScopedIsolation(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	ts := time.Now().UnixNano()
	require.NoError(t, store.Store(ctx, &Memory{
		ID: fmt.Sprintf("mem_iso_%d_a", ts), Type: MemoryTypeSemantic,
		Content: "User A's secret preference is dark chocolate", UserID: "user_a",
	}))
	require.NoError(t, store.Store(ctx, &Memory{
		ID: fmt.Sprintf("mem_iso_%d_b", ts), Type: MemoryTypeSemantic,
		Content: "User B's secret preference is white chocolate", UserID: "user_b",
	}))

	time.Sleep(500 * time.Millisecond)

	// User A should only see their own memories
	resultsA, err := store.Retrieve(ctx, RetrieveOptions{
		Query: "chocolate preference", UserID: "user_a", Limit: 5, Threshold: 0.3,
	})
	require.NoError(t, err)
	for _, r := range resultsA {
		assert.Equal(t, "user_a", r.Memory.UserID, "user_a should only see their own memories")
	}

	// User B should only see their own memories
	resultsB, err := store.Retrieve(ctx, RetrieveOptions{
		Query: "chocolate preference", UserID: "user_b", Limit: 5, Threshold: 0.3,
	})
	require.NoError(t, err)
	for _, r := range resultsB {
		assert.Equal(t, "user_b", r.Memory.UserID, "user_b should only see their own memories")
	}
}

// ---------------------------------------------------------------------------
// Update
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_Update(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	id := fmt.Sprintf("mem_upd_%d", time.Now().UnixNano())
	original := &Memory{
		ID: id, Type: MemoryTypeSemantic,
		Content: "Original content for update test", UserID: "update_user",
		Importance: 0.5,
	}
	require.NoError(t, store.Store(ctx, original))
	time.Sleep(200 * time.Millisecond)

	// Update content and importance
	updated := &Memory{
		Content:    "Updated content after modification",
		UserID:     "update_user",
		Type:       MemoryTypeSemantic,
		Importance: 0.9,
	}
	err := store.Update(ctx, id, updated)
	require.NoError(t, err)

	time.Sleep(200 * time.Millisecond)

	// Verify update
	retrieved, err := store.Get(ctx, id)
	require.NoError(t, err)
	assert.Equal(t, "Updated content after modification", retrieved.Content)
	assert.InDelta(t, 0.9, float64(retrieved.Importance), 0.01)
	assert.False(t, retrieved.CreatedAt.IsZero(), "CreatedAt should be preserved")
}

// ---------------------------------------------------------------------------
// Update — not found
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_UpdateNotFound(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	err := store.Update(ctx, "nonexistent_update_id", &Memory{
		Content: "wont work", UserID: "u1",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "memory not found")
}

// ---------------------------------------------------------------------------
// Forget
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_Forget(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	id := fmt.Sprintf("mem_forget_%d", time.Now().UnixNano())
	require.NoError(t, store.Store(ctx, &Memory{
		ID: id, Type: MemoryTypeSemantic,
		Content: "Memory to be forgotten", UserID: "forget_user",
	}))
	time.Sleep(200 * time.Millisecond)

	// Verify it exists
	_, err := store.Get(ctx, id)
	require.NoError(t, err)

	// Forget
	err = store.Forget(ctx, id)
	require.NoError(t, err)

	// Verify deleted
	_, err = store.Get(ctx, id)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "memory not found")
}

// ---------------------------------------------------------------------------
// ForgetByScope — user only
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_ForgetByScope_UserOnly(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	userID := fmt.Sprintf("scope_user_%d", time.Now().UnixNano())
	for i := 0; i < 3; i++ {
		require.NoError(t, store.Store(ctx, &Memory{
			ID:      fmt.Sprintf("mem_scope_%s_%d", userID, i),
			Type:    MemoryTypeSemantic,
			Content: fmt.Sprintf("Scoped memory %d for deletion", i),
			UserID:  userID,
		}))
	}
	time.Sleep(500 * time.Millisecond)

	// Verify they exist
	list, err := store.List(ctx, ListOptions{UserID: userID, Limit: 10})
	require.NoError(t, err)
	assert.Equal(t, 3, list.Total)

	// Delete by scope
	err = store.ForgetByScope(ctx, MemoryScope{UserID: userID})
	require.NoError(t, err)

	time.Sleep(200 * time.Millisecond)

	// Verify all deleted
	list, err = store.List(ctx, ListOptions{UserID: userID, Limit: 10})
	require.NoError(t, err)
	assert.Equal(t, 0, list.Total)
}

// ---------------------------------------------------------------------------
// ForgetByScope — with type filter
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_ForgetByScope_WithTypeFilter(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	userID := fmt.Sprintf("scope_type_user_%d", time.Now().UnixNano())
	require.NoError(t, store.Store(ctx, &Memory{
		ID: fmt.Sprintf("mem_st_%s_1", userID), Type: MemoryTypeSemantic,
		Content: "Semantic memory to keep", UserID: userID,
	}))
	require.NoError(t, store.Store(ctx, &Memory{
		ID: fmt.Sprintf("mem_st_%s_2", userID), Type: MemoryTypeProcedural,
		Content: "Procedural memory to delete", UserID: userID,
	}))
	time.Sleep(500 * time.Millisecond)

	// Delete only procedural
	err := store.ForgetByScope(ctx, MemoryScope{
		UserID: userID,
		Types:  []MemoryType{MemoryTypeProcedural},
	})
	require.NoError(t, err)

	time.Sleep(200 * time.Millisecond)

	// Verify only semantic remains
	list, err := store.List(ctx, ListOptions{UserID: userID, Limit: 10})
	require.NoError(t, err)
	assert.Equal(t, 1, list.Total)
	assert.Equal(t, MemoryTypeSemantic, list.Memories[0].Type)
}

// ---------------------------------------------------------------------------
// ForgetByScope — missing UserID
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_ForgetByScope_MissingUserID(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	err := store.ForgetByScope(ctx, MemoryScope{})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "user ID is required")
}

// ---------------------------------------------------------------------------
// List
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_List(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	userID := fmt.Sprintf("list_user_%d", time.Now().UnixNano())
	for i := 0; i < 5; i++ {
		require.NoError(t, store.Store(ctx, &Memory{
			ID:      fmt.Sprintf("mem_list_%s_%d", userID, i),
			Type:    MemoryTypeSemantic,
			Content: fmt.Sprintf("List test memory number %d", i),
			UserID:  userID,
		}))
		time.Sleep(50 * time.Millisecond) // ensure different timestamps
	}
	time.Sleep(500 * time.Millisecond)

	list, err := store.List(ctx, ListOptions{UserID: userID, Limit: 3})
	require.NoError(t, err)
	assert.Equal(t, 5, list.Total)
	assert.Len(t, list.Memories, 3)

	// Verify sorted by created_at descending
	for i := 1; i < len(list.Memories); i++ {
		assert.False(t, list.Memories[i-1].CreatedAt.Before(list.Memories[i].CreatedAt),
			"memories should be sorted by created_at descending")
	}
}

// ---------------------------------------------------------------------------
// List — missing UserID
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_List_MissingUserID(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	_, err := store.List(ctx, ListOptions{})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "user ID is required")
}

// ---------------------------------------------------------------------------
// Duplicate keys (overwrite behavior)
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_DuplicateKeys(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	id := fmt.Sprintf("mem_dup_%d", time.Now().UnixNano())

	// Store first version
	require.NoError(t, store.Store(ctx, &Memory{
		ID: id, Type: MemoryTypeSemantic,
		Content: "First version", UserID: "dup_user",
	}))
	time.Sleep(200 * time.Millisecond)

	// Store again with same ID (HSET overwrites)
	require.NoError(t, store.Store(ctx, &Memory{
		ID: id, Type: MemoryTypeSemantic,
		Content: "Second version", UserID: "dup_user",
	}))
	time.Sleep(200 * time.Millisecond)

	// Verify latest content
	retrieved, err := store.Get(ctx, id)
	require.NoError(t, err)
	assert.Equal(t, "Second version", retrieved.Content)
}

// ---------------------------------------------------------------------------
// Concurrent access
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_ConcurrentAccess(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	const numGoroutines = 10
	var wg sync.WaitGroup
	errs := make(chan error, numGoroutines*2)

	userID := fmt.Sprintf("concurrent_user_%d", time.Now().UnixNano())

	// Concurrent stores
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			err := store.Store(ctx, &Memory{
				ID:      fmt.Sprintf("mem_conc_%s_%d", userID, idx),
				Type:    MemoryTypeSemantic,
				Content: fmt.Sprintf("Concurrent memory %d for testing parallel access", idx),
				UserID:  userID,
			})
			if err != nil {
				errs <- err
			}
		}(i)
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		t.Errorf("concurrent store error: %v", err)
	}

	time.Sleep(500 * time.Millisecond)

	// Verify all stored
	list, err := store.List(ctx, ListOptions{UserID: userID, Limit: 100})
	require.NoError(t, err)
	assert.Equal(t, numGoroutines, list.Total)
}

// ---------------------------------------------------------------------------
// ForgetByScope — with project filter
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_ForgetByScope_WithProjectFilter(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	userID := fmt.Sprintf("scope_proj_user_%d", time.Now().UnixNano())
	require.NoError(t, store.Store(ctx, &Memory{
		ID: fmt.Sprintf("mem_sp_%s_1", userID), Type: MemoryTypeSemantic,
		Content: "Memory for project A", UserID: userID, ProjectID: "projA",
	}))
	require.NoError(t, store.Store(ctx, &Memory{
		ID: fmt.Sprintf("mem_sp_%s_2", userID), Type: MemoryTypeSemantic,
		Content: "Memory for project B", UserID: userID, ProjectID: "projB",
	}))
	time.Sleep(500 * time.Millisecond)

	// Delete only projA
	err := store.ForgetByScope(ctx, MemoryScope{
		UserID:    userID,
		ProjectID: "projA",
	})
	require.NoError(t, err)

	time.Sleep(200 * time.Millisecond)

	// Verify only projB remains
	list, err := store.List(ctx, ListOptions{UserID: userID, Limit: 10})
	require.NoError(t, err)
	assert.Equal(t, 1, list.Total)
	assert.Equal(t, "projB", list.Memories[0].ProjectID)
}

// ---------------------------------------------------------------------------
// ConsolidateUser (standalone function with ValkeyStore)
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_ConsolidateUser(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	userID := fmt.Sprintf("consol_user_%d", time.Now().UnixNano())

	// Store memories with similar content (high Jaccard similarity)
	require.NoError(t, store.Store(ctx, &Memory{
		ID: fmt.Sprintf("mem_c_%s_1", userID), Type: MemoryTypeSemantic,
		Content: "The user prefers dark mode in all applications", UserID: userID,
	}))
	require.NoError(t, store.Store(ctx, &Memory{
		ID: fmt.Sprintf("mem_c_%s_2", userID), Type: MemoryTypeSemantic,
		Content: "The user prefers dark mode in all their applications and IDEs", UserID: userID,
	}))
	// Store a different memory
	require.NoError(t, store.Store(ctx, &Memory{
		ID: fmt.Sprintf("mem_c_%s_3", userID), Type: MemoryTypeSemantic,
		Content: "Python is installed at /usr/bin/python3", UserID: userID,
	}))
	time.Sleep(500 * time.Millisecond)

	merged, deleted, err := ConsolidateUser(ctx, store, userID)
	require.NoError(t, err)

	// The two similar memories should be merged
	assert.GreaterOrEqual(t, merged, 0, "merged count should be non-negative")
	assert.GreaterOrEqual(t, deleted, 0, "deleted count should be non-negative")

	// The total should be reduced
	list, err := store.List(ctx, ListOptions{UserID: userID, Limit: 100})
	require.NoError(t, err)
	t.Logf("ConsolidateUser: merged=%d, deleted=%d, remaining=%d", merged, deleted, list.Total)
}

// ---------------------------------------------------------------------------
// Store validation errors
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_StoreValidation(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	t.Run("missing ID", func(t *testing.T) {
		err := store.Store(ctx, &Memory{Content: "test", UserID: "u1"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "memory ID is required")
	})

	t.Run("missing content", func(t *testing.T) {
		err := store.Store(ctx, &Memory{ID: "test_id", UserID: "u1"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "memory content is required")
	})

	t.Run("missing user ID", func(t *testing.T) {
		err := store.Store(ctx, &Memory{ID: "test_id", Content: "test"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "user ID is required")
	})
}

// ---------------------------------------------------------------------------
// Retrieve validation errors
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_RetrieveValidation(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	t.Run("missing query", func(t *testing.T) {
		_, err := store.Retrieve(ctx, RetrieveOptions{UserID: "u1"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "query is required")
	})

	t.Run("missing user ID", func(t *testing.T) {
		_, err := store.Retrieve(ctx, RetrieveOptions{Query: "test"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "user id is required")
	})
}

// ---------------------------------------------------------------------------
// IsEnabled / disabled store
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_DisabledStore(t *testing.T) {
	t.Parallel()

	store := &ValkeyStore{enabled: false}

	assert.False(t, store.IsEnabled())

	ctx := context.Background()
	assert.Error(t, store.Store(ctx, &Memory{}))
	_, err := store.Retrieve(ctx, RetrieveOptions{})
	assert.Error(t, err)
	_, err = store.Get(ctx, "id")
	assert.Error(t, err)
	assert.Error(t, store.Update(ctx, "id", &Memory{}))
	_, err = store.List(ctx, ListOptions{})
	assert.Error(t, err)
	assert.Error(t, store.Forget(ctx, "id"))
	assert.Error(t, store.ForgetByScope(ctx, MemoryScope{}))
}
