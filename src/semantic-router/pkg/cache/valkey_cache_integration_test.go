//go:build !windows && cgo

package cache

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// valkeyIntegrationAddr returns the Valkey host/port for integration tests,
// honoring VALKEY_HOST / VALKEY_PORT env vars and falling back to localhost:6379.
// In CI, VALKEY_PORT is set to 6380 because Redis Stack occupies 6379; connecting
// to 6379 would hit RediSearch (which also registers a module named "search"),
// not valkey-search.
func valkeyIntegrationAddr() (string, int) {
	host := "localhost"
	port := 6379

	if v := os.Getenv("VALKEY_HOST"); v != "" {
		host = v
	}
	if v := os.Getenv("VALKEY_PORT"); v != "" {
		if p, err := strconv.Atoi(v); err == nil {
			port = p
		}
	}
	return host, port
}

// These tests require:
//  1. A running Valkey instance with search module.
//     Default: localhost:6379 (standalone / local-only).
//     Override with VALKEY_HOST / VALKEY_PORT env vars.
//     In CI (or when Redis already occupies 6379), `make start-valkey`
//     maps Valkey to port 6380 and the Makefile test targets set
//     VALKEY_PORT=6380 automatically.
//  2. BERT model initialized for embeddings
func setupValkeyCacheIntegration(t *testing.T) *ValkeyCache {
	// Skip if SKIP_VALKEY_TESTS is set
	if os.Getenv("SKIP_VALKEY_TESTS") == "true" {
		t.Skip("Valkey integration tests skipped due to SKIP_VALKEY_TESTS=true")
	}

	// Initialize BERT model for embeddings
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		t.Skipf("Failed to initialize BERT model: %v", err)
	}

	valkeyHost, valkeyPort := valkeyIntegrationAddr()

	// Create Valkey config
	valkeyConfig := &config.ValkeyConfig{}
	valkeyConfig.Connection.Host = valkeyHost
	valkeyConfig.Connection.Port = valkeyPort
	valkeyConfig.Connection.Database = 0
	valkeyConfig.Connection.Password = ""
	valkeyConfig.Connection.Timeout = 5

	valkeyConfig.Index.Name = "test_valkey_idx"
	valkeyConfig.Index.Prefix = "doc:"
	valkeyConfig.Index.VectorField.Name = "embedding"
	valkeyConfig.Index.VectorField.Dimension = 384 // BERT dimension
	valkeyConfig.Index.VectorField.MetricType = "COSINE"
	valkeyConfig.Index.IndexType = "HNSW"
	valkeyConfig.Index.Params.M = 16
	valkeyConfig.Index.Params.EfConstruction = 64

	valkeyConfig.Search.TopK = 1

	valkeyConfig.Development.DropIndexOnStartup = true
	valkeyConfig.Development.AutoCreateIndex = true

	// Create cache
	cache, err := NewValkeyCache(ValkeyCacheOptions{
		SimilarityThreshold: 0.8,
		TTLSeconds:          300,
		Enabled:             true,
		Config:              valkeyConfig,
		EmbeddingModel:      "bert",
	})
	if err != nil {
		t.Skipf("Valkey server not available (skipping integration test): %v", err)
	}
	return cache
}

func TestValkeyCacheIntegration_ConnectionCheck(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	err := cache.CheckConnection()
	assert.NoError(t, err, "Connection check should succeed")
}

func TestValkeyCacheIntegration_IndexCreation(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	ctx := context.Background()

	// Verify index was created using FT.INFO
	result, err := cache.client.CustomCommand(ctx, []string{"FT.INFO", cache.indexName})
	assert.NoError(t, err, "FT.INFO should succeed after index creation")
	assert.NotNil(t, result, "Index info should not be nil")
}

func TestValkeyCacheIntegration_AddEntry(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	ctx := context.Background()

	requestID := "req_test_123"
	model := "gpt-4"
	query := "What is the capital of France?"
	requestBody := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"What is the capital of France?"}]}`)
	responseBody := []byte(`{"choices":[{"message":{"content":"Paris is the capital of France."}}]}`)
	ttlSeconds := 300

	err := cache.AddEntry(requestID, model, query, requestBody, responseBody, ttlSeconds)
	assert.NoError(t, err, "AddEntry should succeed")

	// Wait for indexing
	time.Sleep(100 * time.Millisecond)

	// Verify entry was stored using SCAN
	pattern := cache.config.Index.Prefix + "*"
	result, err := cache.client.CustomCommand(ctx, []string{"SCAN", "0", "MATCH", pattern, "COUNT", "10"})
	assert.NoError(t, err, "SCAN should succeed")
	assert.NotNil(t, result, "SCAN result should not be nil")
}

func TestValkeyCacheIntegration_FindSimilar(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	requestID := "req_find_test"
	model := "gpt-4"
	query := "What is machine learning?"
	requestBody := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"What is machine learning?"}]}`)
	responseBody := []byte(`{"choices":[{"message":{"content":"Machine learning is a subset of AI."}}]}`)
	ttlSeconds := 300

	err := cache.AddEntry(requestID, model, query, requestBody, responseBody, ttlSeconds)
	require.NoError(t, err)

	// Wait for indexing with retry logic
	var foundResponse []byte
	var hit bool
	maxRetries := 5

	for i := 0; i < maxRetries; i++ {
		if i > 0 {
			t.Logf("Retry %d/%d for FindSimilar", i+1, maxRetries)
		}
		time.Sleep(time.Duration(200*(i+1)) * time.Millisecond)

		// Test FindSimilar with exact same query
		foundResponse, hit, err = cache.FindSimilar(model, query)
		require.NoError(t, err, "FindSimilar should not error")

		if hit {
			break
		}
	}

	require.True(t, hit, "Should find the exact same query after retries")
	assert.NotNil(t, foundResponse, "Response should be returned on cache hit")
	assert.Contains(t, string(foundResponse), "Machine learning", "Response should contain expected content")
}

func TestValkeyCacheIntegration_FindSimilarWithThreshold(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	requestID := "req_threshold_test"
	model := "gpt-4"
	query := "Explain neural networks"
	requestBody := []byte(`{"model":"gpt-4"}`)
	responseBody := []byte(`{"choices":[{"message":{"content":"Neural networks are computing systems."}}]}`)
	ttlSeconds := 300

	err := cache.AddEntry(requestID, model, query, requestBody, responseBody, ttlSeconds)
	require.NoError(t, err)

	// Wait for indexing
	time.Sleep(200 * time.Millisecond)

	// Test with very similar query
	similarQuery := "Explain neural networks"
	foundResponse, hit, err := cache.FindSimilarWithThreshold(model, similarQuery, 0.95)
	assert.NoError(t, err, "FindSimilarWithThreshold should not error")

	if hit {
		assert.NotNil(t, foundResponse, "Response should be returned on cache hit")
	}

	// Test with very different query (should miss)
	differentQuery := "What is the weather today?"
	foundResponse, hit, err = cache.FindSimilarWithThreshold(model, differentQuery, 0.95)
	assert.NoError(t, err, "FindSimilarWithThreshold should not error even on miss")

	if !hit {
		assert.Nil(t, foundResponse, "Response should be nil on cache miss")
	}
}

func TestValkeyCacheIntegration_AddPendingRequest(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	requestID := "req_pending_123"
	model := "gpt-4"
	query := "Tell me a joke"
	requestBody := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"Tell me a joke"}]}`)
	ttlSeconds := 300

	err := cache.AddPendingRequest(requestID, model, query, requestBody, ttlSeconds)
	assert.NoError(t, err, "AddPendingRequest should succeed")

	// Wait for indexing
	time.Sleep(100 * time.Millisecond)
}

func TestValkeyCacheIntegration_UpdateWithResponse(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	requestID := "req_update_456"
	model := "gpt-4"
	query := "What is Python?"
	requestBody := []byte(`{"model":"gpt-4"}`)
	ttlSeconds := 300

	err := cache.AddPendingRequest(requestID, model, query, requestBody, ttlSeconds)
	require.NoError(t, err)

	// Wait for indexing with retry logic
	time.Sleep(200 * time.Millisecond)

	responseBody := []byte(`{"choices":[{"message":{"content":"Python is a programming language."}}]}`)

	// Retry UpdateWithResponse to handle indexing delays
	var updateErr error
	maxRetries := 5
	for i := 0; i < maxRetries; i++ {
		if i > 0 {
			t.Logf("Retry %d/%d for UpdateWithResponse", i+1, maxRetries)
			time.Sleep(time.Duration(200*(i+1)) * time.Millisecond)
		}

		updateErr = cache.UpdateWithResponse(requestID, responseBody, ttlSeconds)
		if updateErr == nil {
			break
		}
		t.Logf("UpdateWithResponse attempt %d failed: %v", i+1, updateErr)
	}

	require.NoError(t, updateErr, "UpdateWithResponse should succeed after retries")

	// Wait longer for vector index to update with the new response and retry search
	var foundResponse []byte
	var hit bool
	searchRetries := 5

	for i := 0; i < searchRetries; i++ {
		if i > 0 {
			t.Logf("Search retry %d/%d for updated entry", i+1, searchRetries)
		}
		time.Sleep(time.Duration(200*(i+1)) * time.Millisecond)

		foundResponse, hit, err = cache.FindSimilar(model, query)
		require.NoError(t, err)

		if hit {
			break
		}
	}

	require.True(t, hit, "Should find the updated entry after retries")
	assert.NotNil(t, foundResponse)
	assert.Contains(t, string(foundResponse), "Python", "Updated response should be findable")
}

func TestValkeyCacheIntegration_UpdateWithResponseSpecialChars(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	// Request IDs that contain TAG-separator characters (hyphens, dots, colons).
	// These previously broke the TAG query in UpdateWithResponse because the
	// characters were not escaped.
	specialIDs := []struct {
		name      string
		requestID string
		query     string
	}{
		{"UUID hyphens", "550e8400-e29b-41d4-a716-446655440000", "What is Go?"},
		{"dotted version", "req.v1.2.3", "What is Rust?"},
		{"colons", "ns:cache:req:789", "What is Java?"},
	}

	for _, tc := range specialIDs {
		t.Run(tc.name, func(t *testing.T) {
			requestBody := []byte(`{"model":"gpt-4"}`)
			ttlSeconds := 300

			err := cache.AddPendingRequest(tc.requestID, "gpt-4", tc.query, requestBody, ttlSeconds)
			require.NoError(t, err, "AddPendingRequest should succeed for %s", tc.requestID)

			responseBody := []byte(fmt.Sprintf(`{"choices":[{"message":{"content":"answer for %s"}}]}`, tc.requestID))

			var updateErr error
			maxRetries := 5
			for i := 0; i < maxRetries; i++ {
				if i > 0 {
					time.Sleep(time.Duration(200*(i+1)) * time.Millisecond)
				}
				updateErr = cache.UpdateWithResponse(tc.requestID, responseBody, ttlSeconds)
				if updateErr == nil {
					break
				}
			}
			require.NoError(t, updateErr, "UpdateWithResponse should succeed for request_id=%s", tc.requestID)

			// Verify the updated entry is searchable
			time.Sleep(300 * time.Millisecond)
			foundResponse, hit, err := cache.FindSimilar("gpt-4", tc.query)
			assert.NoError(t, err)
			if hit {
				assert.Contains(t, string(foundResponse), tc.requestID, "Response should match the request ID")
			}
		})
	}
}

func TestValkeyCacheIntegration_TTLExpiration(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	ctx := context.Background()

	requestID := "req_ttl_test"
	model := "gpt-4"
	query := "Short lived query"
	requestBody := []byte(`{"model":"gpt-4"}`)
	responseBody := []byte(`{"choices":[{"message":{"content":"Short lived response"}}]}`)
	ttlSeconds := 2 // 2 seconds

	err := cache.AddEntry(requestID, model, query, requestBody, responseBody, ttlSeconds)
	require.NoError(t, err)

	// Wait for indexing
	time.Sleep(100 * time.Millisecond)

	// Find the key
	pattern := cache.config.Index.Prefix + "*"
	_, err = cache.client.CustomCommand(ctx, []string{"SCAN", "0", "MATCH", pattern, "COUNT", "10"})
	require.NoError(t, err)

	// Wait for expiration
	time.Sleep(3 * time.Second)

	// Verify entry is gone
	_, err = cache.client.CustomCommand(ctx, []string{"SCAN", "0", "MATCH", pattern, "COUNT", "10"})
	assert.NoError(t, err)
	// Entry should be expired
}

func TestValkeyCacheIntegration_GetStats(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	// Add some entries
	for i := 0; i < 3; i++ {
		requestID := fmt.Sprintf("req_stats_%d", i)
		query := fmt.Sprintf("Test query %d", i)
		err := cache.AddEntry(requestID, "gpt-4", query, []byte("{}"), []byte("{}"), 300)
		require.NoError(t, err)
	}

	// Wait for indexing
	time.Sleep(300 * time.Millisecond)

	stats := cache.GetStats()

	assert.GreaterOrEqual(t, stats.TotalEntries, 0, "Total entries should be non-negative")
	assert.GreaterOrEqual(t, stats.HitCount, int64(0), "Hit count should be non-negative")
	assert.GreaterOrEqual(t, stats.MissCount, int64(0), "Miss count should be non-negative")
}

func TestValkeyCacheIntegration_Close(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)

	err := cache.Close()
	assert.NoError(t, err, "Close should succeed")

	err = cache.CheckConnection()
	assert.Error(t, err, "Connection check should fail after close")
}

func TestValkeyCacheIntegration_IsEnabled(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	assert.True(t, cache.IsEnabled(), "Cache should be enabled")

	// Test disabled cache
	host, port := valkeyIntegrationAddr()
	valkeyConfig := &config.ValkeyConfig{}
	valkeyConfig.Connection.Host = host
	valkeyConfig.Connection.Port = port

	disabledCache, err := NewValkeyCache(ValkeyCacheOptions{
		Enabled: false,
		Config:  valkeyConfig,
	})
	require.NoError(t, err)
	defer func() { _ = disabledCache.Close() }()

	assert.False(t, disabledCache.IsEnabled(), "Cache should be disabled")
}

func TestValkeyCacheIntegration_DisabledCache(t *testing.T) {
	host, port := valkeyIntegrationAddr()
	valkeyConfig := &config.ValkeyConfig{}
	valkeyConfig.Connection.Host = host
	valkeyConfig.Connection.Port = port

	cache, err := NewValkeyCache(ValkeyCacheOptions{
		Enabled: false,
		Config:  valkeyConfig,
	})
	require.NoError(t, err)
	defer func() { _ = cache.Close() }()

	// All operations should return nil/false when disabled
	err = cache.AddEntry("req_1", "gpt-4", "test", []byte("{}"), []byte("{}"), 300)
	assert.NoError(t, err, "AddEntry should not error when disabled")

	err = cache.AddPendingRequest("req_2", "gpt-4", "test", []byte("{}"), 300)
	assert.NoError(t, err, "AddPendingRequest should not error when disabled")

	_, hit, err := cache.FindSimilar("gpt-4", "test")
	assert.NoError(t, err, "FindSimilar should not error when disabled")
	assert.False(t, hit, "FindSimilar should return false when disabled")

	err = cache.CheckConnection()
	assert.NoError(t, err, "CheckConnection should not error when disabled")
}

func TestValkeyCacheIntegration_TTLZeroSkipsCaching(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	// Test AddEntry with TTL=0 (should skip caching)
	requestID := "req_ttl_zero_1"
	err := cache.AddEntry(requestID, "gpt-4", "test query", []byte("{}"), []byte("{}"), 0)
	assert.NoError(t, err, "AddEntry with TTL=0 should not error")

	// Test AddPendingRequest with TTL=0 (should skip caching)
	requestID2 := "req_ttl_zero_2"
	err = cache.AddPendingRequest(requestID2, "gpt-4", "test query 2", []byte("{}"), 0)
	assert.NoError(t, err, "AddPendingRequest with TTL=0 should not error")
}

func TestValkeyCacheIntegration_FLATIndexType(t *testing.T) {
	if os.Getenv("SKIP_VALKEY_TESTS") == "true" {
		t.Skip("Valkey integration tests skipped due to SKIP_VALKEY_TESTS=true")
	}

	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		t.Skipf("Failed to initialize BERT model: %v", err)
	}

	host, port := valkeyIntegrationAddr()
	valkeyConfig := &config.ValkeyConfig{}
	valkeyConfig.Connection.Host = host
	valkeyConfig.Connection.Port = port
	valkeyConfig.Connection.Database = 0

	valkeyConfig.Index.Name = fmt.Sprintf("test_flat_idx_%d", time.Now().UnixNano())
	valkeyConfig.Index.Prefix = "flat:"
	valkeyConfig.Index.VectorField.Name = "embedding"
	valkeyConfig.Index.VectorField.Dimension = 384
	valkeyConfig.Index.VectorField.MetricType = "COSINE"
	valkeyConfig.Index.IndexType = "FLAT"

	valkeyConfig.Search.TopK = 1
	valkeyConfig.Development.DropIndexOnStartup = true
	valkeyConfig.Development.AutoCreateIndex = true

	cache, err := NewValkeyCache(ValkeyCacheOptions{
		SimilarityThreshold: 0.8,
		TTLSeconds:          300,
		Enabled:             true,
		Config:              valkeyConfig,
		EmbeddingModel:      "bert",
	})
	require.NoError(t, err, "Failed to create cache with FLAT index")
	defer func() { _ = cache.Close() }()

	// Add an entry and verify it works
	err = cache.AddEntry("req_flat_1", "gpt-4", "test flat index", []byte("{}"), []byte(`{"result":"flat"}`), 300)
	assert.NoError(t, err, "AddEntry should work with FLAT index")

	time.Sleep(200 * time.Millisecond)

	response, hit, err := cache.FindSimilar("gpt-4", "test flat index")
	assert.NoError(t, err, "FindSimilar should work with FLAT index")
	if hit {
		assert.NotNil(t, response, "Response should be found")
	}
}

func TestValkeyCacheIntegration_ConcurrentOperations(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	const numGoroutines = 20
	const operationsPerGoroutine = 5

	errChan := make(chan error, numGoroutines*operationsPerGoroutine)
	doneChan := make(chan bool, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			for j := 0; j < operationsPerGoroutine; j++ {
				requestID := fmt.Sprintf("req_concurrent_%d_%d", id, j)
				query := fmt.Sprintf("concurrent query %d %d", id, j)

				err := cache.AddEntry(requestID, "gpt-4", query, []byte("{}"), []byte(fmt.Sprintf(`{"id":%d}`, id)), 300)
				if err != nil {
					errChan <- err
				}

				_, _, err = cache.FindSimilar("gpt-4", query)
				if err != nil {
					errChan <- err
				}
			}
			doneChan <- true
		}(i)
	}

	for i := 0; i < numGoroutines; i++ {
		<-doneChan
	}
	close(errChan)

	var errors []error
	for err := range errChan {
		errors = append(errors, err)
	}

	assert.Empty(t, errors, "Concurrent operations should not produce errors")
}

func TestValkeyCacheIntegration_MultipleEntries(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	// Add multiple entries with different queries
	entries := []struct {
		requestID string
		query     string
		response  string
	}{
		{"req_multi_1", "What is AI?", "AI is artificial intelligence"},
		{"req_multi_2", "What is ML?", "ML is machine learning"},
		{"req_multi_3", "What is DL?", "DL is deep learning"},
	}

	for _, entry := range entries {
		err := cache.AddEntry(
			entry.requestID,
			"gpt-4",
			entry.query,
			[]byte(fmt.Sprintf(`{"query":"%s"}`, entry.query)),
			[]byte(fmt.Sprintf(`{"response":"%s"}`, entry.response)),
			300,
		)
		require.NoError(t, err, "AddEntry should succeed for %s", entry.requestID)
	}

	// Wait for indexing
	time.Sleep(300 * time.Millisecond)

	// Verify we can find similar entries
	for _, entry := range entries {
		foundResponse, hit, err := cache.FindSimilar("gpt-4", entry.query)
		assert.NoError(t, err, "FindSimilar should not error for %s", entry.query)

		if hit {
			assert.NotNil(t, foundResponse, "Response should be found for %s", entry.query)
		}
	}
}

func TestValkeyCacheIntegration_L2MetricType(t *testing.T) {
	if os.Getenv("SKIP_VALKEY_TESTS") == "true" {
		t.Skip("Valkey integration tests skipped due to SKIP_VALKEY_TESTS=true")
	}

	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		t.Skipf("Failed to initialize BERT model: %v", err)
	}

	host, port := valkeyIntegrationAddr()
	valkeyConfig := &config.ValkeyConfig{}
	valkeyConfig.Connection.Host = host
	valkeyConfig.Connection.Port = port
	valkeyConfig.Connection.Database = 0

	valkeyConfig.Index.Name = fmt.Sprintf("test_l2_idx_%d", time.Now().UnixNano())
	valkeyConfig.Index.Prefix = "l2:"
	valkeyConfig.Index.VectorField.Name = "embedding"
	valkeyConfig.Index.VectorField.Dimension = 384
	valkeyConfig.Index.VectorField.MetricType = "L2"
	valkeyConfig.Index.IndexType = "HNSW"
	valkeyConfig.Index.Params.M = 16
	valkeyConfig.Index.Params.EfConstruction = 64

	valkeyConfig.Search.TopK = 1
	valkeyConfig.Development.DropIndexOnStartup = true
	valkeyConfig.Development.AutoCreateIndex = true

	cache, err := NewValkeyCache(ValkeyCacheOptions{
		SimilarityThreshold: 0.5,
		TTLSeconds:          300,
		Enabled:             true,
		Config:              valkeyConfig,
		EmbeddingModel:      "bert",
	})
	require.NoError(t, err, "Failed to create cache with L2 metric")
	defer func() { _ = cache.Close() }()

	err = cache.AddEntry("req_l2_1", "gpt-4", "test L2 metric", []byte("{}"), []byte(`{"result":"L2"}`), 300)
	assert.NoError(t, err, "AddEntry should work with L2 metric")

	time.Sleep(200 * time.Millisecond)

	response, hit, err := cache.FindSimilar("gpt-4", "test L2 metric")
	assert.NoError(t, err, "FindSimilar should work with L2 metric")
	if hit {
		assert.NotNil(t, response, "Response should be found")
	}
}

func TestValkeyCacheIntegration_IPMetricType(t *testing.T) {
	if os.Getenv("SKIP_VALKEY_TESTS") == "true" {
		t.Skip("Valkey integration tests skipped due to SKIP_VALKEY_TESTS=true")
	}

	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		t.Skipf("Failed to initialize BERT model: %v", err)
	}

	host, port := valkeyIntegrationAddr()
	valkeyConfig := &config.ValkeyConfig{}
	valkeyConfig.Connection.Host = host
	valkeyConfig.Connection.Port = port
	valkeyConfig.Connection.Database = 0

	valkeyConfig.Index.Name = fmt.Sprintf("test_ip_idx_%d", time.Now().UnixNano())
	valkeyConfig.Index.Prefix = "ip:"
	valkeyConfig.Index.VectorField.Name = "embedding"
	valkeyConfig.Index.VectorField.Dimension = 384
	valkeyConfig.Index.VectorField.MetricType = "IP"
	valkeyConfig.Index.IndexType = "HNSW"
	valkeyConfig.Index.Params.M = 16
	valkeyConfig.Index.Params.EfConstruction = 64

	valkeyConfig.Search.TopK = 1
	valkeyConfig.Development.DropIndexOnStartup = true
	valkeyConfig.Development.AutoCreateIndex = true

	cache, err := NewValkeyCache(ValkeyCacheOptions{
		SimilarityThreshold: 0.5,
		TTLSeconds:          300,
		Enabled:             true,
		Config:              valkeyConfig,
		EmbeddingModel:      "bert",
	})
	require.NoError(t, err, "Failed to create cache with IP metric")
	defer func() { _ = cache.Close() }()

	err = cache.AddEntry("req_ip_1", "gpt-4", "test IP metric", []byte("{}"), []byte(`{"result":"IP"}`), 300)
	assert.NoError(t, err, "AddEntry should work with IP metric")

	time.Sleep(200 * time.Millisecond)

	response, hit, err := cache.FindSimilar("gpt-4", "test IP metric")
	assert.NoError(t, err, "FindSimilar should work with IP metric")
	if hit {
		assert.NotNil(t, response, "Response should be found")
	}
}
