//go:build !windows && cgo

package cache

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestValkeyCacheIntegration_EmptyQuery(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	err := cache.AddEntry("req_empty", "gpt-4", "", []byte("{}"), []byte("{}"), 300)
	assert.NoError(t, err, "AddEntry with empty query should not error")

	_, hit, err := cache.FindSimilar("gpt-4", "")
	assert.NoError(t, err, "FindSimilar with empty query should not error")
	t.Logf("Empty query search hit: %v", hit)
}

func TestValkeyCacheIntegration_LargeResponseBody(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	largeResponse := make([]byte, 1024*1024)
	for i := range largeResponse {
		largeResponse[i] = byte('A' + (i % 26))
	}

	err := cache.AddEntry("req_large", "gpt-4", "large response test", []byte("{}"), largeResponse, 300)
	assert.NoError(t, err, "AddEntry with large response should succeed")

	time.Sleep(200 * time.Millisecond)

	response, hit, err := cache.FindSimilar("gpt-4", "large response test")
	assert.NoError(t, err, "FindSimilar should work with large response")
	if hit {
		assert.NotNil(t, response, "Large response should be retrievable")
		assert.Len(t, response, len(largeResponse), "Response size should match")
	}
}

func TestValkeyCacheIntegration_SpecialCharactersInQuery(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	specialQueries := []string{
		`query with "quotes"`,
		`query with 'single quotes'`,
		`query with @special #chars $and %symbols`,
		`query with \backslash`,
		`query with (parentheses) and [brackets]`,
		`query with {braces}`,
		`query with * asterisk`,
		`query with | pipe`,
	}

	for i, query := range specialQueries {
		requestID := fmt.Sprintf("req_special_%d", i)
		err := cache.AddEntry(requestID, "gpt-4", query, []byte("{}"), []byte(fmt.Sprintf(`{"query":"%d"}`, i)), 300)
		assert.NoError(t, err, "AddEntry should handle special characters: %s", query)
	}

	time.Sleep(300 * time.Millisecond)

	for _, query := range specialQueries {
		_, _, err := cache.FindSimilar("gpt-4", query)
		assert.NoError(t, err, "FindSimilar should handle special characters: %s", query)
	}
}

func TestValkeyCacheIntegration_StatsAccuracy(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	for i := 0; i < 5; i++ {
		requestID := fmt.Sprintf("req_stats_acc_%d", i)
		query := fmt.Sprintf("stats test query %d", i)
		err := cache.AddEntry(requestID, "gpt-4", query, []byte("{}"), []byte(fmt.Sprintf(`{"id":%d}`, i)), 300)
		require.NoError(t, err)
	}

	time.Sleep(300 * time.Millisecond)

	_, _, _ = cache.FindSimilar("gpt-4", "stats test query 0")
	_, _, _ = cache.FindSimilar("gpt-4", "stats test query 1")
	_, _, _ = cache.FindSimilar("gpt-4", "completely different query xyz")

	time.Sleep(100 * time.Millisecond)

	stats := cache.GetStats()

	assert.GreaterOrEqual(t, stats.TotalEntries, 5, "Should have at least 5 entries")
	assert.GreaterOrEqual(t, stats.HitCount+stats.MissCount, int64(3), "Should have at least 3 searches")

	if stats.HitCount+stats.MissCount > 0 {
		assert.GreaterOrEqual(t, stats.HitRatio, 0.0, "Hit ratio should be >= 0")
		assert.LessOrEqual(t, stats.HitRatio, 1.0, "Hit ratio should be <= 1")
	}

	t.Logf("Stats: TotalEntries=%d, HitCount=%d, MissCount=%d, HitRatio=%.2f",
		stats.TotalEntries, stats.HitCount, stats.MissCount, stats.HitRatio)
}

func TestValkeyCacheIntegration_ErrorScenarios(t *testing.T) {
	t.Run("Invalid Valkey host", func(t *testing.T) {
		valkeyConfig := &config.ValkeyConfig{}
		valkeyConfig.Connection.Host = "invalid-host-that-does-not-exist"
		valkeyConfig.Connection.Port = 6379
		valkeyConfig.Connection.Timeout = 1

		valkeyConfig.Index.Name = "test_idx"
		valkeyConfig.Index.Prefix = "doc:"
		valkeyConfig.Index.VectorField.Name = "embedding"
		valkeyConfig.Index.VectorField.Dimension = 384
		valkeyConfig.Index.VectorField.MetricType = "COSINE"
		valkeyConfig.Index.IndexType = "HNSW"
		valkeyConfig.Index.Params.M = 16
		valkeyConfig.Index.Params.EfConstruction = 64
		valkeyConfig.Development.AutoCreateIndex = true

		_, err := NewValkeyCache(ValkeyCacheOptions{
			Enabled:        true,
			Config:         valkeyConfig,
			EmbeddingModel: "bert",
		})
		assert.Error(t, err, "Should fail to connect to invalid host")
	})

	t.Run("Index creation disabled when index doesn't exist", func(t *testing.T) {
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

		valkeyConfig.Index.Name = "nonexistent_idx"
		valkeyConfig.Index.Prefix = "doc:"
		valkeyConfig.Index.VectorField.Name = "embedding"
		valkeyConfig.Index.VectorField.Dimension = 384
		valkeyConfig.Index.VectorField.MetricType = "COSINE"
		valkeyConfig.Index.IndexType = "HNSW"
		valkeyConfig.Index.Params.M = 16
		valkeyConfig.Index.Params.EfConstruction = 64

		valkeyConfig.Development.DropIndexOnStartup = true
		valkeyConfig.Development.AutoCreateIndex = false

		_, err := NewValkeyCache(ValkeyCacheOptions{
			Enabled:        true,
			Config:         valkeyConfig,
			EmbeddingModel: "bert",
		})
		assert.Error(t, err, "Should fail when index doesn't exist and auto-creation is disabled")
		assert.Contains(t, err.Error(), "does not exist", "Error should mention index doesn't exist")
	})
}
