package cache_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/prometheus/client_golang/prometheus/testutil"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/metrics"
)

func TestCache(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Cache Suite")
}

var _ = BeforeSuite(func() {
	// Initialize BERT model once for all cache tests (Linux only)
	err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true)
	Expect(err).NotTo(HaveOccurred())
})

var _ = Describe("Cache Package", func() {
	var tempDir string

	BeforeEach(func() {
		var err error
		tempDir, err = os.MkdirTemp("", "cache_test")
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		os.RemoveAll(tempDir)
	})

	Describe("Cache Factory", func() {
		Describe("NewCacheBackend", func() {
			Context("with memory backend", func() {
				It("should create in-memory cache backend successfully", func() {
					config := cache.CacheConfig{
						BackendType:         cache.InMemoryCacheType,
						Enabled:             true,
						SimilarityThreshold: 0.8,
						MaxEntries:          1000,
						TTLSeconds:          3600,
					}

					backend, err := cache.NewCacheBackend(config)
					Expect(err).NotTo(HaveOccurred())
					Expect(backend).NotTo(BeNil())
					Expect(backend.IsEnabled()).To(BeTrue())
				})

				It("should create disabled cache when enabled is false", func() {
					config := cache.CacheConfig{
						BackendType:         cache.InMemoryCacheType,
						Enabled:             false,
						SimilarityThreshold: 0.8,
						MaxEntries:          1000,
						TTLSeconds:          3600,
					}

					backend, err := cache.NewCacheBackend(config)
					Expect(err).NotTo(HaveOccurred())
					Expect(backend).NotTo(BeNil())
					Expect(backend.IsEnabled()).To(BeFalse())
				})

				It("should default to memory backend when backend_type is empty", func() {
					config := cache.CacheConfig{
						BackendType:         "", // Empty should default to memory
						Enabled:             true,
						SimilarityThreshold: 0.8,
						MaxEntries:          500,
						TTLSeconds:          1800,
					}

					backend, err := cache.NewCacheBackend(config)
					Expect(err).NotTo(HaveOccurred())
					Expect(backend).NotTo(BeNil())
					Expect(backend.IsEnabled()).To(BeTrue())
				})
			})

			Context("with Milvus backend", func() {
				var milvusConfigPath string

				BeforeEach(func() {
					// Skip Milvus tests if environment variable is set
					if os.Getenv("SKIP_MILVUS_TESTS") == "true" {
						Skip("Milvus tests skipped due to SKIP_MILVUS_TESTS=true")
					}

					// Create a test Milvus configuration file
					milvusConfigPath = filepath.Join(tempDir, "milvus.yaml")
					milvusConfig := `
connection:
  host: "localhost"
  port: 19530
  database: "test_cache"
  timeout: 30

collection:
  name: "test_semantic_cache"
  description: "Test semantic cache collection"
  vector_field:
    name: "embedding"
    dimension: 512
    metric_type: "IP"
  index:
    type: "HNSW"
    params:
      M: 16
      efConstruction: 64

search:
  params:
    ef: 64
  topk: 10
  consistency_level: "Session"

development:
  auto_create_collection: true
  verbose_errors: true
`
					err := os.WriteFile(milvusConfigPath, []byte(milvusConfig), 0o644)
					Expect(err).NotTo(HaveOccurred())
				})

				It("should create Milvus cache backend successfully with valid config", func() {
					config := cache.CacheConfig{
						BackendType:         cache.MilvusCacheType,
						Enabled:             true,
						SimilarityThreshold: 0.85,
						TTLSeconds:          7200,
						BackendConfigPath:   milvusConfigPath,
					}

					backend, err := cache.NewCacheBackend(config)

					// Skip test if Milvus is not reachable
					if err != nil {
						if strings.Contains(err.Error(), "failed to create Milvus client") ||
							strings.Contains(err.Error(), "connection") ||
							strings.Contains(err.Error(), "dial") {
							Skip("Milvus server not available: " + err.Error())
						}
						// For other errors, fail the test
						Expect(err).NotTo(HaveOccurred())
					} else {
						// If Milvus is available, creation should succeed
						Expect(backend).NotTo(BeNil())
						Expect(backend.IsEnabled()).To(BeTrue())
					}
				})

				It("should handle disabled Milvus cache", func() {
					config := cache.CacheConfig{
						BackendType:         cache.MilvusCacheType,
						Enabled:             false,
						SimilarityThreshold: 0.8,
						TTLSeconds:          3600,
						BackendConfigPath:   milvusConfigPath,
					}

					backend, err := cache.NewCacheBackend(config)
					Expect(err).NotTo(HaveOccurred())
					Expect(backend).NotTo(BeNil())
					Expect(backend.IsEnabled()).To(BeFalse())
				})
			})

			Context("with unsupported backend type", func() {
				It("should return error for unsupported backend type", func() {
					config := cache.CacheConfig{
						BackendType:         "redis", // Unsupported
						Enabled:             true,
						SimilarityThreshold: 0.8,
						TTLSeconds:          3600,
					}

					backend, err := cache.NewCacheBackend(config)
					Expect(err).To(HaveOccurred())
					Expect(err.Error()).To(ContainSubstring("unsupported cache backend type"))
					Expect(backend).To(BeNil())
				})
			})

			Context("with invalid config but valid backend type", func() {
				It("should return error due to validation when config has invalid values", func() {
					config := cache.CacheConfig{
						BackendType:         cache.InMemoryCacheType, // valid backend type
						Enabled:             true,
						SimilarityThreshold: -0.8, // invalid
						MaxEntries:          10,
						TTLSeconds:          -1, // invalid
					}

					backend, err := cache.NewCacheBackend(config)

					Expect(err).To(HaveOccurred())
					Expect(err.Error()).To(ContainSubstring("invalid cache config")) // ensure from config validation
					Expect(backend).To(BeNil())
				})
			})
		})

		Describe("ValidateCacheConfig", func() {
			It("should validate enabled memory backend configuration", func() {
				config := cache.CacheConfig{
					BackendType:         cache.InMemoryCacheType,
					Enabled:             true,
					SimilarityThreshold: 0.8,
					MaxEntries:          1000,
					TTLSeconds:          3600,
					EvictionPolicy:      "lru",
				}

				err := cache.ValidateCacheConfig(config)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should validate disabled cache configuration", func() {
				config := cache.CacheConfig{
					BackendType:         cache.InMemoryCacheType,
					Enabled:             false,
					SimilarityThreshold: 2.0, // Invalid, but should be ignored for disabled cache
					MaxEntries:          -1,  // Invalid, but should be ignored for disabled cache
				}

				err := cache.ValidateCacheConfig(config)
				Expect(err).NotTo(HaveOccurred()) // Disabled cache should skip validation
			})

			It("should return error for invalid similarity threshold", func() {
				config := cache.CacheConfig{
					BackendType:         cache.InMemoryCacheType,
					Enabled:             true,
					SimilarityThreshold: 1.5, // Invalid: > 1.0
					MaxEntries:          1000,
					TTLSeconds:          3600,
				}

				err := cache.ValidateCacheConfig(config)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("similarity_threshold must be between 0.0 and 1.0"))
			})

			It("should return error for negative similarity threshold", func() {
				config := cache.CacheConfig{
					BackendType:         cache.InMemoryCacheType,
					Enabled:             true,
					SimilarityThreshold: -0.1, // Invalid: < 0.0
					MaxEntries:          1000,
					TTLSeconds:          3600,
				}

				err := cache.ValidateCacheConfig(config)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("similarity_threshold must be between 0.0 and 1.0"))
			})

			It("should return error for negative TTL", func() {
				config := cache.CacheConfig{
					BackendType:         cache.InMemoryCacheType,
					Enabled:             true,
					SimilarityThreshold: 0.8,
					MaxEntries:          1000,
					TTLSeconds:          -1, // Invalid: negative TTL
				}

				err := cache.ValidateCacheConfig(config)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("ttl_seconds cannot be negative"))
			})

			It("should return error for negative max entries in memory backend", func() {
				config := cache.CacheConfig{
					BackendType:         cache.InMemoryCacheType,
					Enabled:             true,
					SimilarityThreshold: 0.8,
					MaxEntries:          -1, // Invalid: negative max entries
					TTLSeconds:          3600,
				}

				err := cache.ValidateCacheConfig(config)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("max_entries cannot be negative"))
			})

			It("should return error for unsupported eviction_policy value in memory backend", func() {
				config := cache.CacheConfig{
					BackendType:         cache.InMemoryCacheType,
					Enabled:             true,
					SimilarityThreshold: 0.8,
					MaxEntries:          1000,
					TTLSeconds:          3600,
					EvictionPolicy:      "random", // unsupported
				}

				err := cache.ValidateCacheConfig(config)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("unsupported eviction_policy"))
			})

			It("should return error for Milvus backend without config path", func() {
				config := cache.CacheConfig{
					BackendType:         cache.MilvusCacheType,
					Enabled:             true,
					SimilarityThreshold: 0.8,
					TTLSeconds:          3600,
					// BackendConfigPath is missing
				}

				err := cache.ValidateCacheConfig(config)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("backend_config_path is required for Milvus"))
			})

			It("should return error when Milvus backend_config_path file doesn't exist", func() {
				config := cache.CacheConfig{
					BackendType:         cache.MilvusCacheType,
					Enabled:             true,
					SimilarityThreshold: 0.8,
					TTLSeconds:          3600,
					BackendConfigPath:   "/nonexistent/milvus.yaml",
				}

				err := cache.ValidateCacheConfig(config)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("config file not found"))
			})

			It("should validate edge case values", func() {
				config := cache.CacheConfig{
					BackendType:         cache.InMemoryCacheType,
					Enabled:             true,
					SimilarityThreshold: 0.0, // Valid: minimum threshold
					MaxEntries:          0,   // Valid: unlimited entries
					TTLSeconds:          0,   // Valid: no expiration
				}

				err := cache.ValidateCacheConfig(config)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should validate maximum threshold value", func() {
				config := cache.CacheConfig{
					BackendType:         cache.InMemoryCacheType,
					Enabled:             true,
					SimilarityThreshold: 1.0, // Valid: maximum threshold
					MaxEntries:          10000,
					TTLSeconds:          86400,
				}

				err := cache.ValidateCacheConfig(config)
				Expect(err).NotTo(HaveOccurred())
			})
		})

		Describe("GetDefaultCacheConfig", func() {
			It("should return valid default configuration", func() {
				config := cache.GetDefaultCacheConfig()

				Expect(config.BackendType).To(Equal(cache.InMemoryCacheType))
				Expect(config.Enabled).To(BeTrue())
				Expect(config.SimilarityThreshold).To(Equal(float32(0.8)))
				Expect(config.MaxEntries).To(Equal(1000))
				Expect(config.TTLSeconds).To(Equal(3600))
				Expect(config.BackendConfigPath).To(BeEmpty())

				// Default config should pass validation
				err := cache.ValidateCacheConfig(config)
				Expect(err).NotTo(HaveOccurred())
			})
		})

		Describe("GetAvailableCacheBackends", func() {
			It("should return information about available backends", func() {
				backends := cache.GetAvailableCacheBackends()

				Expect(backends).To(HaveLen(2)) // Memory and Milvus

				// Check memory backend info
				memoryBackend := backends[0]
				Expect(memoryBackend.Type).To(Equal(cache.InMemoryCacheType))
				Expect(memoryBackend.Name).To(Equal("In-Memory Cache"))
				Expect(memoryBackend.Description).To(ContainSubstring("in-memory semantic cache"))
				Expect(memoryBackend.Features).To(ContainElement("Fast access"))
				Expect(memoryBackend.Features).To(ContainElement("No external dependencies"))

				// Check Milvus backend info
				milvusBackend := backends[1]
				Expect(milvusBackend.Type).To(Equal(cache.MilvusCacheType))
				Expect(milvusBackend.Name).To(Equal("Milvus Vector Database"))
				Expect(milvusBackend.Description).To(ContainSubstring("Milvus vector database"))
				Expect(milvusBackend.Features).To(ContainElement("Highly scalable"))
				Expect(milvusBackend.Features).To(ContainElement("Persistent storage"))
			})
		})
	})

	Describe("InMemoryCache", func() {
		var inMemoryCache cache.CacheBackend

		BeforeEach(func() {
			options := cache.InMemoryCacheOptions{
				Enabled:             true,
				SimilarityThreshold: 0.8,
				MaxEntries:          100,
				TTLSeconds:          300,
			}
			inMemoryCache = cache.NewInMemoryCache(options)
		})

		AfterEach(func() {
			if inMemoryCache != nil {
				inMemoryCache.Close()
			}
			// BERT model is initialized once per process, no need to reset
		})

		It("should implement CacheBackend interface", func() {
			// Check that the concrete type implements the interface
			_ = inMemoryCache
			Expect(inMemoryCache).NotTo(BeNil())
		})

		It("should report enabled status correctly", func() {
			Expect(inMemoryCache.IsEnabled()).To(BeTrue())

			// Create disabled cache
			disabledOptions := cache.InMemoryCacheOptions{
				Enabled:             false,
				SimilarityThreshold: 0.8,
				MaxEntries:          100,
				TTLSeconds:          300,
			}
			disabledCache := cache.NewInMemoryCache(disabledOptions)
			defer disabledCache.Close()

			Expect(disabledCache.IsEnabled()).To(BeFalse())
		})

		It("should handle basic cache operations without embeddings", func() {
			// Test GetStats on empty cache
			stats := inMemoryCache.GetStats()
			Expect(stats.TotalEntries).To(Equal(0))
			Expect(stats.HitCount).To(Equal(int64(0)))
			Expect(stats.MissCount).To(Equal(int64(0)))
			Expect(stats.HitRatio).To(Equal(0.0))
		})

		It("should handle AddEntry operation with embeddings", func() {
			err := inMemoryCache.AddEntry("test-request-id", "test-model", "test query", []byte("request"), []byte("response"))
			Expect(err).NotTo(HaveOccurred())

			stats := inMemoryCache.GetStats()
			Expect(stats.TotalEntries).To(Equal(1))
		})

		It("should handle FindSimilar operation with embeddings", func() {
			// First add an entry
			err := inMemoryCache.AddEntry("test-request-id", "test-model", "test query", []byte("request"), []byte("response"))
			Expect(err).NotTo(HaveOccurred())

			// Search for similar query
			response, found, err := inMemoryCache.FindSimilar("test-model", "test query")
			Expect(err).NotTo(HaveOccurred())
			Expect(found).To(BeTrue()) // Should find exact match
			Expect(response).To(Equal([]byte("response")))

			// Search for different model (should not match)
			response, found, err = inMemoryCache.FindSimilar("different-model", "test query")
			Expect(err).NotTo(HaveOccurred())
			Expect(found).To(BeFalse()) // Should not match different model
			Expect(response).To(BeNil())
		})

		It("should handle AddPendingRequest and UpdateWithResponse", func() {
			err := inMemoryCache.AddPendingRequest("test-request-id", "test-model", "test query", []byte("request"))
			Expect(err).NotTo(HaveOccurred())

			// Update with response
			err = inMemoryCache.UpdateWithResponse("test-request-id", []byte("response"))
			Expect(err).NotTo(HaveOccurred())

			// Should now be able to find it
			response, found, err := inMemoryCache.FindSimilar("test-model", "test query")
			Expect(err).NotTo(HaveOccurred())
			Expect(found).To(BeTrue())
			Expect(response).To(Equal([]byte("response")))
		})

		It("should update cache entries metric when cleanup occurs during UpdateWithResponse", func() {
			// Reset gauge defensively so the assertion stands alone even if other specs fail early
			metrics.UpdateCacheEntries("memory", 0)

			Expect(inMemoryCache.Close()).NotTo(HaveOccurred())
			inMemoryCache = cache.NewInMemoryCache(cache.InMemoryCacheOptions{
				Enabled:             true,
				SimilarityThreshold: 0.8,
				MaxEntries:          100,
				TTLSeconds:          1,
			})

			err := inMemoryCache.AddPendingRequest("expired-request-id", "test-model", "stale query", []byte("request"))
			Expect(err).NotTo(HaveOccurred())
			Expect(testutil.ToFloat64(metrics.CacheEntriesTotal.WithLabelValues("memory"))).To(Equal(float64(1)))

			// Wait for TTL to expire before triggering the update path
			time.Sleep(2 * time.Second)

			err = inMemoryCache.UpdateWithResponse("expired-request-id", []byte("response"))
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("no pending request"))

			Expect(testutil.ToFloat64(metrics.CacheEntriesTotal.WithLabelValues("memory"))).To(BeZero())
		})

		It("should respect similarity threshold", func() {
			// Add entry with a very high similarity threshold
			highThresholdOptions := cache.InMemoryCacheOptions{
				Enabled:             true,
				SimilarityThreshold: 0.99, // Very high threshold
				MaxEntries:          100,
				TTLSeconds:          300,
			}
			highThresholdCache := cache.NewInMemoryCache(highThresholdOptions)
			defer highThresholdCache.Close()

			err := highThresholdCache.AddEntry("test-request-id", "test-model", "machine learning", []byte("request"), []byte("ml response"))
			Expect(err).NotTo(HaveOccurred())

			// Exact match should work
			response, found, err := highThresholdCache.FindSimilar("test-model", "machine learning")
			Expect(err).NotTo(HaveOccurred())
			Expect(found).To(BeTrue())
			Expect(response).To(Equal([]byte("ml response")))

			// Different query should not match due to high threshold
			response, found, err = highThresholdCache.FindSimilar("test-model", "artificial intelligence")
			Expect(err).NotTo(HaveOccurred())
			Expect(found).To(BeFalse())
			Expect(response).To(BeNil())
		})

		It("should track hit and miss statistics", func() {
			// Add an entry with a specific query
			err := inMemoryCache.AddEntry("test-request-id", "test-model", "What is machine learning?", []byte("request"), []byte("ML is a subset of AI"))
			Expect(err).NotTo(HaveOccurred())

			// Search for the exact cached query (should be a hit)
			response, found, err := inMemoryCache.FindSimilar("test-model", "What is machine learning?")
			Expect(err).NotTo(HaveOccurred())
			Expect(found).To(BeTrue())
			Expect(response).To(Equal([]byte("ML is a subset of AI")))

			// Search for a completely unrelated query (should be a miss)
			response, found, err = inMemoryCache.FindSimilar("test-model", "How do I cook pasta?")
			Expect(err).NotTo(HaveOccurred())
			Expect(found).To(BeFalse())
			Expect(response).To(BeNil())

			// Check statistics
			stats := inMemoryCache.GetStats()
			Expect(stats.HitCount).To(Equal(int64(1)))
			Expect(stats.MissCount).To(Equal(int64(1)))
			Expect(stats.HitRatio).To(Equal(0.5))
		})

		It("should skip expired entries during similarity search", func() {
			ttlCache := cache.NewInMemoryCache(cache.InMemoryCacheOptions{
				Enabled:             true,
				SimilarityThreshold: 0.1,
				MaxEntries:          10,
				TTLSeconds:          1,
			})
			defer ttlCache.Close()

			err := ttlCache.AddEntry("ttl-request-id", "ttl-model", "time-sensitive query", []byte("request"), []byte("response"))
			Expect(err).NotTo(HaveOccurred())

			time.Sleep(1100 * time.Millisecond)

			response, found, err := ttlCache.FindSimilar("ttl-model", "time-sensitive query")
			Expect(err).NotTo(HaveOccurred())
			Expect(found).To(BeFalse())
			Expect(response).To(BeNil())

			stats := ttlCache.GetStats()
			Expect(stats.HitCount).To(Equal(int64(0)))
			Expect(stats.MissCount).To(Equal(int64(1)))
		})

		It("should handle error when updating non-existent pending request", func() {
			err := inMemoryCache.UpdateWithResponse("non-existent-query", []byte("response"))
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("no pending request found"))
		})

		It("should handle close operation", func() {
			err := inMemoryCache.Close()
			Expect(err).NotTo(HaveOccurred())

			// Stats should show zero entries after close
			stats := inMemoryCache.GetStats()
			Expect(stats.TotalEntries).To(Equal(0))
		})

		It("should handle disabled cache operations gracefully", func() {
			disabledOptions := cache.InMemoryCacheOptions{
				Enabled:             false,
				SimilarityThreshold: 0.8,
				MaxEntries:          100,
				TTLSeconds:          300,
			}
			disabledCache := cache.NewInMemoryCache(disabledOptions)
			defer disabledCache.Close()

			// Disabled cache operations should not error but should be no-ops
			// They should NOT try to generate embeddings
			err := disabledCache.AddPendingRequest("test-request-id", "test-model", "test query", []byte("request"))
			Expect(err).NotTo(HaveOccurred())

			err = disabledCache.UpdateWithResponse("test-request-id", []byte("response"))
			Expect(err).NotTo(HaveOccurred())

			err = disabledCache.AddEntry("test-request-id", "test-model", "test query", []byte("request"), []byte("response"))
			Expect(err).NotTo(HaveOccurred())

			response, found, err := disabledCache.FindSimilar("model", "query")
			Expect(err).NotTo(HaveOccurred())
			Expect(found).To(BeFalse())
			Expect(response).To(BeNil())

			// Stats should show zero activity
			stats := disabledCache.GetStats()
			Expect(stats.TotalEntries).To(Equal(0))
			Expect(stats.HitCount).To(Equal(int64(0)))
			Expect(stats.MissCount).To(Equal(int64(0)))
		})
	})

	Describe("Cache Backend Types", func() {
		It("should have correct backend type constants", func() {
			Expect(cache.InMemoryCacheType).To(Equal(cache.CacheBackendType("memory")))
			Expect(cache.MilvusCacheType).To(Equal(cache.CacheBackendType("milvus")))
		})
	})

	Describe("Cache Configuration Types", func() {
		It("should support all required configuration fields", func() {
			config := cache.CacheConfig{
				BackendType:         cache.MilvusCacheType,
				Enabled:             true,
				SimilarityThreshold: 0.9,
				MaxEntries:          2000,
				TTLSeconds:          7200,
				BackendConfigPath:   "config/cache/milvus.yaml",
			}

			// Verify all fields are accessible
			Expect(string(config.BackendType)).To(Equal("milvus"))
			Expect(config.Enabled).To(BeTrue())
			Expect(config.SimilarityThreshold).To(Equal(float32(0.9)))
			Expect(config.MaxEntries).To(Equal(2000))
			Expect(config.TTLSeconds).To(Equal(7200))
			Expect(config.BackendConfigPath).To(Equal("config/cache/milvus.yaml"))
		})
	})

	Describe("Cache Stats", func() {
		It("should calculate hit ratio correctly", func() {
			stats := cache.CacheStats{
				TotalEntries: 100,
				HitCount:     75,
				MissCount:    25,
				HitRatio:     0.75,
			}

			Expect(stats.HitRatio).To(Equal(0.75))
			Expect(stats.HitCount + stats.MissCount).To(Equal(int64(100)))
		})

		It("should handle zero values correctly", func() {
			stats := cache.CacheStats{
				TotalEntries: 0,
				HitCount:     0,
				MissCount:    0,
				HitRatio:     0.0,
			}

			Expect(stats.HitRatio).To(Equal(0.0))
			Expect(stats.TotalEntries).To(Equal(0))
		})
	})
})
