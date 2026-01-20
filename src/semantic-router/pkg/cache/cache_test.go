//go:build !windows && cgo

package cache

import (
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"gopkg.in/yaml.v3"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
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
					config := CacheConfig{
						BackendType:         InMemoryCacheType,
						Enabled:             true,
						SimilarityThreshold: 0.8,
						MaxEntries:          1000,
						TTLSeconds:          3600,
						EmbeddingModel:      "bert",
					}

					backend, err := NewCacheBackend(config)
					Expect(err).NotTo(HaveOccurred())
					Expect(backend).NotTo(BeNil())
					Expect(backend.IsEnabled()).To(BeTrue())
				})

				It("should create disabled cache when enabled is false", func() {
					config := CacheConfig{
						BackendType:         InMemoryCacheType,
						Enabled:             false,
						SimilarityThreshold: 0.8,
						MaxEntries:          1000,
						TTLSeconds:          3600,
						EmbeddingModel:      "bert",
					}

					backend, err := NewCacheBackend(config)
					Expect(err).NotTo(HaveOccurred())
					Expect(backend).NotTo(BeNil())
					Expect(backend.IsEnabled()).To(BeFalse())
				})

				It("should default to memory backend when backend_type is empty", func() {
					config := CacheConfig{
						BackendType:         "", // Empty should default to memory
						Enabled:             true,
						SimilarityThreshold: 0.8,
						MaxEntries:          500,
						TTLSeconds:          1800,
						EmbeddingModel:      "bert",
					}

					backend, err := NewCacheBackend(config)
					Expect(err).NotTo(HaveOccurred())
					Expect(backend).NotTo(BeNil())
					Expect(backend.IsEnabled()).To(BeTrue())
				})
			})

			Context("(Deprecated) with file base Milvus backend", func() {
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

				It("should create Milvus cache backend successfully with valid config (Deprecated)", func() {
					config := CacheConfig{
						BackendType:         MilvusCacheType,
						Enabled:             true,
						SimilarityThreshold: 0.85,
						TTLSeconds:          7200,
						BackendConfigPath:   milvusConfigPath,
						EmbeddingModel:      "bert",
					}

					backend, err := NewCacheBackend(config)

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

				It("should handle disabled Milvus cache (Deprecated)", func() {
					config := CacheConfig{
						BackendType:         MilvusCacheType,
						Enabled:             false,
						SimilarityThreshold: 0.8,
						TTLSeconds:          3600,
						BackendConfigPath:   milvusConfigPath,
						EmbeddingModel:      "bert",
					}

					backend, err := NewCacheBackend(config)
					Expect(err).NotTo(HaveOccurred())
					Expect(backend).NotTo(BeNil())
					Expect(backend.IsEnabled()).To(BeFalse())
				})
			})

			Context("with inline Milvus configuration", func() {
				var milvusConfig *config.MilvusConfig
				BeforeEach(func() {
					// Skip Milvus tests if environment variable is set
					if os.Getenv("SKIP_MILVUS_TESTS") == "true" {
						Skip("Milvus tests skipped due to SKIP_MILVUS_TESTS=true")
					}

					yamlConfig := `
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
					err := yaml.Unmarshal([]byte(yamlConfig), &milvusConfig)
					Expect(err).NotTo(HaveOccurred())
				})

				It("should create Milvus cache backend successfully with valid config", func() {
					config := CacheConfig{
						BackendType:         MilvusCacheType,
						Enabled:             true,
						SimilarityThreshold: 0.85,
						TTLSeconds:          7200,
						Milvus:              milvusConfig,
						EmbeddingModel:      "bert",
					}

					backend, err := NewCacheBackend(config)

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
			})

			Context("(Deprecated) with Redis backend", func() {
				var redisConfigPath string

				BeforeEach(func() {
					if os.Getenv("SKIP_REDIS_TESTS") == "true" {
						Skip("Redis tests skipped due to SKIP_REDIS_TESTS=true")
					}

					redisConfigPath = filepath.Join(tempDir, "redis.yaml")
					redisConfig := `
connection:
  host: "localhost"
  port: 6379
  database: 0
  timeout: 30

index:
  name: "test_semantic_cache"
  prefix: "doc:"
  vector_field:
    name: "embedding"
    dimension: 384
    metric_type: "COSINE"
  index_type: "HNSW"
  params:
    M: 16
    efConstruction: 64

search:
  topk: 1

logging:
  enable_query_log: false
  enable_metrics: false

development:
  drop_index_on_startup: true
  auto_create_index: true
  verbose_errors: true
`
					err := os.WriteFile(redisConfigPath, []byte(redisConfig), 0o644)
					Expect(err).NotTo(HaveOccurred())
				})

				It("should create Redis cache backend successfully with valid config (Deprecated)", func() {
					config := CacheConfig{
						BackendType:         RedisCacheType,
						Enabled:             true,
						SimilarityThreshold: 0.8,
						TTLSeconds:          3600,
						BackendConfigPath:   redisConfigPath,
						EmbeddingModel:      "bert",
					}

					backend, err := NewCacheBackend(config)

					if err != nil {
						if strings.Contains(err.Error(), "failed to connect to Redis") ||
							strings.Contains(err.Error(), "connection refused") ||
							strings.Contains(err.Error(), "failed to initialize index") {
							Skip("Redis server not available: " + err.Error())
						}
						Expect(err).NotTo(HaveOccurred())
					} else {
						Expect(backend).NotTo(BeNil())
						Expect(backend.IsEnabled()).To(BeTrue())
						Expect(backend.Close()).To(Succeed())
					}
				})

				It("should handle disabled Redis cache (Deprecated)", func() {
					config := CacheConfig{
						BackendType:         RedisCacheType,
						Enabled:             false,
						SimilarityThreshold: 0.8,
						TTLSeconds:          3600,
						BackendConfigPath:   redisConfigPath,
						EmbeddingModel:      "bert",
					}

					backend, err := NewCacheBackend(config)
					Expect(err).NotTo(HaveOccurred())
					Expect(backend).NotTo(BeNil())
					Expect(backend.IsEnabled()).To(BeFalse())
					Expect(backend.Close()).To(Succeed())
				})
			})

			Context("with inline Redis configuration", func() {
				var redisConfig *config.RedisConfig
				// Skip Milvus tests if environment variable is set
				if os.Getenv("SKIP_REDIS_TESTS") == "true" {
					Skip("Redis tests skipped due to SKIP_REDIS_TESTS=true")
				}

				BeforeEach(func() {
					yamlConfig := `
connection:
    host: "localhost"
    port: 6379
    database: 0
    password: ""
    timeout: 30
    tls:
      enabled: false
      cert_file: ""
      key_file: ""
      ca_file: ""
index:
  name: "semantic_cache_idx"
  prefix: "doc:"
  vector_field:
    name: "embedding"
    dimension: 384
    metric_type: "COSINE"
  index_type: "HNSW"
  params:
    M: 16
    efConstruction: 64
search:
  topk: 1
logging:
  level: "info"
  enable_query_log: false
  enable_metrics: true
development:
  drop_index_on_startup: true
  auto_create_index: true
  verbose_errors: true
`
					err := yaml.Unmarshal([]byte(yamlConfig), &redisConfig)
					Expect(err).NotTo(HaveOccurred())
				})

				It("should create Redis cache backend successfully with valid config", func() {
					config := CacheConfig{
						BackendType:         RedisCacheType,
						Enabled:             true,
						SimilarityThreshold: 0.8,
						TTLSeconds:          3600,
						Redis:               redisConfig,
						EmbeddingModel:      "bert",
					}

					backend, err := NewCacheBackend(config)

					if err != nil {
						if strings.Contains(err.Error(), "failed to connect to Redis") ||
							strings.Contains(err.Error(), "connection refused") ||
							strings.Contains(err.Error(), "failed to initialize index") {
							Skip("Redis server not available: " + err.Error())
						}
						Expect(err).NotTo(HaveOccurred())
					} else {
						Expect(backend).NotTo(BeNil())
						Expect(backend.IsEnabled()).To(BeTrue())
						Expect(backend.Close()).To(Succeed())
					}
				})
			})

			Context("Milvus connection timeouts", func() {
				It("should respect connection timeout when endpoint is unreachable", func() {
					unreachableConfigPath := filepath.Join(tempDir, "milvus-unreachable.yaml")
					unreachableHost := "10.255.255.1" // unroutable address to simulate a hanging dial
					unreachableConfig := fmt.Sprintf(`
connection:
  host: "%s"
  port: 19530
  database: "test_cache"
  timeout: 1
`, unreachableHost)

					err := os.WriteFile(unreachableConfigPath, []byte(unreachableConfig), 0o644)
					Expect(err).NotTo(HaveOccurred())

					done := make(chan struct{})
					var cacheErr error

					go func() {
						defer GinkgoRecover()
						_, cacheErr = NewMilvusCache(MilvusCacheOptions{
							Enabled:             true,
							SimilarityThreshold: 0.85,
							TTLSeconds:          60,
							ConfigPath:          unreachableConfigPath,
						})
						close(done)
					}()

					Eventually(done, 2*time.Second, 100*time.Millisecond).Should(BeClosed())
					Expect(cacheErr).To(HaveOccurred())
					Expect(cacheErr.Error()).To(Or(
						ContainSubstring("context deadline exceeded"),
						ContainSubstring("timeout"),
					))
				})
			})

			Context("Milvus YAML parsing", func() {
				It("should parse snake_case configuration fields without default fallbacks", func() {
					configPath := filepath.Join(tempDir, "milvus-snake.yaml")
					configYAML := `
connection:
  host: "localhost"
  port: 19530
collection:
  name: "yaml_snake_case"
  vector_field:
    name: "custom_embedding"
    dimension: 512
    metric_type: "L2"
  index:
    type: "IVF_FLAT"
    params:
      M: 24
      efConstruction: 128
search:
  params:
    ef: 42
  topk: 25
development:
  auto_create_collection: true
  drop_collection_on_startup: true
`
					err := os.WriteFile(configPath, []byte(configYAML), 0o644)
					Expect(err).NotTo(HaveOccurred())

					config, err := loadMilvusConfig(configPath)
					Expect(err).NotTo(HaveOccurred())
					Expect(config.Collection.Name).To(Equal("yaml_snake_case"))
					Expect(config.Collection.VectorField.Name).To(Equal("custom_embedding"))
					Expect(config.Collection.VectorField.Dimension).To(Equal(512))
					Expect(config.Collection.VectorField.MetricType).To(Equal("L2"))
					Expect(config.Collection.Index.Type).To(Equal("IVF_FLAT"))
					Expect(config.Collection.Index.Params.M).To(Equal(24))
					Expect(config.Collection.Index.Params.EfConstruction).To(Equal(128))
					Expect(config.Search.Params.Ef).To(Equal(42))
					Expect(config.Search.TopK).To(Equal(25))
					Expect(config.Development.AutoCreateCollection).To(BeTrue())
					Expect(config.Development.DropCollectionOnStartup).To(BeTrue())
				})
			})

			Context("with unsupported backend type", func() {
				It("should return error for unsupported backend type", func() {
					config := CacheConfig{
						BackendType:         "unsupported_type", // Unsupported
						Enabled:             true,
						SimilarityThreshold: 0.8,
						TTLSeconds:          3600,
						EmbeddingModel:      "bert",
					}

					backend, err := NewCacheBackend(config)
					Expect(err).To(HaveOccurred())
					Expect(err.Error()).To(ContainSubstring("unsupported cache backend type"))
					Expect(backend).To(BeNil())
				})
			})

			Context("with invalid config but valid backend type", func() {
				It("should return error due to validation when config has invalid values", func() {
					config := CacheConfig{
						BackendType:         InMemoryCacheType, // valid backend type
						Enabled:             true,
						SimilarityThreshold: -0.8, // invalid
						MaxEntries:          10,
						TTLSeconds:          -1, // invalid
						EmbeddingModel:      "bert",
					}

					backend, err := NewCacheBackend(config)

					Expect(err).To(HaveOccurred())
					Expect(err.Error()).To(ContainSubstring("invalid cache config")) // ensure from config validation
					Expect(backend).To(BeNil())
				})
			})
		})

		Describe("ValidateCacheConfig", func() {
			It("should validate enabled memory backend configuration", func() {
				config := CacheConfig{
					BackendType:         InMemoryCacheType,
					Enabled:             true,
					SimilarityThreshold: 0.8,
					MaxEntries:          1000,
					TTLSeconds:          3600,
					EmbeddingModel:      "bert",
					EvictionPolicy:      "lru",
				}

				err := ValidateCacheConfig(config)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should validate disabled cache configuration", func() {
				config := CacheConfig{
					BackendType:         InMemoryCacheType,
					Enabled:             false,
					SimilarityThreshold: 2.0, // Invalid, but should be ignored for disabled cache
					MaxEntries:          -1,  // Invalid, but should be ignored for disabled cache
				}

				err := ValidateCacheConfig(config)
				Expect(err).NotTo(HaveOccurred()) // Disabled cache should skip validation
			})

			It("should return error for invalid similarity threshold", func() {
				config := CacheConfig{
					BackendType:         InMemoryCacheType,
					Enabled:             true,
					SimilarityThreshold: 1.5, // Invalid: > 1.0
					MaxEntries:          1000,
					TTLSeconds:          3600,
					EmbeddingModel:      "bert",
				}

				err := ValidateCacheConfig(config)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("similarity_threshold must be between 0.0 and 1.0"))
			})

			It("should return error for negative similarity threshold", func() {
				config := CacheConfig{
					BackendType:         InMemoryCacheType,
					Enabled:             true,
					SimilarityThreshold: -0.1, // Invalid: < 0.0
					MaxEntries:          1000,
					TTLSeconds:          3600,
					EmbeddingModel:      "bert",
				}

				err := ValidateCacheConfig(config)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("similarity_threshold must be between 0.0 and 1.0"))
			})

			It("should return error for negative TTL", func() {
				config := CacheConfig{
					BackendType:         InMemoryCacheType,
					Enabled:             true,
					SimilarityThreshold: 0.8,
					MaxEntries:          1000,
					TTLSeconds:          -1, // Invalid: negative TTL
					EmbeddingModel:      "bert",
				}

				err := ValidateCacheConfig(config)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("ttl_seconds cannot be negative"))
			})

			It("should return error for negative max entries in memory backend", func() {
				config := CacheConfig{
					BackendType:         InMemoryCacheType,
					Enabled:             true,
					SimilarityThreshold: 0.8,
					MaxEntries:          -1, // Invalid: negative max entries
					TTLSeconds:          3600,
					EmbeddingModel:      "bert",
				}

				err := ValidateCacheConfig(config)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("max_entries cannot be negative"))
			})

			It("should return error for unsupported eviction_policy value in memory backend", func() {
				config := CacheConfig{
					BackendType:         InMemoryCacheType,
					Enabled:             true,
					SimilarityThreshold: 0.8,
					MaxEntries:          1000,
					TTLSeconds:          3600,
					EmbeddingModel:      "bert",
					EvictionPolicy:      "random", // unsupported
				}

				err := ValidateCacheConfig(config)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("unsupported eviction_policy"))
			})

			It("should return error for Milvus backend without config path", func() {
				config := CacheConfig{
					BackendType:         MilvusCacheType,
					Enabled:             true,
					SimilarityThreshold: 0.8,
					TTLSeconds:          3600,
					EmbeddingModel:      "bert",
					// BackendConfigPath is missing
				}

				err := ValidateCacheConfig(config)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("backend_config_path is required for Milvus"))
			})

			It("should return error when Milvus backend_config_path file doesn't exist", func() {
				config := CacheConfig{
					BackendType:         MilvusCacheType,
					Enabled:             true,
					SimilarityThreshold: 0.8,
					TTLSeconds:          3600,
					EmbeddingModel:      "bert",
					BackendConfigPath:   "/nonexistent/milvus.yaml",
				}

				err := ValidateCacheConfig(config)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("config file not found"))
			})

			It("should validate edge case values", func() {
				config := CacheConfig{
					BackendType:         InMemoryCacheType,
					Enabled:             true,
					SimilarityThreshold: 0.0, // Valid: minimum threshold
					MaxEntries:          0,   // Valid: unlimited entries
					TTLSeconds:          0,   // Valid: no expiration
				}

				err := ValidateCacheConfig(config)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should validate maximum threshold value", func() {
				config := CacheConfig{
					BackendType:         InMemoryCacheType,
					Enabled:             true,
					SimilarityThreshold: 1.0, // Valid: maximum threshold
					MaxEntries:          10000,
					TTLSeconds:          86400,
					EmbeddingModel:      "bert",
				}

				err := ValidateCacheConfig(config)
				Expect(err).NotTo(HaveOccurred())
			})
		})

		Describe("GetDefaultCacheConfig", func() {
			It("should return valid default configuration", func() {
				config := GetDefaultCacheConfig()

				Expect(config.BackendType).To(Equal(InMemoryCacheType))
				Expect(config.Enabled).To(BeTrue())
				Expect(config.SimilarityThreshold).To(Equal(float32(0.8)))
				Expect(config.MaxEntries).To(Equal(1000))
				Expect(config.TTLSeconds).To(Equal(3600))
				Expect(config.BackendConfigPath).To(BeEmpty())

				// Default config should pass validation
				err := ValidateCacheConfig(config)
				Expect(err).NotTo(HaveOccurred())
			})
		})

		Describe("GetAvailableCacheBackends", func() {
			It("should return information about available backends", func() {
				backends := GetAvailableCacheBackends()

				Expect(backends).To(HaveLen(3)) // Memory, Milvus, and Redis

				// Check memory backend info
				memoryBackend := backends[0]
				Expect(memoryBackend.Type).To(Equal(InMemoryCacheType))
				Expect(memoryBackend.Name).To(Equal("In-Memory Cache"))
				Expect(memoryBackend.Description).To(ContainSubstring("in-memory semantic cache"))
				Expect(memoryBackend.Features).To(ContainElement("Fast access"))
				Expect(memoryBackend.Features).To(ContainElement("No external dependencies"))

				// Check Milvus backend info
				milvusBackend := backends[1]
				Expect(milvusBackend.Type).To(Equal(MilvusCacheType))
				Expect(milvusBackend.Name).To(Equal("Milvus Vector Database"))
				Expect(milvusBackend.Description).To(ContainSubstring("Milvus vector database"))
				Expect(milvusBackend.Features).To(ContainElement("Highly scalable"))
				Expect(milvusBackend.Features).To(ContainElement("Persistent storage"))

				// Check Redis backend info
				redisBackend := backends[2]
				Expect(redisBackend.Type).To(Equal(RedisCacheType))
				Expect(redisBackend.Name).To(Equal("Redis Vector Database"))
				Expect(redisBackend.Description).To(ContainSubstring("Redis with vector search"))
				Expect(redisBackend.Features).To(ContainElement("Fast in-memory performance"))
				Expect(redisBackend.Features).To(ContainElement("TTL support"))
			})
		})
	})

	Describe("InMemoryCache", func() {
		var inMemoryCache CacheBackend

		BeforeEach(func() {
			options := InMemoryCacheOptions{
				Enabled:             true,
				SimilarityThreshold: 0.8,
				MaxEntries:          100,
				TTLSeconds:          300,
				EmbeddingModel:      "bert",
			}
			inMemoryCache = NewInMemoryCache(options)
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
			disabledOptions := InMemoryCacheOptions{
				Enabled:             false,
				SimilarityThreshold: 0.8,
				MaxEntries:          100,
				TTLSeconds:          300,
				EmbeddingModel:      "bert",
			}
			disabledCache := NewInMemoryCache(disabledOptions)
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
			err := inMemoryCache.AddEntry("test-request-id", "test-model", "test query", []byte("request"), []byte("response"), -1)
			Expect(err).NotTo(HaveOccurred())

			stats := inMemoryCache.GetStats()
			Expect(stats.TotalEntries).To(Equal(1))
		})

		It("should handle FindSimilar operation with embeddings", func() {
			// First add an entry
			err := inMemoryCache.AddEntry("test-request-id", "test-model", "test query", []byte("request"), []byte("response"), -1)
			Expect(err).NotTo(HaveOccurred())

			// Search for similar query
			response, found, err := inMemoryCache.FindSimilar("test-model", "test query")
			Expect(err).NotTo(HaveOccurred())
			Expect(found).To(BeTrue()) // Should find exact match
			Expect(response).To(Equal([]byte("response")))

			// Search for different model - should match due to cross-model cache sharing
			// (model filtering removed to improve cache hit rates)
			response, found, err = inMemoryCache.FindSimilar("different-model", "test query")
			Expect(err).NotTo(HaveOccurred())
			Expect(found).To(BeTrue())
			Expect(response).To(Equal([]byte("response")))
		})

		It("should handle AddPendingRequest and UpdateWithResponse", func() {
			err := inMemoryCache.AddPendingRequest("test-request-id", "test-model", "test query", []byte("request"), -1)
			Expect(err).NotTo(HaveOccurred())

			// Update with response
			err = inMemoryCache.UpdateWithResponse("test-request-id", []byte("response"), -1)
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
			inMemoryCache = NewInMemoryCache(InMemoryCacheOptions{
				Enabled:             true,
				SimilarityThreshold: 0.8,
				MaxEntries:          100,
				TTLSeconds:          1,
				EmbeddingModel:      "bert",
			})

			err := inMemoryCache.AddPendingRequest("expired-request-id", "test-model", "stale query", []byte("request"), -1)
			Expect(err).NotTo(HaveOccurred())
			Expect(testutil.ToFloat64(metrics.CacheEntriesTotal.WithLabelValues("memory"))).To(Equal(float64(1)))

			// Wait for TTL to expire before triggering the update path
			time.Sleep(2 * time.Second)

			err = inMemoryCache.UpdateWithResponse("expired-request-id", []byte("response"), -1)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("no pending request"))

			Expect(testutil.ToFloat64(metrics.CacheEntriesTotal.WithLabelValues("memory"))).To(BeZero())
		})

		It("should respect similarity threshold", func() {
			// Add entry with a very high similarity threshold
			highThresholdOptions := InMemoryCacheOptions{
				Enabled:             true,
				SimilarityThreshold: 0.99, // Very high threshold
				MaxEntries:          100,
				TTLSeconds:          300,
				EmbeddingModel:      "bert",
			}
			highThresholdCache := NewInMemoryCache(highThresholdOptions)
			defer highThresholdCache.Close()

			err := highThresholdCache.AddEntry("test-request-id", "test-model", "machine learning", []byte("request"), []byte("ml response"), -1)
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
			err := inMemoryCache.AddEntry("test-request-id", "test-model", "What is machine learning?", []byte("request"), []byte("ML is a subset of AI"), -1)
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
			ttlCache := NewInMemoryCache(InMemoryCacheOptions{
				Enabled:             true,
				SimilarityThreshold: 0.1,
				MaxEntries:          10,
				TTLSeconds:          1,
				EmbeddingModel:      "bert",
			})
			defer ttlCache.Close()

			err := ttlCache.AddEntry("ttl-request-id", "ttl-model", "time-sensitive query", []byte("request"), []byte("response"), -1)
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
			err := inMemoryCache.UpdateWithResponse("non-existent-query", []byte("response"), -1)
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
			disabledOptions := InMemoryCacheOptions{
				Enabled:             false,
				SimilarityThreshold: 0.8,
				MaxEntries:          100,
				TTLSeconds:          300,
				EmbeddingModel:      "bert",
			}
			disabledCache := NewInMemoryCache(disabledOptions)
			defer disabledCache.Close()

			// Disabled cache operations should not error but should be no-ops
			// They should NOT try to generate embeddings
			err := disabledCache.AddPendingRequest("test-request-id", "test-model", "test query", []byte("request"), -1)
			Expect(err).NotTo(HaveOccurred())

			err = disabledCache.UpdateWithResponse("test-request-id", []byte("response"), -1)
			Expect(err).NotTo(HaveOccurred())

			err = disabledCache.AddEntry("test-request-id", "test-model", "test query", []byte("request"), []byte("response"), -1)
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

		It("should keep existing HNSW nodes searchable after eviction", func() {
			cacheWithHNSW := NewInMemoryCache(InMemoryCacheOptions{
				Enabled:             true,
				SimilarityThreshold: 0.1,
				MaxEntries:          2,
				TTLSeconds:          60, // Set TTL long enough to avoid expiration during test
				EvictionPolicy:      FIFOEvictionPolicyType,
				UseHNSW:             true,
				HNSWM:               4,
				HNSWEfConstruction:  8,
				HNSWEfSearch:        8,
				EmbeddingModel:      "bert",
			})
			defer cacheWithHNSW.Close()

			err := cacheWithHNSW.AddEntry("req-1", "test-model", "first query text", []byte("request-1"), []byte("response-1"), -1)
			Expect(err).NotTo(HaveOccurred())

			err = cacheWithHNSW.AddEntry("req-2", "test-model", "second query text", []byte("request-2"), []byte("response-2"), -1)
			Expect(err).NotTo(HaveOccurred())

			// Sanity check: the second entry should be retrievable before any eviction occurs.
			resp, found, err := cacheWithHNSW.FindSimilar("test-model", "second query text")
			Expect(err).NotTo(HaveOccurred())
			Expect(found).To(BeTrue())
			Expect(resp).To(Equal([]byte("response-2")))

			// Adding a third entry triggers eviction (max entries = 2).
			err = cacheWithHNSW.AddEntry("req-3", "test-model", "third query text", []byte("request-3"), []byte("response-3"), -1)
			Expect(err).NotTo(HaveOccurred())

			// Entry 2 should still be searchable even after eviction reshuffles the slice.
			resp, found, err = cacheWithHNSW.FindSimilar("test-model", "second query text")
			Expect(err).NotTo(HaveOccurred())
			Expect(found).To(BeTrue())
			Expect(resp).To(Equal([]byte("response-2")))
		})
	})

	Describe("Cache Backend Types", func() {
		It("should have correct backend type constants", func() {
			Expect(InMemoryCacheType).To(Equal(CacheBackendType("memory")))
			Expect(MilvusCacheType).To(Equal(CacheBackendType("milvus")))
		})
	})

	Describe("Cache Configuration Types", func() {
		It("should support all required configuration fields", func() {
			config := CacheConfig{
				BackendType:         MilvusCacheType,
				Enabled:             true,
				SimilarityThreshold: 0.9,
				MaxEntries:          2000,
				TTLSeconds:          7200,
				EmbeddingModel:      "bert",
				BackendConfigPath:   "config/semantic-cache/milvus.yaml",
			}

			// Verify all fields are accessible
			Expect(string(config.BackendType)).To(Equal("milvus"))
			Expect(config.Enabled).To(BeTrue())
			Expect(config.SimilarityThreshold).To(Equal(float32(0.9)))
			Expect(config.MaxEntries).To(Equal(2000))
			Expect(config.TTLSeconds).To(Equal(7200))
			Expect(config.BackendConfigPath).To(Equal("config/semantic-cache/milvus.yaml"))
		})
	})

	Describe("Cache Stats", func() {
		It("should calculate hit ratio correctly", func() {
			stats := CacheStats{
				TotalEntries: 100,
				HitCount:     75,
				MissCount:    25,
				HitRatio:     0.75,
			}

			Expect(stats.HitRatio).To(Equal(0.75))
			Expect(stats.HitCount + stats.MissCount).To(Equal(int64(100)))
		})

		It("should handle zero values correctly", func() {
			stats := CacheStats{
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

// ContentLength defines different query content sizes
type ContentLength int

const (
	ShortContent  ContentLength = 20  // ~20 words
	MediumContent ContentLength = 50  // ~50 words
	LongContent   ContentLength = 100 // ~100 words
)

func (c ContentLength) String() string {
	switch c {
	case ShortContent:
		return "short"
	case MediumContent:
		return "medium"
	case LongContent:
		return "long"
	default:
		return "unknown"
	}
}

// GenerateQuery generates a query with maximum semantic diversity using hash-based randomization
func generateQuery(length ContentLength, index int) string {
	// Hash the index to get pseudo-random values (deterministic but well-distributed)
	hash := uint64(index) // #nosec G115 -- index is always positive and bounded
	hash *= 2654435761    // Knuth's multiplicative hash

	// Expanded templates for maximum diversity
	templates := []string{
		// Technical how-to questions
		"How to implement %s using %s and %s for %s applications in production environments",
		"What are the best practices for %s when building %s systems with %s constraints",
		"Can you explain the architecture of %s systems that integrate %s and %s components",
		"How do I configure %s to work with %s while ensuring %s compatibility",
		"What is the recommended approach for %s development using %s and %s technologies",

		// Comparison questions
		"Explain the difference between %s and %s in the context of %s development",
		"Compare and contrast %s approaches versus %s methods for %s use cases",
		"What is the performance impact of %s versus %s for %s workloads",
		"Which is better for %s: %s or %s, considering %s requirements",
		"When should I use %s instead of %s for %s scenarios",

		// Debugging/troubleshooting
		"Can you help me debug %s issues related to %s when using %s framework",
		"Why is my %s failing when I integrate %s with %s system",
		"How to troubleshoot %s errors in %s when deploying to %s environment",
		"What causes %s problems in %s architecture with %s configuration",

		// Optimization questions
		"How do I optimize %s for %s while maintaining %s requirements",
		"What are the performance bottlenecks in %s when using %s with %s",
		"How can I improve %s throughput in %s systems running %s",
		"What are common pitfalls when optimizing %s with %s in %s environments",

		// Design/architecture questions
		"How should I design %s to handle %s and support %s functionality",
		"What are the scalability considerations for %s when implementing %s with %s",
		"How to architect %s systems that require %s and %s capabilities",
		"What design patterns work best for %s in %s architectures with %s",
	}

	// Massively expanded topics for semantic diversity
	topics := []string{
		// ML/AI
		"machine learning", "deep learning", "neural networks", "reinforcement learning",
		"computer vision", "NLP", "transformers", "embeddings", "fine-tuning",

		// Infrastructure
		"microservices", "distributed systems", "message queues", "event streaming",
		"container orchestration", "service mesh", "API gateway", "load balancing",
		"database sharding", "data replication", "consensus algorithms", "circuit breakers",

		// Data
		"data pipelines", "ETL", "data warehousing", "real-time analytics",
		"stream processing", "batch processing", "data lakes", "data modeling",

		// Security
		"authentication", "authorization", "encryption", "TLS", "OAuth",
		"API security", "zero trust", "secrets management", "key rotation",

		// Observability
		"monitoring", "logging", "tracing", "metrics", "alerting",
		"observability", "profiling", "debugging", "APM",

		// Performance
		"caching strategies", "rate limiting", "connection pooling", "query optimization",
		"memory management", "garbage collection", "CPU profiling", "I/O optimization",

		// Reliability
		"high availability", "fault tolerance", "disaster recovery", "backups",
		"failover", "redundancy", "chaos engineering", "SLA management",

		// Cloud/DevOps
		"CI/CD", "GitOps", "infrastructure as code", "configuration management",
		"auto-scaling", "serverless", "edge computing", "multi-cloud",

		// Databases
		"SQL databases", "NoSQL", "graph databases", "time series databases",
		"vector databases", "in-memory databases", "database indexing", "query planning",
	}

	// Additional random modifiers for even more diversity
	modifiers := []string{
		"large-scale", "enterprise", "cloud-native", "production-grade",
		"real-time", "distributed", "fault-tolerant", "high-performance",
		"mission-critical", "scalable", "secure", "compliant",
	}

	// Use hash to pseudo-randomly select (but deterministic for same index)
	templateIdx := int(hash % uint64(len(templates))) // #nosec G115 -- modulo operation is bounded by array length
	hash = hash * 16807 % 2147483647                  // LCG for next random

	topic1Idx := int(hash % uint64(len(topics))) // #nosec G115 -- modulo operation is bounded by array length
	hash = hash * 16807 % 2147483647

	topic2Idx := int(hash % uint64(len(topics))) // #nosec G115 -- modulo operation is bounded by array length
	hash = hash * 16807 % 2147483647

	topic3Idx := int(hash % uint64(len(topics))) // #nosec G115 -- modulo operation is bounded by array length
	hash = hash * 16807 % 2147483647

	// Build query with selected template and topics
	query := fmt.Sprintf(templates[templateIdx],
		topics[topic1Idx],
		topics[topic2Idx],
		topics[topic3Idx],
		modifiers[int(hash%uint64(len(modifiers)))]) // #nosec G115 -- modulo operation is bounded by array length

	// Add unique identifier to guarantee uniqueness
	query += fmt.Sprintf(" [Request ID: REQ-%d]", index)

	// Add extra context for longer queries
	if length > MediumContent {
		hash = hash * 16807 % 2147483647
		extraTopicIdx := int(hash % uint64(len(topics))) // #nosec G115 -- modulo operation is bounded by array length
		query += fmt.Sprintf(" Also considering %s integration and %s compatibility requirements.",
			topics[extraTopicIdx],
			modifiers[int(hash%uint64(len(modifiers)))]) // #nosec G115 -- modulo operation is bounded by array length
	}

	return query
}

// BenchmarkComprehensive runs comprehensive benchmarks across multiple dimensions
func BenchmarkComprehensive(b *testing.B) {
	// Initialize BERT model
	useCPU := os.Getenv("USE_CPU") != "false" // Default to CPU
	modelName := "sentence-transformers/all-MiniLM-L6-v2"
	if err := candle_binding.InitModel(modelName, useCPU); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	// Determine hardware type
	hardware := "cpu"
	if !useCPU {
		hardware = "gpu"
	}

	// Test configurations
	cacheSizes := []int{100, 500, 1000, 5000}
	contentLengths := []ContentLength{ShortContent, MediumContent, LongContent}
	hnswConfigs := []struct {
		name string
		m    int
		ef   int
	}{
		{"default", 16, 200},
		{"fast", 8, 100},
		{"accurate", 32, 400},
	}

	// Open CSV file for results
	csvFile, err := os.OpenFile(
		"../../benchmark_results/benchmark_data.csv",
		os.O_APPEND|os.O_CREATE|os.O_WRONLY,
		0o644)
	if err != nil {
		b.Logf("Warning: Could not open CSV file: %v", err)
	} else {
		defer csvFile.Close()
	}

	// Run benchmarks
	for _, cacheSize := range cacheSizes {
		for _, contentLen := range contentLengths {
			// Generate test data
			testQueries := make([]string, cacheSize)
			for i := 0; i < cacheSize; i++ {
				testQueries[i] = generateQuery(contentLen, i)
			}

			// Benchmark Linear Search
			b.Run(fmt.Sprintf("%s/Linear/%s/%dEntries", hardware, contentLen.String(), cacheSize), func(b *testing.B) {
				cache := NewInMemoryCache(InMemoryCacheOptions{
					Enabled:             true,
					MaxEntries:          cacheSize * 2,
					SimilarityThreshold: 0.85,
					TTLSeconds:          0,
					UseHNSW:             false,
				})

				// Populate cache
				for i, query := range testQueries {
					reqID := fmt.Sprintf("req%d", i)
					_ = cache.AddEntry(reqID, "test-model", query, []byte(query), []byte("response"), -1)
				}

				searchQuery := generateQuery(contentLen, cacheSize/2)
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					_, _, _ = cache.FindSimilar("test-model", searchQuery)
				}

				b.StopTimer()

				// Write to CSV
				if csvFile != nil {
					nsPerOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N)

					line := fmt.Sprintf("%s,%s,%d,linear,0,0,%.0f,0,0,%d,1.0\n",
						hardware, contentLen.String(), cacheSize, nsPerOp, b.N)
					if _, err := csvFile.WriteString(line); err != nil {
						b.Logf("Warning: failed to write to CSV: %v", err)
					}
				}
			})

			// Benchmark HNSW with different configurations
			for _, hnswCfg := range hnswConfigs {
				b.Run(fmt.Sprintf("%s/HNSW_%s/%s/%dEntries", hardware, hnswCfg.name, contentLen.String(), cacheSize), func(b *testing.B) {
					cache := NewInMemoryCache(InMemoryCacheOptions{
						Enabled:             true,
						MaxEntries:          cacheSize * 2,
						SimilarityThreshold: 0.85,
						TTLSeconds:          0,
						UseHNSW:             true,
						HNSWM:               hnswCfg.m,
						HNSWEfConstruction:  hnswCfg.ef,
					})

					// Populate cache
					for i, query := range testQueries {
						reqID := fmt.Sprintf("req%d", i)
						_ = cache.AddEntry(reqID, "test-model", query, []byte(query), []byte("response"), -1)
					}

					searchQuery := generateQuery(contentLen, cacheSize/2)
					b.ResetTimer()

					for i := 0; i < b.N; i++ {
						_, _, _ = cache.FindSimilar("test-model", searchQuery)
					}

					b.StopTimer()

					// Write to CSV
					if csvFile != nil {
						nsPerOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N)

						line := fmt.Sprintf("%s,%s,%d,hnsw_%s,%d,%d,%.0f,0,0,%d,0.0\n",
							hardware, contentLen.String(), cacheSize, hnswCfg.name,
							hnswCfg.m, hnswCfg.ef, nsPerOp, b.N)
						if _, err := csvFile.WriteString(line); err != nil {
							b.Logf("Warning: failed to write to CSV: %v", err)
						}
					}
				})
			}
		}
	}
}

// BenchmarkIndexConstruction benchmarks HNSW index build time
func BenchmarkIndexConstruction(b *testing.B) {
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	cacheSizes := []int{100, 500, 1000, 5000}
	contentLengths := []ContentLength{ShortContent, MediumContent, LongContent}

	for _, cacheSize := range cacheSizes {
		for _, contentLen := range contentLengths {
			testQueries := make([]string, cacheSize)
			for i := 0; i < cacheSize; i++ {
				testQueries[i] = generateQuery(contentLen, i)
			}

			b.Run(fmt.Sprintf("BuildIndex/%s/%dEntries", contentLen.String(), cacheSize), func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					b.StopTimer()
					cache := NewInMemoryCache(InMemoryCacheOptions{
						Enabled:             true,
						MaxEntries:          cacheSize * 2,
						SimilarityThreshold: 0.85,
						TTLSeconds:          0,
						UseHNSW:             true,
						HNSWM:               16,
						HNSWEfConstruction:  200,
					})
					b.StartTimer()

					// Build index by adding entries
					for j, query := range testQueries {
						reqID := fmt.Sprintf("req%d", j)
						_ = cache.AddEntry(reqID, "test-model", query, []byte(query), []byte("response"), -1)
					}
				}
			})
		}
	}
}

func TestFIFOPolicy(t *testing.T) {
	policy := &FIFOPolicy{}

	// Test empty entries
	if victim := policy.SelectVictim([]CacheEntry{}); victim != -1 {
		t.Errorf("Expected -1 for empty entries, got %d", victim)
	}

	// Test with entries
	now := time.Now()
	entries := []CacheEntry{
		{Query: "query1", Timestamp: now.Add(-3 * time.Second)},
		{Query: "query2", Timestamp: now.Add(-1 * time.Second)},
		{Query: "query3", Timestamp: now.Add(-2 * time.Second)},
	}

	victim := policy.SelectVictim(entries)
	if victim != 0 {
		t.Errorf("Expected victim index 0 (oldest), got %d", victim)
	}
}

func TestLRUPolicy(t *testing.T) {
	policy := &LRUPolicy{}

	// Test empty entries
	if victim := policy.SelectVictim([]CacheEntry{}); victim != -1 {
		t.Errorf("Expected -1 for empty entries, got %d", victim)
	}

	// Test with entries
	now := time.Now()
	entries := []CacheEntry{
		{Query: "query1", LastAccessAt: now.Add(-3 * time.Second)},
		{Query: "query2", LastAccessAt: now.Add(-1 * time.Second)},
		{Query: "query3", LastAccessAt: now.Add(-2 * time.Second)},
	}

	victim := policy.SelectVictim(entries)
	if victim != 0 {
		t.Errorf("Expected victim index 0 (least recently used), got %d", victim)
	}
}

func TestLFUPolicy(t *testing.T) {
	policy := &LFUPolicy{}

	// Test empty entries
	if victim := policy.SelectVictim([]CacheEntry{}); victim != -1 {
		t.Errorf("Expected -1 for empty entries, got %d", victim)
	}

	// Test with entries
	now := time.Now()
	entries := []CacheEntry{
		{Query: "query1", HitCount: 5, LastAccessAt: now.Add(-2 * time.Second)},
		{Query: "query2", HitCount: 1, LastAccessAt: now.Add(-3 * time.Second)},
		{Query: "query3", HitCount: 3, LastAccessAt: now.Add(-1 * time.Second)},
	}

	victim := policy.SelectVictim(entries)
	if victim != 1 {
		t.Errorf("Expected victim index 1 (least frequently used), got %d", victim)
	}
}

func TestLFUPolicyTiebreaker(t *testing.T) {
	policy := &LFUPolicy{}

	// Test tiebreaker: same frequency, choose least recently used
	now := time.Now()
	entries := []CacheEntry{
		{Query: "query1", HitCount: 2, LastAccessAt: now.Add(-1 * time.Second)},
		{Query: "query2", HitCount: 2, LastAccessAt: now.Add(-3 * time.Second)},
		{Query: "query3", HitCount: 5, LastAccessAt: now.Add(-2 * time.Second)},
	}

	victim := policy.SelectVictim(entries)
	if victim != 1 {
		t.Errorf("Expected victim index 1 (LRU tiebreaker), got %d", victim)
	}
}

// TestFIFOPolicyOptimized tests the O(1) FIFO policy operations
func TestFIFOPolicyOptimized(t *testing.T) {
	policy := NewFIFOPolicy()
	entries := []CacheEntry{
		{RequestID: "req-0"},
		{RequestID: "req-1"},
		{RequestID: "req-2"},
	}

	// Test OnInsert and SelectVictim
	for i, e := range entries {
		policy.OnInsert(i, e.RequestID)
	}

	victim := policy.SelectVictim(entries)
	if victim != 0 {
		t.Errorf("Expected victim 0 (oldest), got %d", victim)
	}

	// Test Evict
	evicted := policy.Evict()
	if evicted != 0 {
		t.Errorf("Expected evicted index 0, got %d", evicted)
	}

	// Test UpdateIndex (simulating swap after eviction)
	policy.UpdateIndex("req-2", 2, 0)
	victim = policy.SelectVictim(entries)
	if victim != 1 {
		t.Errorf("Expected victim 1 after swap, got %d", victim)
	}

	// Test OnRemove
	policy.OnRemove(1, "req-1")
	victim = policy.SelectVictim(entries)
	if victim != 0 {
		t.Errorf("Expected victim 0 (req-2 moved), got %d", victim)
	}
}

// TestLRUPolicyOptimized tests the O(1) LRU policy operations
func TestLRUPolicyOptimized(t *testing.T) {
	policy := NewLRUPolicy()
	entries := []CacheEntry{
		{RequestID: "req-0"},
		{RequestID: "req-1"},
		{RequestID: "req-2"},
	}

	// Insert all entries
	for i, e := range entries {
		policy.OnInsert(i, e.RequestID)
	}

	// LRU order: req-2 (MRU) -> req-1 -> req-0 (LRU)
	victim := policy.SelectVictim(entries)
	if victim != 0 {
		t.Errorf("Expected victim 0 (LRU), got %d", victim)
	}

	// Access req-0 to make it MRU
	policy.OnAccess(0, "req-0")
	victim = policy.SelectVictim(entries)
	if victim != 1 {
		t.Errorf("Expected victim 1 after accessing req-0, got %d", victim)
	}

	// Test Evict
	evicted := policy.Evict()
	if evicted != 1 {
		t.Errorf("Expected evicted index 1, got %d", evicted)
	}

	// Test UpdateIndex
	policy.UpdateIndex("req-2", 2, 1)

	// Test OnRemove
	policy.OnRemove(1, "req-2")

	// Only req-0 should remain
	victim = policy.SelectVictim(entries)
	if victim != 0 {
		t.Errorf("Expected victim 0, got %d", victim)
	}
}

// TestLFUPolicyOptimized tests the O(1) LFU policy operations
func TestLFUPolicyOptimized(t *testing.T) {
	policy := NewLFUPolicy()
	entries := []CacheEntry{
		{RequestID: "req-0"},
		{RequestID: "req-1"},
		{RequestID: "req-2"},
	}

	// Insert all entries (all start with freq=1)
	for i, e := range entries {
		policy.OnInsert(i, e.RequestID)
	}

	// Access req-2 multiple times to increase frequency
	for i := 0; i < 5; i++ {
		policy.OnAccess(2, "req-2")
	}

	// req-0 and req-1 have freq=1, req-2 has freq=6
	victim := policy.SelectVictim(entries)
	if victim != 0 && victim != 1 {
		t.Errorf("Expected victim 0 or 1 (lowest freq), got %d", victim)
	}

	// Test Evict
	evicted := policy.Evict()
	if evicted != 0 && evicted != 1 {
		t.Errorf("Expected evicted 0 or 1, got %d", evicted)
	}

	// Test UpdateIndex
	policy.UpdateIndex("req-2", 2, 0)

	// Test OnRemove
	policy.OnRemove(1, "req-1")
}

// TestExpirationHeapOperations tests all ExpirationHeap operations
func TestExpirationHeapOperations(t *testing.T) {
	now := time.Now()
	heap := NewExpirationHeap()

	// Test Add
	heap.Add("req-0", 0, now.Add(1*time.Hour))
	heap.Add("req-1", 1, now.Add(30*time.Minute))
	heap.Add("req-2", 2, now.Add(2*time.Hour))

	// Test Size
	if heap.Size() != 3 {
		t.Errorf("Expected size 3, got %d", heap.Size())
	}

	// Test PeekNext (should be req-1, earliest expiration)
	reqID, idx, expiresAt, ok := heap.PeekNext()
	if !ok || reqID != "req-1" || idx != 1 {
		t.Errorf("Expected PeekNext to return req-1, got %s (idx=%d, ok=%v)", reqID, idx, ok)
	}
	_ = expiresAt

	// Test UpdateExpiration (move req-1 to later)
	heap.UpdateExpiration("req-1", now.Add(3*time.Hour))

	// Now req-0 should be earliest
	reqID, _, _, ok = heap.PeekNext()
	if !ok || reqID != "req-0" {
		t.Errorf("Expected PeekNext to return req-0 after update, got %s", reqID)
	}

	// Test UpdateIndex
	heap.UpdateIndex("req-0", 5)

	// Test Remove
	heap.Remove("req-0")
	if heap.Size() != 2 {
		t.Errorf("Expected size 2 after remove, got %d", heap.Size())
	}

	// Test PopExpired
	heap.Add("req-expired", 10, now.Add(-1*time.Hour)) // Already expired
	expired := heap.PopExpired(now)
	if len(expired) != 1 || expired[0] != "req-expired" {
		t.Errorf("Expected 1 expired entry, got %v", expired)
	}
}

// TestInMemoryCacheEviction tests cache eviction with O(1) policies
func TestInMemoryCacheEviction(t *testing.T) {
	cache := NewInMemoryCache(InMemoryCacheOptions{
		Enabled:             true,
		MaxEntries:          3,
		TTLSeconds:          3600,
		SimilarityThreshold: 0.9,
		EvictionPolicy:      LRUEvictionPolicyType,
	})

	// Add entries up to max
	for i := 0; i < 3; i++ {
		embedding := make([]float32, 384)
		embedding[i] = 1.0
		cache.mu.Lock()
		cache.entries = append(cache.entries, CacheEntry{
			RequestID: fmt.Sprintf("req-%d", i),
			Query:     fmt.Sprintf("query %d", i),
			Embedding: embedding,
		})
		idx := len(cache.entries) - 1
		cache.entryMap[fmt.Sprintf("req-%d", i)] = idx
		cache.registerEntryWithEvictionPolicy(idx, fmt.Sprintf("req-%d", i))
		cache.mu.Unlock()
	}

	// Verify we have 3 entries
	stats := cache.GetStats()
	if stats.TotalEntries != 3 {
		t.Errorf("Expected 3 entries, got %d", stats.TotalEntries)
	}

	// Add one more to trigger eviction
	cache.mu.Lock()
	cache.evictOne()
	cache.mu.Unlock()

	// Should have 2 entries now
	cache.mu.RLock()
	count := len(cache.entries)
	cache.mu.RUnlock()
	if count != 2 {
		t.Errorf("Expected 2 entries after eviction, got %d", count)
	}
}

// TestHybridCacheDisabled tests that disabled hybrid cache returns immediately
func TestHybridCacheDisabled(t *testing.T) {
	cache, err := NewHybridCache(HybridCacheOptions{
		Enabled: false,
	})
	if err != nil {
		t.Fatalf("Failed to create disabled cache: %v", err)
	}
	defer cache.Close()

	if cache.IsEnabled() {
		t.Error("Cache should be disabled")
	}

	// All operations should be no-ops
	err = cache.AddEntry("req1", "model1", "test query", []byte("request"), []byte("response"), -1)
	if err != nil {
		t.Errorf("AddEntry should not error on disabled cache: %v", err)
	}

	_, found, err := cache.FindSimilar("model1", "test query")
	if err != nil {
		t.Errorf("FindSimilar should not error on disabled cache: %v", err)
	}
	if found {
		t.Error("FindSimilar should not find anything on disabled cache")
	}
}

// TestHybridCacheBasicOperations tests basic cache operations
func TestHybridCacheBasicOperations(t *testing.T) {
	// Skip if Milvus tests are disabled
	if os.Getenv("SKIP_MILVUS_TESTS") == "true" {
		t.Skip("Skipping Milvus-dependent test (SKIP_MILVUS_TESTS=true)")
	}

	t.Log("Starting TestHybridCacheBasicOperations - this may take 30-60 seconds...")

	// Create a test Milvus config
	milvusConfig, cleanup, err := createTestMilvusConfig("test_hybrid_cache", 200, true)
	if err != nil {
		t.Fatalf("Failed to create test config: %v", err)
	}
	defer cleanup()

	cache, err := NewHybridCache(HybridCacheOptions{
		Enabled:             true,
		SimilarityThreshold: 0.8,
		TTLSeconds:          300,
		MaxMemoryEntries:    100,
		HNSWM:               16,
		HNSWEfConstruction:  200,
		MilvusConfigPath:    milvusConfig,
	})
	if err != nil {
		t.Fatalf("Failed to create hybrid cache: %v", err)
	}
	defer cache.Close()

	if !cache.IsEnabled() {
		t.Fatal("Cache should be enabled")
	}

	// Test AddEntry
	testQuery := "What is the meaning of life?"
	testResponse := []byte(`{"response": "42"}`)

	err = cache.AddEntry("req1", "gpt-4", testQuery, []byte("{}"), testResponse, -1)
	if err != nil {
		t.Fatalf("Failed to add entry: %v", err)
	}

	// Verify stats
	stats := cache.GetStats()
	if stats.TotalEntries != 1 {
		t.Errorf("Expected 1 entry, got %d", stats.TotalEntries)
	}

	// Test FindSimilar with exact same query (should hit)
	// Wait for Milvus to index the entry
	time.Sleep(2 * time.Second)

	response, found, err := cache.FindSimilar("gpt-4", testQuery)
	if err != nil {
		t.Fatalf("FindSimilar failed: %v", err)
	}
	if !found {
		t.Error("Expected to find cached entry")
	}
	if string(response) != string(testResponse) {
		t.Errorf("Response mismatch: got %s, want %s", string(response), string(testResponse))
	}

	// Test FindSimilar with similar query (should hit)
	_, found, err = cache.FindSimilar("gpt-4", "What's the meaning of life?")
	if err != nil {
		t.Fatalf("FindSimilar failed: %v", err)
	}
	if !found {
		t.Error("Expected to find similar cached entry")
	}

	// Test FindSimilar with dissimilar query (should miss)
	_, found, err = cache.FindSimilar("gpt-4", "How to cook pasta?")
	if err != nil {
		t.Fatalf("FindSimilar failed: %v", err)
	}
	if found {
		t.Error("Should not find dissimilar query")
	}

	// Verify updated stats
	stats = cache.GetStats()
	if stats.HitCount < 1 {
		t.Errorf("Expected at least 1 hit, got %d", stats.HitCount)
	}
	if stats.MissCount < 1 {
		t.Errorf("Expected at least 1 miss, got %d", stats.MissCount)
	}
}

// TestHybridCachePendingRequest tests pending request flow
func TestHybridCachePendingRequest(t *testing.T) {
	// Skip if Milvus tests are disabled
	if os.Getenv("SKIP_MILVUS_TESTS") == "true" {
		t.Skip("Skipping Milvus-dependent test (SKIP_MILVUS_TESTS=true)")
	}

	t.Log("Starting TestHybridCachePendingRequest - this may take 30-60 seconds...")

	milvusConfig, cleanup, err := createTestMilvusConfig("test_hybrid_pending", 64, true)
	if err != nil {
		t.Fatalf("Failed to create test config: %v", err)
	}
	defer cleanup()

	cache, err := NewHybridCache(HybridCacheOptions{
		Enabled:             true,
		SimilarityThreshold: 0.8,
		TTLSeconds:          300,
		MaxMemoryEntries:    100,
		MilvusConfigPath:    milvusConfig,
	})
	if err != nil {
		t.Fatalf("Failed to create hybrid cache: %v", err)
	}
	defer cache.Close()

	// Add pending request
	testQuery := "Explain quantum computing"
	err = cache.AddPendingRequest("req1", "gpt-4", testQuery, []byte("{}"), -1)
	if err != nil {
		t.Fatalf("Failed to add pending request: %v", err)
	}

	// Update with response
	testResponse := []byte(`{"answer": "Quantum computing uses qubits..."}`)
	err = cache.UpdateWithResponse("req1", testResponse, -1)
	if err != nil {
		t.Fatalf("Failed to update with response: %v", err)
	}

	// Wait for indexing
	time.Sleep(100 * time.Millisecond)

	// Try to find it
	response, found, err := cache.FindSimilar("gpt-4", testQuery)
	if err != nil {
		t.Fatalf("FindSimilar failed: %v", err)
	}
	if !found {
		t.Error("Expected to find cached entry after update")
	}
	if string(response) != string(testResponse) {
		t.Errorf("Response mismatch: got %s, want %s", string(response), string(testResponse))
	}
}

// TestHybridCacheEviction tests memory eviction behavior
func TestHybridCacheEviction(t *testing.T) {
	// Skip if Milvus tests are disabled
	if os.Getenv("SKIP_MILVUS_TESTS") == "true" {
		t.Skip("Skipping Milvus-dependent test (SKIP_MILVUS_TESTS=true)")
	}

	t.Log("Starting TestHybridCacheEviction - this may take 30-60 seconds...")

	milvusConfig, cleanup, err := createTestMilvusConfig("test_hybrid_eviction", 64, true)
	if err != nil {
		t.Fatalf("Failed to create test config: %v", err)
	}
	defer cleanup()

	// Create cache with very small memory limit
	cache, err := NewHybridCache(HybridCacheOptions{
		Enabled:             true,
		SimilarityThreshold: 0.8,
		TTLSeconds:          300,
		MaxMemoryEntries:    5, // Only 5 entries in memory
		MilvusConfigPath:    milvusConfig,
	})
	if err != nil {
		t.Fatalf("Failed to create hybrid cache: %v", err)
	}
	defer cache.Close()

	// Add 10 entries (will trigger evictions)
	for i := 0; i < 10; i++ {
		query := fmt.Sprintf("Query number %d", i)
		response := []byte(fmt.Sprintf(`{"answer": "Response %d"}`, i))
		err = cache.AddEntry(fmt.Sprintf("req%d", i), "gpt-4", query, []byte("{}"), response, -1)
		if err != nil {
			t.Fatalf("Failed to add entry %d: %v", i, err)
		}
	}

	// Check that we have at most MaxMemoryEntries in HNSW
	stats := cache.GetStats()
	if stats.TotalEntries > 5 {
		t.Errorf("Expected at most 5 entries in memory, got %d", stats.TotalEntries)
	}

	// All entries should still be in Milvus
	// Try to find a recent entry (should be in memory)
	// Wait for Milvus to index all entries
	time.Sleep(2 * time.Second)
	_, found, err := cache.FindSimilar("gpt-4", "Query number 9")
	if err != nil {
		t.Fatalf("FindSimilar failed: %v", err)
	}
	if !found {
		t.Error("Expected to find recent entry")
	}

	// Try to find an old evicted entry (should be in Milvus)
	_, _, err = cache.FindSimilar("gpt-4", "Query number 0")
	if err != nil {
		t.Fatalf("FindSimilar failed: %v", err)
	}
	// May or may not find it depending on Milvus indexing speed
	// Just verify no error
}

// TestHybridCacheLocalCacheHit tests local cache hot path
func TestHybridCacheLocalCacheHit(t *testing.T) {
	// Skip if Milvus tests are disabled
	if os.Getenv("SKIP_MILVUS_TESTS") == "true" {
		t.Skip("Skipping Milvus-dependent test (SKIP_MILVUS_TESTS=true)")
	}

	t.Log("Starting TestHybridCacheLocalCacheHit - this may take 30-60 seconds...")

	milvusConfig, cleanup, err := createTestMilvusConfig("test_hybrid_local", 64, true)
	if err != nil {
		t.Fatalf("Failed to create test config: %v", err)
	}
	defer cleanup()

	cache, err := NewHybridCache(HybridCacheOptions{
		Enabled:             true,
		SimilarityThreshold: 0.8,
		TTLSeconds:          300,
		MaxMemoryEntries:    100,
		MilvusConfigPath:    milvusConfig,
	})
	if err != nil {
		t.Fatalf("Failed to create hybrid cache: %v", err)
	}
	defer cache.Close()

	// Add an entry
	testQuery := "What is machine learning?"
	testResponse := []byte(`{"answer": "ML is..."}`)
	err = cache.AddEntry("req1", "gpt-4", testQuery, []byte("{}"), testResponse, -1)
	if err != nil {
		t.Fatalf("Failed to add entry: %v", err)
	}

	// Wait longer for Milvus to index the entry
	time.Sleep(2 * time.Second)

	// First search - should populate local cache
	response1, found, err := cache.FindSimilar("gpt-4", testQuery)
	if err != nil {
		t.Fatalf("FindSimilar failed: %v", err)
	}
	if !found {
		t.Fatal("Expected to find entry")
	}
	t.Logf("First search returned: %s", string(response1))

	// Second search - should hit local cache (much faster)
	startTime := time.Now()
	response, found, err := cache.FindSimilar("gpt-4", testQuery)
	localLatency := time.Since(startTime)
	if err != nil {
		t.Fatalf("FindSimilar failed: %v", err)
	}
	if !found {
		t.Fatal("Expected to find entry in local cache")
	}
	t.Logf("Second search returned: %s", string(response))
	if string(response) != string(testResponse) {
		t.Errorf("Response mismatch: got %s, want %s", string(response), string(testResponse))
	}

	// Local cache should be very fast (< 10ms)
	if localLatency > 10*time.Millisecond {
		t.Logf("Local cache hit took %v (expected < 10ms, but may vary)", localLatency)
	}

	stats := cache.GetStats()
	if stats.HitCount < 2 {
		t.Errorf("Expected at least 2 hits, got %d", stats.HitCount)
	}
}

// Ensures hybrid layer search skips candidates that are already worse than the frontier.
func TestHybridCacheSearchLayerPrunesWeakerBranch(t *testing.T) {
	// Regression fixture: the buggy comparison let the frontier accept a much
	// worse neighbor (node 3) even after ef was saturated. That re-opened the
	// branch to node 4, so the search would walk every reachable node—hurting
	// latency and risking a worse match. We wire an artificial edge (3→4) to
	// isolate the pruning logic; production HNSW builders try to avoid such links.
	embeddings := [][]float32{
		{0.80},  // node 0: entry point
		{0.79},  // node 1: near-tie neighbor
		{0.78},  // node 2: another strong neighbor
		{0.10},  // node 3: weak branch that should be pruned
		{0.995}, // node 4: hidden best reachable only via node 3
	}

	nodes := []*HNSWNode{
		{
			entryIndex: 0,
			neighbors: map[int][]int{
				0: {1, 2, 3},
			},
			maxLayer: 0,
		},
		{
			entryIndex: 1,
			neighbors: map[int][]int{
				0: {0},
			},
			maxLayer: 0,
		},
		{
			entryIndex: 2,
			neighbors: map[int][]int{
				0: {0},
			},
			maxLayer: 0,
		},
		{
			entryIndex: 3,
			neighbors: map[int][]int{
				0: {0, 4},
			},
			maxLayer: 0,
		},
		{
			entryIndex: 4,
			neighbors: map[int][]int{
				0: {3},
			},
			maxLayer: 0,
		},
	}

	nodeIndex := map[int]*HNSWNode{
		0: nodes[0],
		1: nodes[1],
		2: nodes[2],
		3: nodes[3],
		4: nodes[4],
	}

	cache := &HybridCache{
		hnswIndex: &HNSWIndex{
			nodes:          nodes,
			nodeIndex:      nodeIndex,
			entryPoint:     0,
			maxLayer:       0,
			efConstruction: 4,
			M:              4,
			Mmax:           4,
			Mmax0:          4,
			ml:             1,
		},
		embeddings: embeddings,
		idMap:      map[int]string{},
	}

	results := cache.searchLayerHybrid([]float32{1}, 3, 0, []int{0})
	if len(results) != 3 {
		t.Fatalf("expected frontier to keep three best neighbors, got %v", results)
	}
	if slices.Contains(results, 4) {
		t.Fatalf("expected weaker branch to stay pruned, got %v", results)
	}
	if !slices.Contains(results, 1) {
		t.Fatalf("expected best neighbor 1 to remain in results, got %v", results)
	}
}

// BenchmarkHybridCacheAddEntry benchmarks adding entries to hybrid cache
func BenchmarkHybridCacheAddEntry(b *testing.B) {
	if os.Getenv("MILVUS_URI") == "" {
		b.Skip("Skipping: MILVUS_URI not set")
	}

	milvusConfig := "/tmp/bench_milvus_config.yaml"
	err := os.WriteFile(milvusConfig, []byte(`
milvus:
  address: "localhost:19530"
  collection_name: "bench_hybrid_cache"
  dimension: 384
  index_type: "HNSW"
  metric_type: "IP"
`),
		0o644)
	if err != nil {
		b.Fatalf("Failed to create test config: %v", err)
	}
	defer os.Remove(milvusConfig)

	cache, err := NewHybridCache(HybridCacheOptions{
		Enabled:             true,
		SimilarityThreshold: 0.8,
		TTLSeconds:          300,
		MaxMemoryEntries:    10000,
		MilvusConfigPath:    milvusConfig,
	})
	if err != nil {
		b.Fatalf("Failed to create hybrid cache: %v", err)
	}
	defer cache.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		query := fmt.Sprintf("Benchmark query number %d", i)
		response := []byte(fmt.Sprintf(`{"answer": "Response %d"}`, i))
		err := cache.AddEntry(fmt.Sprintf("req%d", i), "gpt-4", query, []byte("{}"), response, -1)
		if err != nil {
			b.Fatalf("AddEntry failed: %v", err)
		}
	}
}

// BenchmarkHybridCacheFindSimilar benchmarks searching in hybrid cache
func BenchmarkHybridCacheFindSimilar(b *testing.B) {
	if os.Getenv("MILVUS_URI") == "" {
		b.Skip("Skipping: MILVUS_URI not set")
	}

	milvusConfig := "/tmp/bench_milvus_search_config.yaml"
	err := os.WriteFile(milvusConfig, []byte(`
milvus:
  address: "localhost:19530"
  collection_name: "bench_hybrid_search"
  dimension: 384
  index_type: "HNSW"
  metric_type: "IP"
`),
		0o644)
	if err != nil {
		b.Fatalf("Failed to create test config: %v", err)
	}
	defer os.Remove(milvusConfig)

	cache, err := NewHybridCache(HybridCacheOptions{
		Enabled:             true,
		SimilarityThreshold: 0.8,
		TTLSeconds:          300,
		MaxMemoryEntries:    1000,
		MilvusConfigPath:    milvusConfig,
	})
	if err != nil {
		b.Fatalf("Failed to create hybrid cache: %v", err)
	}
	defer cache.Close()

	// Pre-populate cache
	for i := 0; i < 100; i++ {
		query := fmt.Sprintf("Benchmark query number %d", i)
		response := []byte(fmt.Sprintf(`{"answer": "Response %d"}`, i))
		err := cache.AddEntry(fmt.Sprintf("req%d", i), "gpt-4", query, []byte("{}"), response, -1)
		if err != nil {
			b.Fatalf("AddEntry failed: %v", err)
		}
	}

	time.Sleep(500 * time.Millisecond) // Allow indexing

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		query := fmt.Sprintf("Benchmark query number %d", i%100)
		_, _, err := cache.FindSimilar("gpt-4", query)
		if err != nil {
			b.Fatalf("FindSimilar failed: %v", err)
		}
	}
}

// TestBenchmarkResult stores detailed benchmark metrics for test cases
type TestBenchmarkResult struct {
	CacheType           string
	CacheSize           int
	Operation           string
	AvgLatencyNs        int64
	AvgLatencyMs        float64
	P50LatencyMs        float64
	P95LatencyMs        float64
	P99LatencyMs        float64
	QPS                 float64
	MemoryUsageMB       float64
	HitRate             float64
	DatabaseCalls       int64
	TotalRequests       int64
	DatabaseCallPercent float64
}

// LatencyDistribution tracks percentile latencies
type LatencyDistribution struct {
	latencies []time.Duration
	mu        sync.Mutex
}

func (ld *LatencyDistribution) Record(latency time.Duration) {
	ld.mu.Lock()
	defer ld.mu.Unlock()
	ld.latencies = append(ld.latencies, latency)
}

func (ld *LatencyDistribution) GetPercentile(p float64) float64 {
	ld.mu.Lock()
	defer ld.mu.Unlock()

	if len(ld.latencies) == 0 {
		return 0
	}

	// Sort latencies
	sorted := make([]time.Duration, len(ld.latencies))
	copy(sorted, ld.latencies)
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	idx := int(float64(len(sorted)) * p)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}

	return float64(sorted[idx].Nanoseconds()) / 1e6
}

// DatabaseCallCounter tracks Milvus database calls
type DatabaseCallCounter struct {
	calls int64
}

func (dcc *DatabaseCallCounter) Increment() {
	atomic.AddInt64(&dcc.calls, 1)
}

func (dcc *DatabaseCallCounter) Get() int64 {
	return atomic.LoadInt64(&dcc.calls)
}

func (dcc *DatabaseCallCounter) Reset() {
	atomic.StoreInt64(&dcc.calls, 0)
}

// createTestMilvusConfig creates a temporary Milvus config file for testing
// Returns the path to the config file and cleanup function
func createTestMilvusConfig(collectionName string, efConstruction int, dropOnStartup bool) (string, func(), error) {
	configPath := fmt.Sprintf("/tmp/test_milvus_%s_config.yaml", collectionName)

	configYAML := fmt.Sprintf(`connection:
  host: "localhost"
  port: 19530
  timeout: 30
collection:
  name: "%s"
  vector_field:
    name: "embedding"
    dimension: 384
    metric_type: "IP"
  index:
    type: "HNSW"
    params:
      M: 16
      efConstruction: %d
search:
  params:
    ef: 64
  topk: 10
development:
  auto_create_collection: true
  drop_collection_on_startup: %t
`, collectionName, efConstruction, dropOnStartup)

	err := os.WriteFile(configPath, []byte(configYAML), 0o644)
	if err != nil {
		return "", nil, fmt.Errorf("failed to create test config: %w", err)
	}

	cleanup := func() {
		os.Remove(configPath)
	}

	return configPath, cleanup, nil
}

// getMilvusConfigPath returns the path to milvus.yaml config file
func getMilvusConfigPath() string {
	// Check for environment variable first
	if envPath := os.Getenv("MILVUS_CONFIG_PATH"); envPath != "" {
		if _, err := os.Stat(envPath); err == nil {
			return envPath
		}
	}

	// Try relative from project root (when run via make)
	configPath := "config/semantic-cache/milvus.yaml"
	if _, err := os.Stat(configPath); err == nil {
		return configPath
	}

	// Fallback to relative from test directory
	return "../../../../../config/semantic-cache/milvus.yaml"
}

// BenchmarkHybridVsMilvus is the comprehensive benchmark comparing hybrid cache vs pure Milvus
// This validates the claims from the hybrid HNSW storage architecture paper
func BenchmarkHybridVsMilvus(b *testing.B) {
	// Initialize BERT model
	useCPU := os.Getenv("USE_CPU") != "false"
	modelName := "sentence-transformers/all-MiniLM-L6-v2"
	if err := candle_binding.InitModel(modelName, useCPU); err != nil {
		b.Fatalf("Failed to initialize BERT model: %v", err)
	}

	// Test configurations - realistic production scales
	cacheSizes := []int{
		10000,  // Medium: 10K entries
		50000,  // Large: 50K entries
		100000, // Extra Large: 100K entries
	}

	// CSV output file - save to project benchmark_results directory
	// Use PROJECT_ROOT environment variable, fallback to working directory
	projectRoot := os.Getenv("PROJECT_ROOT")
	if projectRoot == "" {
		// If not set, use current working directory
		var err error
		projectRoot, err = os.Getwd()
		if err != nil {
			b.Logf("Warning: Could not determine working directory: %v", err)
			projectRoot = "."
		}
	}
	resultsDir := filepath.Join(projectRoot, "benchmark_results", "hybrid_vs_milvus")
	_ = os.MkdirAll(resultsDir, 0o755)
	timestamp := time.Now().Format("20060102_150405")
	csvPath := filepath.Join(resultsDir, fmt.Sprintf("results_%s.csv", timestamp))
	csvFile, err := os.Create(csvPath)
	if err != nil {
		b.Logf("Warning: Could not create CSV file at %s: %v", csvPath, err)
	} else {
		defer csvFile.Close()
		b.Logf("Results will be saved to: %s", csvPath)
		// Write CSV header
		if _, err := csvFile.WriteString("cache_type,cache_size,operation,avg_latency_ns,avg_latency_ms,p50_ms,p95_ms,p99_ms,qps,memory_mb,hit_rate,db_calls,total_requests,db_call_percent\n"); err != nil {
			b.Logf("Warning: Could not write CSV header: %v", err)
		}
	}

	b.Logf("=== Hybrid Cache vs Pure Milvus Benchmark ===")
	b.Logf("")

	for _, cacheSize := range cacheSizes {
		b.Run(fmt.Sprintf("CacheSize_%d", cacheSize), func(b *testing.B) {
			// Generate test queries
			b.Logf("Generating %d test queries...", cacheSize)
			testQueries := make([]string, cacheSize)
			for i := 0; i < cacheSize; i++ {
				testQueries[i] = generateQuery(MediumContent, i)
			}

			// Test two realistic hit rate scenarios
			scenarios := []struct {
				name    string
				hitRate float64
			}{
				{"HitRate_5pct", 0.05},  // 5% hit rate - very realistic for semantic cache
				{"HitRate_20pct", 0.20}, // 20% hit rate - optimistic but realistic
			}

			// Generate search queries for each scenario
			allSearchQueries := make(map[string][]string)
			for _, scenario := range scenarios {
				queries := make([]string, 100)
				hitCount := int(scenario.hitRate * 100)

				// Hits: reuse cached queries
				for i := 0; i < hitCount; i++ {
					queries[i] = testQueries[i%cacheSize]
				}

				// Misses: generate new queries
				for i := hitCount; i < 100; i++ {
					queries[i] = generateQuery(MediumContent, cacheSize+i)
				}

				allSearchQueries[scenario.name] = queries
				b.Logf("Generated queries for %s: %d hits, %d misses",
					scenario.name, hitCount, 100-hitCount)
			}

			// ============================================================
			// 1. Benchmark Pure Milvus Cache (Optional via SKIP_MILVUS env var)
			// ============================================================
			b.Run("Milvus", func(b *testing.B) {
				if os.Getenv("SKIP_MILVUS") == "true" {
					b.Skip("Skipping Milvus benchmark (SKIP_MILVUS=true)")
					return
				}
				b.Logf("\n=== Testing Pure Milvus Cache ===")

				milvusCache, err := NewMilvusCache(MilvusCacheOptions{
					Enabled:             true,
					SimilarityThreshold: 0.80,
					TTLSeconds:          3600,
					ConfigPath:          getMilvusConfigPath(),
				})
				if err != nil {
					b.Fatalf("Failed to create Milvus cache: %v", err)
				}
				defer milvusCache.Close()

				// Wait for Milvus to be ready
				time.Sleep(2 * time.Second)

				// Populate cache using batch insert for speed
				b.Logf("Populating Milvus with %d entries (using batch insert)...", cacheSize)
				populateStart := time.Now()

				// Prepare all entries
				entries := make([]CacheEntry, cacheSize)
				for i := 0; i < cacheSize; i++ {
					entries[i] = CacheEntry{
						RequestID:    fmt.Sprintf("req-milvus-%d", i),
						Model:        "test-model",
						Query:        testQueries[i],
						RequestBody:  []byte(fmt.Sprintf("request-%d", i)),
						ResponseBody: []byte(fmt.Sprintf("response-%d-this-is-a-longer-response-body-to-simulate-realistic-llm-output", i)),
					}
				}

				// Insert in batches of 100
				batchSize := 100
				for i := 0; i < cacheSize; i += batchSize {
					end := i + batchSize
					if end > cacheSize {
						end = cacheSize
					}

					err := milvusCache.AddEntriesBatch(entries[i:end])
					if err != nil {
						b.Fatalf("Failed to add batch: %v", err)
					}

					if (i+batchSize)%1000 == 0 {
						b.Logf("  Populated %d/%d entries", i+batchSize, cacheSize)
					}
				}

				// Flush once after all batches
				b.Logf("Flushing Milvus...")
				if err := milvusCache.Flush(); err != nil {
					b.Logf("Warning: flush failed: %v", err)
				}

				populateTime := time.Since(populateStart)
				b.Logf("✓ Populated in %v (%.0f entries/sec)", populateTime, float64(cacheSize)/populateTime.Seconds())

				// Wait for Milvus to be ready
				time.Sleep(2 * time.Second)

				// Test each hit rate scenario
				for _, scenario := range scenarios {
					searchQueries := allSearchQueries[scenario.name]

					b.Run(scenario.name, func(b *testing.B) {
						// Benchmark search operations
						b.Logf("Running search benchmark for %s...", scenario.name)
						latencyDist := &LatencyDistribution{latencies: make([]time.Duration, 0, b.N)}
						dbCallCounter := &DatabaseCallCounter{}
						hits := 0
						misses := 0

						b.ResetTimer()
						start := time.Now()

						for i := 0; i < b.N; i++ {
							queryIdx := i % len(searchQueries)
							searchStart := time.Now()

							// Every Milvus FindSimilar is a database call
							dbCallCounter.Increment()

							_, found, err := milvusCache.FindSimilar("test-model", searchQueries[queryIdx])
							searchLatency := time.Since(searchStart)

							if err != nil {
								b.Logf("Warning: search error at iteration %d: %v", i, err)
							}

							latencyDist.Record(searchLatency)

							if found {
								hits++
							} else {
								misses++
							}
						}

						elapsed := time.Since(start)
						b.StopTimer()

						// Calculate metrics
						avgLatencyNs := elapsed.Nanoseconds() / int64(b.N)
						avgLatencyMs := float64(avgLatencyNs) / 1e6
						qps := float64(b.N) / elapsed.Seconds()
						hitRate := float64(hits) / float64(b.N) * 100
						dbCalls := dbCallCounter.Get()
						dbCallPercent := float64(dbCalls) / float64(b.N) * 100

						// Memory usage estimation
						memUsageMB := estimateMilvusMemory(cacheSize)

						result := TestBenchmarkResult{
							CacheType:           "milvus",
							CacheSize:           cacheSize,
							Operation:           "search",
							AvgLatencyNs:        avgLatencyNs,
							AvgLatencyMs:        avgLatencyMs,
							P50LatencyMs:        latencyDist.GetPercentile(0.50),
							P95LatencyMs:        latencyDist.GetPercentile(0.95),
							P99LatencyMs:        latencyDist.GetPercentile(0.99),
							QPS:                 qps,
							MemoryUsageMB:       memUsageMB,
							HitRate:             hitRate,
							DatabaseCalls:       dbCalls,
							TotalRequests:       int64(b.N),
							DatabaseCallPercent: dbCallPercent,
						}

						// Report results
						b.Logf("\n--- Milvus Results (%s) ---", scenario.name)
						b.Logf("Avg Latency: %.2f ms", avgLatencyMs)
						b.Logf("P50: %.2f ms, P95: %.2f ms, P99: %.2f ms", result.P50LatencyMs, result.P95LatencyMs, result.P99LatencyMs)
						b.Logf("QPS: %.0f", qps)
						b.Logf("Hit Rate: %.1f%% (expected: %.0f%%)", hitRate, scenario.hitRate*100)
						b.Logf("Hits: %d, Misses: %d out of %d total", hits, misses, b.N)
						b.Logf("Database Calls: %d/%d (%.0f%%)", dbCalls, b.N, dbCallPercent)
						b.Logf("Memory Usage: %.1f MB", memUsageMB)

						// Write to CSV
						if csvFile != nil {
							writeBenchmarkResultToCSV(csvFile, result)
						}

						b.ReportMetric(avgLatencyMs, "ms/op")
						b.ReportMetric(qps, "qps")
						b.ReportMetric(hitRate, "hit_rate_%")
					})
				}
			})

			// ============================================================
			// 2. Benchmark Hybrid Cache
			// ============================================================
			b.Run("Hybrid", func(b *testing.B) {
				b.Logf("\n=== Testing Hybrid Cache ===")

				hybridCache, err := NewHybridCache(HybridCacheOptions{
					Enabled:             true,
					SimilarityThreshold: 0.80,
					TTLSeconds:          3600,
					MaxMemoryEntries:    cacheSize,
					HNSWM:               16,
					HNSWEfConstruction:  200,
					MilvusConfigPath:    getMilvusConfigPath(),
				})
				if err != nil {
					b.Fatalf("Failed to create Hybrid cache: %v", err)
				}
				defer hybridCache.Close()

				// Wait for initialization
				time.Sleep(2 * time.Second)

				// Populate cache using batch insert for speed
				b.Logf("Populating Hybrid cache with %d entries (using batch insert)...", cacheSize)
				populateStart := time.Now()

				// Prepare all entries
				entries := make([]CacheEntry, cacheSize)
				for i := 0; i < cacheSize; i++ {
					entries[i] = CacheEntry{
						RequestID:    fmt.Sprintf("req-hybrid-%d", i),
						Model:        "test-model",
						Query:        testQueries[i],
						RequestBody:  []byte(fmt.Sprintf("request-%d", i)),
						ResponseBody: []byte(fmt.Sprintf("response-%d-this-is-a-longer-response-body-to-simulate-realistic-llm-output", i)),
					}
				}

				// Insert in batches of 100
				batchSize := 100
				for i := 0; i < cacheSize; i += batchSize {
					end := i + batchSize
					if end > cacheSize {
						end = cacheSize
					}

					err := hybridCache.AddEntriesBatch(entries[i:end])
					if err != nil {
						b.Fatalf("Failed to add batch: %v", err)
					}

					if (i+batchSize)%1000 == 0 {
						b.Logf("  Populated %d/%d entries", i+batchSize, cacheSize)
					}
				}

				// Flush once after all batches
				b.Logf("Flushing Milvus...")
				if err := hybridCache.Flush(); err != nil {
					b.Logf("Warning: flush failed: %v", err)
				}

				populateTime := time.Since(populateStart)
				b.Logf("✓ Populated in %v (%.0f entries/sec)", populateTime, float64(cacheSize)/populateTime.Seconds())

				// Wait for Milvus to be ready
				time.Sleep(2 * time.Second)

				// Test each hit rate scenario
				for _, scenario := range scenarios {
					searchQueries := allSearchQueries[scenario.name]

					b.Run(scenario.name, func(b *testing.B) {
						// Get initial memory stats
						var memBefore runtime.MemStats
						runtime.ReadMemStats(&memBefore)

						// Benchmark search operations
						b.Logf("Running search benchmark for %s...", scenario.name)
						latencyDist := &LatencyDistribution{latencies: make([]time.Duration, 0, b.N)}
						hits := 0
						misses := 0

						// Track database calls (Hybrid should make fewer calls due to threshold filtering)
						initialMilvusCallCount := hybridCache.milvusCache.hitCount + hybridCache.milvusCache.missCount

						b.ResetTimer()
						start := time.Now()

						for i := 0; i < b.N; i++ {
							queryIdx := i % len(searchQueries)
							searchStart := time.Now()

							_, found, err := hybridCache.FindSimilar("test-model", searchQueries[queryIdx])
							searchLatency := time.Since(searchStart)

							if err != nil {
								b.Logf("Warning: search error at iteration %d: %v", i, err)
							}

							latencyDist.Record(searchLatency)

							if found {
								hits++
							} else {
								misses++
							}
						}

						elapsed := time.Since(start)
						b.StopTimer()

						// Calculate database calls (both hits and misses involve Milvus calls)
						finalMilvusCallCount := hybridCache.milvusCache.hitCount + hybridCache.milvusCache.missCount
						dbCalls := finalMilvusCallCount - initialMilvusCallCount

						// Get final memory stats
						var memAfter runtime.MemStats
						runtime.ReadMemStats(&memAfter)

						// Fix: Prevent unsigned integer underflow if GC ran during benchmark
						var memUsageMB float64
						if memAfter.Alloc >= memBefore.Alloc {
							memUsageMB = float64(memAfter.Alloc-memBefore.Alloc) / 1024 / 1024
						} else {
							// GC ran, use estimation instead
							memUsageMB = estimateHybridMemory(cacheSize)
						}

						// Calculate metrics
						avgLatencyNs := elapsed.Nanoseconds() / int64(b.N)
						avgLatencyMs := float64(avgLatencyNs) / 1e6
						qps := float64(b.N) / elapsed.Seconds()
						hitRate := float64(hits) / float64(b.N) * 100
						dbCallPercent := float64(dbCalls) / float64(b.N) * 100

						result := TestBenchmarkResult{
							CacheType:           "hybrid",
							CacheSize:           cacheSize,
							Operation:           "search",
							AvgLatencyNs:        avgLatencyNs,
							AvgLatencyMs:        avgLatencyMs,
							P50LatencyMs:        latencyDist.GetPercentile(0.50),
							P95LatencyMs:        latencyDist.GetPercentile(0.95),
							P99LatencyMs:        latencyDist.GetPercentile(0.99),
							QPS:                 qps,
							MemoryUsageMB:       memUsageMB,
							HitRate:             hitRate,
							DatabaseCalls:       dbCalls,
							TotalRequests:       int64(b.N),
							DatabaseCallPercent: dbCallPercent,
						}

						// Report results
						b.Logf("\n--- Hybrid Cache Results (%s) ---", scenario.name)
						b.Logf("Avg Latency: %.2f ms", avgLatencyMs)
						b.Logf("P50: %.2f ms, P95: %.2f ms, P99: %.2f ms", result.P50LatencyMs, result.P95LatencyMs, result.P99LatencyMs)
						b.Logf("QPS: %.0f", qps)
						b.Logf("Hit Rate: %.1f%% (expected: %.0f%%)", hitRate, scenario.hitRate*100)
						b.Logf("Hits: %d, Misses: %d out of %d total", hits, misses, b.N)
						b.Logf("Database Calls: %d/%d (%.0f%%)", dbCalls, b.N, dbCallPercent)
						b.Logf("Memory Usage: %.1f MB", memUsageMB)

						// Write to CSV
						if csvFile != nil {
							writeBenchmarkResultToCSV(csvFile, result)
						}

						b.ReportMetric(avgLatencyMs, "ms/op")
						b.ReportMetric(qps, "qps")
						b.ReportMetric(hitRate, "hit_rate_%")
						b.ReportMetric(dbCallPercent, "db_call_%")
					})
				}
			})
		})
	}
}

// BenchmarkComponentLatency measures individual component latencies
func BenchmarkComponentLatency(b *testing.B) {
	// Initialize BERT model
	useCPU := os.Getenv("USE_CPU") != "false"
	modelName := "sentence-transformers/all-MiniLM-L6-v2"
	if err := candle_binding.InitModel(modelName, useCPU); err != nil {
		b.Fatalf("Failed to initialize BERT model: %v", err)
	}

	cacheSize := 10000
	testQueries := make([]string, cacheSize)
	for i := 0; i < cacheSize; i++ {
		testQueries[i] = generateQuery(MediumContent, i)
	}

	b.Run("EmbeddingGeneration", func(b *testing.B) {
		query := testQueries[0]
		b.ResetTimer()
		start := time.Now()
		for i := 0; i < b.N; i++ {
			_, err := candle_binding.GetEmbedding(query, 0)
			if err != nil {
				b.Fatal(err)
			}
		}
		elapsed := time.Since(start)
		avgMs := float64(elapsed.Nanoseconds()) / float64(b.N) / 1e6
		b.Logf("Embedding generation: %.2f ms/op", avgMs)
		b.ReportMetric(avgMs, "ms/op")
	})

	b.Run("HNSWSearch", func(b *testing.B) {
		// Build HNSW index
		cache := NewInMemoryCache(InMemoryCacheOptions{
			Enabled:             true,
			SimilarityThreshold: 0.80,
			MaxEntries:          cacheSize,
			UseHNSW:             true,
			HNSWM:               16,
			HNSWEfConstruction:  200,
		})

		b.Logf("Building HNSW index with %d entries...", cacheSize)
		for i := 0; i < cacheSize; i++ {
			_ = cache.AddEntry(fmt.Sprintf("req-%d", i), "model", testQueries[i], []byte("req"), []byte("resp"), -1)
		}
		b.Logf("✓ HNSW index built")

		query := testQueries[0]

		b.ResetTimer()
		start := time.Now()
		for i := 0; i < b.N; i++ {
			// Note: HNSW search uses entries slice internally
			_, _, _ = cache.FindSimilar("model", query)
		}
		elapsed := time.Since(start)
		avgMs := float64(elapsed.Nanoseconds()) / float64(b.N) / 1e6
		b.Logf("HNSW search: %.2f ms/op", avgMs)
		b.ReportMetric(avgMs, "ms/op")
	})

	b.Run("MilvusVectorSearch", func(b *testing.B) {
		milvusCache, err := NewMilvusCache(MilvusCacheOptions{
			Enabled:             true,
			SimilarityThreshold: 0.80,
			TTLSeconds:          3600,
			ConfigPath:          getMilvusConfigPath(),
		})
		if err != nil {
			b.Fatalf("Failed to create Milvus cache: %v", err)
		}
		defer milvusCache.Close()

		time.Sleep(2 * time.Second)

		b.Logf("Populating Milvus with %d entries...", cacheSize)
		for i := 0; i < cacheSize; i++ {
			_ = milvusCache.AddEntry(fmt.Sprintf("req-%d", i), "model", testQueries[i], []byte("req"), []byte("resp"), -1)
		}
		time.Sleep(2 * time.Second)
		b.Logf("✓ Milvus populated")

		query := testQueries[0]

		b.ResetTimer()
		start := time.Now()
		for i := 0; i < b.N; i++ {
			_, _, _ = milvusCache.FindSimilar("model", query)
		}
		elapsed := time.Since(start)
		avgMs := float64(elapsed.Nanoseconds()) / float64(b.N) / 1e6
		b.Logf("Milvus vector search: %.2f ms/op", avgMs)
		b.ReportMetric(avgMs, "ms/op")
	})

	b.Run("MilvusGetByID", func(b *testing.B) {
		// This would test Milvus get by ID if we exposed that method
		b.Skip("Milvus GetByID not exposed in current implementation")
	})
}

// BenchmarkThroughputUnderLoad tests throughput with concurrent requests
func BenchmarkThroughputUnderLoad(b *testing.B) {
	// Initialize BERT model
	useCPU := os.Getenv("USE_CPU") != "false"
	modelName := "sentence-transformers/all-MiniLM-L6-v2"
	if err := candle_binding.InitModel(modelName, useCPU); err != nil {
		b.Fatalf("Failed to initialize BERT model: %v", err)
	}

	cacheSize := 10000
	concurrencyLevels := []int{1, 10, 50, 100}

	testQueries := make([]string, cacheSize)
	for i := 0; i < cacheSize; i++ {
		testQueries[i] = generateQuery(MediumContent, i)
	}

	for _, concurrency := range concurrencyLevels {
		b.Run(fmt.Sprintf("Milvus_Concurrency_%d", concurrency), func(b *testing.B) {
			milvusCache, err := NewMilvusCache(MilvusCacheOptions{
				Enabled:             true,
				SimilarityThreshold: 0.80,
				TTLSeconds:          3600,
				ConfigPath:          getMilvusConfigPath(),
			})
			if err != nil {
				b.Fatalf("Failed to create Milvus cache: %v", err)
			}
			defer milvusCache.Close()

			time.Sleep(2 * time.Second)

			// Populate
			for i := 0; i < cacheSize; i++ {
				_ = milvusCache.AddEntry(fmt.Sprintf("req-%d", i), "model", testQueries[i], []byte("req"), []byte("resp"), -1)
			}
			time.Sleep(2 * time.Second)

			b.ResetTimer()
			b.SetParallelism(concurrency)
			start := time.Now()

			b.RunParallel(func(pb *testing.PB) {
				i := 0
				for pb.Next() {
					query := testQueries[i%len(testQueries)]
					_, _, _ = milvusCache.FindSimilar("model", query)
					i++
				}
			})

			elapsed := time.Since(start)
			qps := float64(b.N) / elapsed.Seconds()
			b.Logf("QPS with %d concurrent workers: %.0f", concurrency, qps)
			b.ReportMetric(qps, "qps")
		})

		b.Run(fmt.Sprintf("Hybrid_Concurrency_%d", concurrency), func(b *testing.B) {
			hybridCache, err := NewHybridCache(HybridCacheOptions{
				Enabled:             true,
				SimilarityThreshold: 0.80,
				TTLSeconds:          3600,
				MaxMemoryEntries:    cacheSize,
				HNSWM:               16,
				HNSWEfConstruction:  200,
				MilvusConfigPath:    getMilvusConfigPath(),
			})
			if err != nil {
				b.Fatalf("Failed to create Hybrid cache: %v", err)
			}
			defer hybridCache.Close()

			time.Sleep(2 * time.Second)

			// Populate
			for i := 0; i < cacheSize; i++ {
				_ = hybridCache.AddEntry(fmt.Sprintf("req-%d", i), "model", testQueries[i], []byte("req"), []byte("resp"), -1)
			}
			time.Sleep(2 * time.Second)

			b.ResetTimer()
			b.SetParallelism(concurrency)
			start := time.Now()

			b.RunParallel(func(pb *testing.PB) {
				i := 0
				for pb.Next() {
					query := testQueries[i%len(testQueries)]
					_, _, _ = hybridCache.FindSimilar("model", query)
					i++
				}
			})

			elapsed := time.Since(start)
			qps := float64(b.N) / elapsed.Seconds()
			b.Logf("QPS with %d concurrent workers: %.0f", concurrency, qps)
			b.ReportMetric(qps, "qps")
		})
	}
}

// Helper functions

func estimateMilvusMemory(cacheSize int) float64 {
	// Milvus memory estimation (rough)
	// - Embeddings: cacheSize × 384 × 4 bytes
	// - HNSW index: cacheSize × 16 × 2 × 4 bytes (M=16, bidirectional)
	// - Metadata: cacheSize × 0.5 KB
	embeddingMB := float64(cacheSize*384*4) / 1024 / 1024
	indexMB := float64(cacheSize*16*2*4) / 1024 / 1024
	metadataMB := float64(cacheSize) * 0.5 / 1024
	return embeddingMB + indexMB + metadataMB
}

func estimateHybridMemory(cacheSize int) float64 {
	// Hybrid memory estimation (in-memory HNSW only, documents in Milvus)
	// - Embeddings: cacheSize × 384 × 4 bytes
	// - HNSW index: cacheSize × 16 × 2 × 4 bytes (M=16, bidirectional)
	// - ID map: cacheSize × 50 bytes (average string length)
	embeddingMB := float64(cacheSize*384*4) / 1024 / 1024
	indexMB := float64(cacheSize*16*2*4) / 1024 / 1024
	idMapMB := float64(cacheSize*50) / 1024 / 1024
	return embeddingMB + indexMB + idMapMB
}

func writeBenchmarkResultToCSV(file *os.File, result TestBenchmarkResult) {
	line := fmt.Sprintf("%s,%d,%s,%d,%.3f,%.3f,%.3f,%.3f,%.0f,%.1f,%.1f,%d,%d,%.1f\n",
		result.CacheType,
		result.CacheSize,
		result.Operation,
		result.AvgLatencyNs,
		result.AvgLatencyMs,
		result.P50LatencyMs,
		result.P95LatencyMs,
		result.P99LatencyMs,
		result.QPS,
		result.MemoryUsageMB,
		result.HitRate,
		result.DatabaseCalls,
		result.TotalRequests,
		result.DatabaseCallPercent,
	)
	if _, err := file.WriteString(line); err != nil {
		// Ignore write errors in benchmark helper
		_ = err
	}
}

// TestHybridVsMilvusSmoke is a quick smoke test to verify both caches work
func TestHybridVsMilvusSmoke(t *testing.T) {
	// Skip if Milvus tests are disabled
	if os.Getenv("SKIP_MILVUS_TESTS") == "true" {
		t.Skip("Skipping Milvus-dependent test (SKIP_MILVUS_TESTS=true)")
	}

	t.Log("Starting TestHybridVsMilvusSmoke - this may take 2-3 minutes...")

	// Create test Milvus config
	milvusConfig, cleanup, err := createTestMilvusConfig("test_smoke_cache", 64, true)
	if err != nil {
		t.Fatalf("Failed to create test config: %v", err)
	}
	defer cleanup()

	// Initialize BERT model
	useCPU := os.Getenv("USE_CPU") != "false"
	modelName := "sentence-transformers/all-MiniLM-L6-v2"
	if err := candle_binding.InitModel(modelName, useCPU); err != nil {
		t.Fatalf("Failed to initialize BERT model: %v", err)
	}

	// Test Milvus cache
	t.Run("Milvus", func(t *testing.T) {
		cache, err := NewMilvusCache(MilvusCacheOptions{
			Enabled:             true,
			SimilarityThreshold: 0.85,
			TTLSeconds:          3600,
			ConfigPath:          milvusConfig,
		})
		if err != nil {
			t.Fatalf("Failed to create Milvus cache: %v", err)
		}
		defer cache.Close()

		time.Sleep(1 * time.Second)

		// Add entry
		err = cache.AddEntry("req-1", "model", "What is machine learning?", []byte("req"), []byte("ML is..."), -1)
		if err != nil {
			t.Fatalf("Failed to add entry: %v", err)
		}

		time.Sleep(1 * time.Second)

		// Find similar
		resp, found, err := cache.FindSimilar("model", "What is machine learning?")
		if err != nil {
			t.Fatalf("FindSimilar failed: %v", err)
		}
		if !found {
			t.Fatalf("Expected to find entry, but got miss")
		}
		if string(resp) != "ML is..." {
			t.Fatalf("Expected 'ML is...', got '%s'", string(resp))
		}

		t.Logf("✓ Milvus cache smoke test passed")
	})

	// Test Hybrid cache
	t.Run("Hybrid", func(t *testing.T) {
		cache, err := NewHybridCache(HybridCacheOptions{
			Enabled:             true,
			SimilarityThreshold: 0.85,
			TTLSeconds:          3600,
			MaxMemoryEntries:    1000,
			HNSWM:               16,
			HNSWEfConstruction:  200,
			MilvusConfigPath:    milvusConfig,
		})
		if err != nil {
			t.Fatalf("Failed to create Hybrid cache: %v", err)
		}
		defer cache.Close()

		time.Sleep(1 * time.Second)

		// Add entry
		err = cache.AddEntry("req-1", "model", "What is deep learning?", []byte("req"), []byte("DL is..."), -1)
		if err != nil {
			t.Fatalf("Failed to add entry: %v", err)
		}

		time.Sleep(1 * time.Second)

		// Find similar
		resp, found, err := cache.FindSimilar("model", "What is deep learning?")
		if err != nil {
			t.Fatalf("FindSimilar failed: %v", err)
		}
		if !found {
			t.Fatalf("Expected to find entry, but got miss")
		}
		if string(resp) != "DL is..." {
			t.Fatalf("Expected 'DL is...', got '%s'", string(resp))
		}

		t.Logf("✓ Hybrid cache smoke test passed")
	})
}

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
			[]byte("request1"), []byte("response1"), -1)
		if err != nil {
			t.Fatalf("Failed to add first entry: %v", err)
		}

		// Step 2: Add second entry (cache at capacity)
		err = cache.AddEntry("req2", "test-model", "Good morning",
			[]byte("request2"), []byte("response2"), -1)
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
			responseBody, found, findErr := cache.FindSimilar("test-model", "Hello world")
			if findErr != nil {
				t.Logf("FindSimilar failed (expected due to high threshold): %v", findErr)
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
			[]byte("request3"), []byte("response3"), -1)
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
		err := cache.AddPendingRequest("req1", "test-model", "test query", []byte("request"), -1)
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
		err = cache.UpdateWithResponse("req1", []byte("response1"), -1)
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

// TestInMemoryCacheHNSW tests the HNSW index functionality
func TestInMemoryCacheHNSW(t *testing.T) {
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		t.Skipf("Failed to initialize BERT model: %v", err)
	}

	// Test with HNSW enabled
	cacheHNSW := NewInMemoryCache(InMemoryCacheOptions{
		Enabled:             true,
		MaxEntries:          100,
		SimilarityThreshold: 0.85,
		TTLSeconds:          0,
		UseHNSW:             true,
		HNSWM:               16,
		HNSWEfConstruction:  200,
	})

	// Test without HNSW (linear search)
	cacheLinear := NewInMemoryCache(InMemoryCacheOptions{
		Enabled:             true,
		MaxEntries:          100,
		SimilarityThreshold: 0.85,
		TTLSeconds:          0,
		UseHNSW:             false,
	})

	testQueries := []struct {
		query    string
		model    string
		response string
	}{
		{"What is machine learning?", "test-model", "ML is a subset of AI"},
		{"Explain neural networks", "test-model", "NNs are inspired by the brain"},
		{"How does backpropagation work?", "test-model", "Backprop calculates gradients"},
		{"What is deep learning?", "test-model", "DL uses multiple layers"},
		{"Define artificial intelligence", "test-model", "AI mimics human intelligence"},
	}

	t.Run("HNSW_Basic_Operations", func(t *testing.T) {
		// Add entries to both caches
		for i, q := range testQueries {
			reqID := fmt.Sprintf("req%d", i)
			err := cacheHNSW.AddEntry(reqID, q.model, q.query, []byte(q.query), []byte(q.response), -1)
			if err != nil {
				t.Fatalf("Failed to add entry to HNSW cache: %v", err)
			}

			err = cacheLinear.AddEntry(reqID, q.model, q.query, []byte(q.query), []byte(q.response), -1)
			if err != nil {
				t.Fatalf("Failed to add entry to linear cache: %v", err)
			}
		}

		// Verify HNSW index was built
		if cacheHNSW.hnswIndex == nil {
			t.Fatal("HNSW index is nil")
		}
		if len(cacheHNSW.hnswIndex.nodes) != len(testQueries) {
			t.Errorf("Expected %d HNSW nodes, got %d", len(testQueries), len(cacheHNSW.hnswIndex.nodes))
		}

		// Test exact match search
		response, found, err := cacheHNSW.FindSimilar("test-model", "What is machine learning?")
		if err != nil {
			t.Fatalf("HNSW FindSimilar error: %v", err)
		}
		if !found {
			t.Error("HNSW should find exact match")
		}
		if string(response) != "ML is a subset of AI" {
			t.Errorf("Expected 'ML is a subset of AI', got %s", string(response))
		}

		// Test similar query search
		response, found, err = cacheHNSW.FindSimilar("test-model", "What is ML?")
		if err != nil {
			t.Logf("HNSW FindSimilar error (may not find due to threshold): %v", err)
		}
		if found {
			t.Logf("HNSW found similar entry: %s", string(response))
		}

		// Compare stats
		statsHNSW := cacheHNSW.GetStats()
		statsLinear := cacheLinear.GetStats()

		t.Logf("HNSW Cache Stats: Entries=%d, Hits=%d, Misses=%d, HitRatio=%.2f",
			statsHNSW.TotalEntries, statsHNSW.HitCount, statsHNSW.MissCount, statsHNSW.HitRatio)
		t.Logf("Linear Cache Stats: Entries=%d, Hits=%d, Misses=%d, HitRatio=%.2f",
			statsLinear.TotalEntries, statsLinear.HitCount, statsLinear.MissCount, statsLinear.HitRatio)
	})

	t.Run("HNSW_Rebuild_After_Cleanup", func(t *testing.T) {
		// Create cache with short TTL
		cacheTTL := NewInMemoryCache(InMemoryCacheOptions{
			Enabled:             true,
			MaxEntries:          100,
			SimilarityThreshold: 0.85,
			TTLSeconds:          1,
			UseHNSW:             true,
			HNSWM:               16,
			HNSWEfConstruction:  200,
		})

		// Add an entry
		err := cacheTTL.AddEntry("req1", "test-model", "test query", []byte("request"), []byte("response"), -1)
		if err != nil {
			t.Fatalf("Failed to add entry: %v", err)
		}

		initialNodes := len(cacheTTL.hnswIndex.nodes)
		if initialNodes != 1 {
			t.Errorf("Expected 1 HNSW node initially, got %d", initialNodes)
		}

		// Manually trigger cleanup (in real scenario, TTL would expire)
		cacheTTL.mu.Lock()
		cacheTTL.cleanupExpiredEntries()
		cacheTTL.mu.Unlock()

		t.Logf("After cleanup: %d entries, %d HNSW nodes",
			len(cacheTTL.entries), len(cacheTTL.hnswIndex.nodes))
	})
}

// ===== Benchmark Tests =====

// BenchmarkInMemoryCacheSearch benchmarks search performance with and without HNSW
func BenchmarkInMemoryCacheSearch(b *testing.B) {
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	// Test different cache sizes
	cacheSizes := []int{100, 500, 1000, 5000}

	for _, size := range cacheSizes {
		// Prepare test data
		entries := make([]struct {
			query    string
			response string
		}, size)

		for i := 0; i < size; i++ {
			entries[i].query = fmt.Sprintf("Test query number %d about machine learning and AI", i)
			entries[i].response = fmt.Sprintf("Response %d", i)
		}

		// Benchmark Linear Search
		b.Run(fmt.Sprintf("LinearSearch_%d_entries", size), func(b *testing.B) {
			cache := NewInMemoryCache(InMemoryCacheOptions{
				Enabled:             true,
				MaxEntries:          size * 2,
				SimilarityThreshold: 0.85,
				TTLSeconds:          0,
				UseHNSW:             false,
			})

			// Populate cache
			for i, entry := range entries {
				reqID := fmt.Sprintf("req%d", i)
				_ = cache.AddEntry(reqID, "test-model", entry.query, []byte(entry.query), []byte(entry.response), -1)
			}

			// Benchmark search
			searchQuery := "What is machine learning and artificial intelligence?"
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _, _ = cache.FindSimilar("test-model", searchQuery)
			}
		})

		// Benchmark HNSW Search
		b.Run(fmt.Sprintf("HNSWSearch_%d_entries", size), func(b *testing.B) {
			cache := NewInMemoryCache(InMemoryCacheOptions{
				Enabled:             true,
				MaxEntries:          size * 2,
				SimilarityThreshold: 0.85,
				TTLSeconds:          0,
				UseHNSW:             true,
				HNSWM:               16,
				HNSWEfConstruction:  200,
			})

			// Populate cache
			for i, entry := range entries {
				reqID := fmt.Sprintf("req%d", i)
				_ = cache.AddEntry(reqID, "test-model", entry.query, []byte(entry.query), []byte(entry.response), -1)
			}

			// Benchmark search
			searchQuery := "What is machine learning and artificial intelligence?"
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _, _ = cache.FindSimilar("test-model", searchQuery)
			}
		})
	}
}

// BenchmarkHNSWIndexConstruction benchmarks HNSW index construction time
func BenchmarkHNSWIndexConstruction(b *testing.B) {
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	entryCounts := []int{100, 500, 1000, 5000}

	for _, count := range entryCounts {
		b.Run(fmt.Sprintf("AddEntries_%d", count), func(b *testing.B) {
			// Generate test queries outside the benchmark loop
			testQueries := make([]string, count)
			for i := 0; i < count; i++ {
				testQueries[i] = fmt.Sprintf("Query %d: machine learning deep neural networks", i)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				cache := NewInMemoryCache(InMemoryCacheOptions{
					Enabled:             true,
					MaxEntries:          count * 2,
					SimilarityThreshold: 0.85,
					TTLSeconds:          0,
					UseHNSW:             true,
					HNSWM:               16,
					HNSWEfConstruction:  200,
				})
				b.StartTimer()

				// Add entries and build index
				for j := 0; j < count; j++ {
					reqID := fmt.Sprintf("req%d", j)
					_ = cache.AddEntry(reqID, "test-model", testQueries[j], []byte(testQueries[j]), []byte("response"), -1)
				}
			}
		})
	}
}

// BenchmarkHNSWParameters benchmarks different HNSW parameter configurations
func BenchmarkHNSWParameters(b *testing.B) {
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	cacheSize := 1000
	testConfigs := []struct {
		name           string
		m              int
		efConstruction int
	}{
		{"M8_EF100", 8, 100},
		{"M16_EF200", 16, 200},
		{"M32_EF400", 32, 400},
	}

	// Prepare test data
	entries := make([]struct {
		query    string
		response string
	}, cacheSize)

	for i := 0; i < cacheSize; i++ {
		entries[i].query = fmt.Sprintf("Query %d about AI and machine learning", i)
		entries[i].response = fmt.Sprintf("Response %d", i)
	}

	for _, config := range testConfigs {
		b.Run(config.name, func(b *testing.B) {
			cache := NewInMemoryCache(InMemoryCacheOptions{
				Enabled:             true,
				MaxEntries:          cacheSize * 2,
				SimilarityThreshold: 0.85,
				TTLSeconds:          0,
				UseHNSW:             true,
				HNSWM:               config.m,
				HNSWEfConstruction:  config.efConstruction,
			})

			// Populate cache
			for i, entry := range entries {
				reqID := fmt.Sprintf("req%d", i)
				_ = cache.AddEntry(reqID, "test-model", entry.query, []byte(entry.query), []byte(entry.response), -1)
			}

			// Benchmark search
			searchQuery := "What is artificial intelligence and machine learning?"
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _, _ = cache.FindSimilar("test-model", searchQuery)
			}
		})
	}
}

// BenchmarkCacheOperations benchmarks complete cache workflow
func BenchmarkCacheOperations(b *testing.B) {
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	b.Run("LinearSearch_AddAndFind", func(b *testing.B) {
		cache := NewInMemoryCache(InMemoryCacheOptions{
			Enabled:             true,
			MaxEntries:          10000,
			SimilarityThreshold: 0.85,
			TTLSeconds:          0,
			UseHNSW:             false,
		})

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			query := fmt.Sprintf("Test query %d", i%100)
			reqID := fmt.Sprintf("req%d", i)

			// Add entry
			_ = cache.AddEntry(reqID, "test-model", query, []byte(query), []byte("response"), -1)

			// Find similar
			_, _, _ = cache.FindSimilar("test-model", query)
		}
	})

	b.Run("HNSWSearch_AddAndFind", func(b *testing.B) {
		cache := NewInMemoryCache(InMemoryCacheOptions{
			Enabled:             true,
			MaxEntries:          10000,
			SimilarityThreshold: 0.85,
			TTLSeconds:          0,
			UseHNSW:             true,
			HNSWM:               16,
			HNSWEfConstruction:  200,
		})

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			query := fmt.Sprintf("Test query %d", i%100)
			reqID := fmt.Sprintf("req%d", i)

			// Add entry
			_ = cache.AddEntry(reqID, "test-model", query, []byte(query), []byte("response"), -1)

			// Find similar
			_, _, _ = cache.FindSimilar("test-model", query)
		}
	})
}

// BenchmarkHNSWRebuild benchmarks index rebuild performance
func BenchmarkHNSWRebuild(b *testing.B) {
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	sizes := []int{100, 500, 1000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Rebuild_%d_entries", size), func(b *testing.B) {
			// Create and populate cache
			cache := NewInMemoryCache(InMemoryCacheOptions{
				Enabled:             true,
				MaxEntries:          size * 2,
				SimilarityThreshold: 0.85,
				TTLSeconds:          0,
				UseHNSW:             true,
				HNSWM:               16,
				HNSWEfConstruction:  200,
			})

			// Populate with test data
			for i := 0; i < size; i++ {
				query := fmt.Sprintf("Query %d about machine learning", i)
				reqID := fmt.Sprintf("req%d", i)
				_ = cache.AddEntry(reqID, "test-model", query, []byte(query), []byte("response"), -1)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				cache.mu.Lock()
				cache.rebuildHNSWIndex()
				cache.mu.Unlock()
			}
		})
	}
}

func TestSearchLayerHeapManagement(t *testing.T) {
	t.Run("retains the closest neighbor when ef is saturated", func(t *testing.T) {
		// Regression fixture: with the previous max-heap candidates/min-heap results
		// mix, trimming to ef would evict the best element instead of the worst.
		queryEmbedding := []float32{1.0}

		entries := []CacheEntry{
			{Embedding: []float32{0.1}}, // entry point has low similarity
			{Embedding: []float32{1.0}}, // neighbor is the true nearest
		}

		entryNode := &HNSWNode{
			entryIndex: 0,
			neighbors: map[int][]int{
				0: {1},
			},
			maxLayer: 0,
		}

		neighborNode := &HNSWNode{
			entryIndex: 1,
			neighbors: map[int][]int{
				0: {0},
			},
			maxLayer: 0,
		}

		index := &HNSWIndex{
			nodes: []*HNSWNode{entryNode, neighborNode},
			nodeIndex: map[int]*HNSWNode{
				0: entryNode,
				1: neighborNode,
			},
			entryPoint:     0,
			maxLayer:       0,
			efConstruction: 2,
			M:              1,
			Mmax:           1,
			Mmax0:          2,
			ml:             1,
		}

		results := index.searchLayer(queryEmbedding, index.entryPoint, 1, 0, entries)

		if !slices.Contains(results, 1) {
			t.Fatalf("expected results to contain best neighbor 1, got %v", results)
		}
		if slices.Contains(results, 0) {
			t.Fatalf("expected results to drop entry point 0 once ef trimmed, got %v", results)
		}
	})

	t.Run("continues exploring even when next candidate looks worse", func(t *testing.T) {
		// Regression fixture: the break condition used the wrong polarity so the
		// search stopped before expanding the intermediate (worse) vertex, making
		// the actual best neighbor unreachable.
		queryEmbedding := []float32{1.0}

		entries := []CacheEntry{
			{Embedding: []float32{0.2}},  // entry point
			{Embedding: []float32{0.05}}, // intermediate node with poor similarity
			{Embedding: []float32{1.0}},  // hidden best match
		}

		entryNode := &HNSWNode{
			entryIndex: 0,
			neighbors: map[int][]int{
				0: {1},
			},
			maxLayer: 0,
		}

		intermediateNode := &HNSWNode{
			entryIndex: 1,
			neighbors: map[int][]int{
				0: {0, 2},
			},
			maxLayer: 0,
		}

		bestNode := &HNSWNode{
			entryIndex: 2,
			neighbors: map[int][]int{
				0: {1},
			},
			maxLayer: 0,
		}

		index := &HNSWIndex{
			nodes: []*HNSWNode{entryNode, intermediateNode, bestNode},
			nodeIndex: map[int]*HNSWNode{
				0: entryNode,
				1: intermediateNode,
				2: bestNode,
			},
			entryPoint:     0,
			maxLayer:       0,
			efConstruction: 2,
			M:              1,
			Mmax:           1,
			Mmax0:          2,
			ml:             1,
		}

		results := index.searchLayer(queryEmbedding, index.entryPoint, 2, 0, entries)

		if !slices.Contains(results, 2) {
			t.Fatalf("expected results to reach best neighbor 2 via intermediate node, got %v", results)
		}
	})
}

// BenchmarkLargeScale tests HNSW vs Linear at scales where HNSW shows advantages (10K-100K entries)
func BenchmarkLargeScale(b *testing.B) {
	// Initialize BERT model (GPU by default)
	useCPU := os.Getenv("USE_CPU") == "true"
	modelName := "sentence-transformers/all-MiniLM-L6-v2"
	if err := candle_binding.InitModel(modelName, useCPU); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	// Large scale cache sizes where HNSW shines
	cacheSizes := []int{10000, 50000, 100000}

	// Quick mode: only run 10K for fast demo
	if os.Getenv("BENCHMARK_QUICK") == "true" {
		cacheSizes = []int{10000}
	}

	// Use medium length queries for consistency
	contentLen := MediumContent

	// HNSW configurations
	// Only using default config since performance is similar across configs
	hnswConfigs := []struct {
		name string
		m    int
		ef   int
	}{
		{"HNSW_default", 16, 200},
	}

	// Open CSV file for results
	// Create benchmark_results directory if it doesn't exist
	resultsDir := "../../benchmark_results"
	if err := os.MkdirAll(resultsDir, 0o755); err != nil {
		b.Logf("Warning: Could not create results directory: %v", err)
	}

	csvFile, err := os.OpenFile(resultsDir+"/large_scale_benchmark.csv",
		os.O_APPEND|os.O_CREATE|os.O_WRONLY,
		0o644)
	if err != nil {
		b.Logf("Warning: Could not open CSV file: %v", err)
	} else {
		defer csvFile.Close()
		// Write header if file is new
		stat, _ := csvFile.Stat()
		if stat.Size() == 0 {
			header := "cache_size,search_method,hnsw_m,hnsw_ef,avg_latency_ns,iterations,speedup_vs_linear\n"
			if _, err := csvFile.WriteString(header); err != nil {
				b.Logf("Warning: failed to write CSV header: %v", err)
			}
		}
	}

	for _, cacheSize := range cacheSizes {
		b.Run(fmt.Sprintf("CacheSize_%d", cacheSize), func(b *testing.B) {
			// Generate test data
			b.Logf("Generating %d test queries...", cacheSize)
			testQueries := make([]string, cacheSize)
			for i := 0; i < cacheSize; i++ {
				testQueries[i] = generateQuery(contentLen, i)
			}

			// Generate query embeddings once
			useCPUStr := "CPU"
			if !useCPU {
				useCPUStr = "GPU"
			}
			b.Logf("Generating embeddings for %d queries using %s...", cacheSize, useCPUStr)
			testEmbeddings := make([][]float32, cacheSize)
			embStart := time.Now()
			embProgressInterval := cacheSize / 10
			if embProgressInterval < 1000 {
				embProgressInterval = 1000
			}

			for i := 0; i < cacheSize; i++ {
				emb, err := candle_binding.GetEmbedding(testQueries[i], 0)
				if err != nil {
					b.Fatalf("Failed to generate embedding: %v", err)
				}
				testEmbeddings[i] = emb

				// Progress indicator
				if (i+1)%embProgressInterval == 0 {
					elapsed := time.Since(embStart)
					embPerSec := float64(i+1) / elapsed.Seconds()
					remaining := time.Duration(float64(cacheSize-i-1) / embPerSec * float64(time.Second))
					b.Logf("  [Embeddings] %d/%d (%.0f%%, %.0f emb/sec, ~%v remaining)",
						i+1, cacheSize, float64(i+1)/float64(cacheSize)*100,
						embPerSec, remaining.Round(time.Second))
				}
			}
			b.Logf("✓ Generated %d embeddings in %v (%.0f emb/sec)",
				cacheSize, time.Since(embStart), float64(cacheSize)/time.Since(embStart).Seconds())

			// Test query (use a query similar to middle entries for realistic search)
			searchQuery := generateQuery(contentLen, cacheSize/2)

			var linearLatency float64

			// Benchmark Linear Search
			b.Run("Linear", func(b *testing.B) {
				b.Logf("=== Testing Linear Search with %d entries ===", cacheSize)
				cache := NewInMemoryCache(InMemoryCacheOptions{
					Enabled:             true,
					SimilarityThreshold: 0.8,
					MaxEntries:          cacheSize,
					UseHNSW:             false, // Linear search
				})

				// Populate cache
				b.Logf("Building cache with %d entries...", cacheSize)
				progressInterval := cacheSize / 10
				if progressInterval < 1000 {
					progressInterval = 1000
				}

				for i := 0; i < cacheSize; i++ {
					err := cache.AddEntry(
						fmt.Sprintf("req-%d", i),
						"test-model",
						testQueries[i],
						[]byte(fmt.Sprintf("request-%d", i)),
						[]byte(fmt.Sprintf("response-%d", i)),
						-1,
					)
					if err != nil {
						b.Fatalf("Failed to add entry: %v", err)
					}

					if (i+1)%progressInterval == 0 {
						b.Logf("  [Linear] Added %d/%d entries (%.0f%%)",
							i+1, cacheSize, float64(i+1)/float64(cacheSize)*100)
					}
				}
				b.Logf("✓ Linear cache built. Starting search benchmark...")

				// Run search benchmark
				b.ResetTimer()
				start := time.Now()
				for i := 0; i < b.N; i++ {
					_, _, err := cache.FindSimilar("test-model", searchQuery)
					if err != nil {
						b.Fatalf("FindSimilar failed: %v", err)
					}
				}
				b.StopTimer()

				linearLatency = float64(time.Since(start).Nanoseconds()) / float64(b.N)
				b.Logf("✓ Linear search complete: %.2f ms per query (%d iterations)",
					linearLatency/1e6, b.N)

				// Write to CSV
				if csvFile != nil {
					line := fmt.Sprintf("%d,linear,0,0,%.0f,%d,1.0\n",
						cacheSize, linearLatency, b.N)
					if _, err := csvFile.WriteString(line); err != nil {
						b.Logf("Warning: failed to write to CSV: %v", err)
					}
				}

				b.ReportMetric(linearLatency/1e6, "ms/op")
			})

			// Benchmark HNSW configurations
			for _, config := range hnswConfigs {
				b.Run(config.name, func(b *testing.B) {
					b.Logf("=== Testing %s with %d entries (M=%d, ef=%d) ===",
						config.name, cacheSize, config.m, config.ef)
					cache := NewInMemoryCache(InMemoryCacheOptions{
						Enabled:             true,
						SimilarityThreshold: 0.8,
						MaxEntries:          cacheSize,
						UseHNSW:             true,
						HNSWM:               config.m,
						HNSWEfConstruction:  config.ef,
					})

					// Populate cache
					b.Logf("Building HNSW index with %d entries (M=%d, ef=%d)...",
						cacheSize, config.m, config.ef)
					buildStart := time.Now()
					progressInterval := cacheSize / 10
					if progressInterval < 1000 {
						progressInterval = 1000
					}

					for i := 0; i < cacheSize; i++ {
						err := cache.AddEntry(
							fmt.Sprintf("req-%d", i),
							"test-model",
							testQueries[i],
							[]byte(fmt.Sprintf("request-%d", i)),
							[]byte(fmt.Sprintf("response-%d", i)),
							-1,
						)
						if err != nil {
							b.Fatalf("Failed to add entry: %v", err)
						}

						// Progress indicator
						if (i+1)%progressInterval == 0 {
							elapsed := time.Since(buildStart)
							entriesPerSec := float64(i+1) / elapsed.Seconds()
							remaining := time.Duration(float64(cacheSize-i-1) / entriesPerSec * float64(time.Second))
							b.Logf("  [%s] %d/%d entries (%.0f%%, %v elapsed, ~%v remaining, %.0f entries/sec)",
								config.name, i+1, cacheSize,
								float64(i+1)/float64(cacheSize)*100,
								elapsed.Round(time.Second),
								remaining.Round(time.Second),
								entriesPerSec)
						}
					}
					buildTime := time.Since(buildStart)
					b.Logf("✓ HNSW index built in %v (%.0f entries/sec)",
						buildTime, float64(cacheSize)/buildTime.Seconds())

					// Run search benchmark
					b.Logf("Starting search benchmark...")
					b.ResetTimer()
					start := time.Now()
					for i := 0; i < b.N; i++ {
						_, _, err := cache.FindSimilar("test-model", searchQuery)
						if err != nil {
							b.Fatalf("FindSimilar failed: %v", err)
						}
					}
					b.StopTimer()

					hnswLatency := float64(time.Since(start).Nanoseconds()) / float64(b.N)
					speedup := linearLatency / hnswLatency

					b.Logf("✓ HNSW search complete: %.2f ms per query (%d iterations)",
						hnswLatency/1e6, b.N)
					b.Logf("📊 SPEEDUP: %.1fx faster than linear search (%.2f ms vs %.2f ms)",
						speedup, hnswLatency/1e6, linearLatency/1e6)

					// Write to CSV
					if csvFile != nil {
						line := fmt.Sprintf("%d,%s,%d,%d,%.0f,%d,%.2f\n",
							cacheSize, config.name, config.m, config.ef,
							hnswLatency, b.N, speedup)
						if _, err := csvFile.WriteString(line); err != nil {
							b.Logf("Warning: failed to write to CSV: %v", err)
						}
					}

					b.ReportMetric(hnswLatency/1e6, "ms/op")
					b.ReportMetric(speedup, "speedup")
					b.ReportMetric(float64(buildTime.Milliseconds()), "build_ms")
				})
			}
		})
	}
}

// BenchmarkScalability tests how performance scales with cache size
func BenchmarkScalability(b *testing.B) {
	useCPU := os.Getenv("USE_CPU") == "true"
	modelName := "sentence-transformers/all-MiniLM-L6-v2"
	if err := candle_binding.InitModel(modelName, useCPU); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	// Test cache sizes from small to very large
	cacheSizes := []int{1000, 5000, 10000, 25000, 50000, 100000}

	// CSV output
	resultsDir := "../../benchmark_results"
	if err := os.MkdirAll(resultsDir, 0o755); err != nil {
		b.Logf("Warning: Could not create results directory: %v", err)
	}

	csvFile, err := os.OpenFile(resultsDir+"/scalability_benchmark.csv",
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		b.Logf("Warning: Could not open CSV file: %v", err)
	} else {
		defer csvFile.Close()
		stat, _ := csvFile.Stat()
		if stat.Size() == 0 {
			header := "cache_size,method,avg_latency_ns,latency_ms,ops_per_sec\n"
			if _, err := csvFile.WriteString(header); err != nil {
				b.Logf("Warning: failed to write CSV header: %v", err)
			}
		}
	}

	for _, cacheSize := range cacheSizes {
		// Skip linear search for very large sizes (too slow)
		testLinear := cacheSize <= 25000

		b.Run(fmt.Sprintf("Size_%d", cacheSize), func(b *testing.B) {
			// Generate test data
			testQueries := make([]string, cacheSize)
			for i := 0; i < cacheSize; i++ {
				testQueries[i] = generateQuery(MediumContent, i)
			}
			searchQuery := generateQuery(MediumContent, cacheSize/2)

			if testLinear {
				b.Run("Linear", func(b *testing.B) {
					cache := NewInMemoryCache(InMemoryCacheOptions{
						Enabled:             true,
						SimilarityThreshold: 0.8,
						MaxEntries:          cacheSize,
						UseHNSW:             false,
					})

					for i := 0; i < cacheSize; i++ {
						if err := cache.AddEntry(fmt.Sprintf("req-%d", i), "model",
							testQueries[i], []byte("req"), []byte("resp"), -1); err != nil {
							b.Fatalf("AddEntry failed: %v", err)
						}
					}

					b.ResetTimer()
					start := time.Now()
					for i := 0; i < b.N; i++ {
						if _, _, err := cache.FindSimilar("model", searchQuery); err != nil {
							b.Fatalf("FindSimilar failed: %v", err)
						}
					}
					elapsed := time.Since(start)

					avgLatency := float64(elapsed.Nanoseconds()) / float64(b.N)
					latencyMS := avgLatency / 1e6
					opsPerSec := float64(b.N) / elapsed.Seconds()

					if csvFile != nil {
						line := fmt.Sprintf("%d,linear,%.0f,%.3f,%.0f\n",
							cacheSize, avgLatency, latencyMS, opsPerSec)
						if _, err := csvFile.WriteString(line); err != nil {
							b.Logf("Warning: failed to write to CSV: %v", err)
						}
					}

					b.ReportMetric(latencyMS, "ms/op")
					b.ReportMetric(opsPerSec, "qps")
				})
			}

			b.Run("HNSW", func(b *testing.B) {
				cache := NewInMemoryCache(InMemoryCacheOptions{
					Enabled:             true,
					SimilarityThreshold: 0.8,
					MaxEntries:          cacheSize,
					UseHNSW:             true,
					HNSWM:               16,
					HNSWEfConstruction:  200,
				})

				buildStart := time.Now()
				for i := 0; i < cacheSize; i++ {
					if err := cache.AddEntry(fmt.Sprintf("req-%d", i), "model",
						testQueries[i], []byte("req"), []byte("resp"), -1); err != nil {
						b.Fatalf("AddEntry failed: %v", err)
					}
					if (i+1)%10000 == 0 {
						b.Logf("  Built %d/%d entries", i+1, cacheSize)
					}
				}
				b.Logf("HNSW build time: %v", time.Since(buildStart))

				b.ResetTimer()
				start := time.Now()
				for i := 0; i < b.N; i++ {
					if _, _, err := cache.FindSimilar("model", searchQuery); err != nil {
						b.Fatalf("FindSimilar failed: %v", err)
					}
				}
				elapsed := time.Since(start)

				avgLatency := float64(elapsed.Nanoseconds()) / float64(b.N)
				latencyMS := avgLatency / 1e6
				opsPerSec := float64(b.N) / elapsed.Seconds()

				if csvFile != nil {
					line := fmt.Sprintf("%d,hnsw,%.0f,%.3f,%.0f\n",
						cacheSize, avgLatency, latencyMS, opsPerSec)
					if _, err := csvFile.WriteString(line); err != nil {
						b.Logf("Warning: failed to write to CSV: %v", err)
					}
				}

				b.ReportMetric(latencyMS, "ms/op")
				b.ReportMetric(opsPerSec, "qps")
			})
		})
	}
}

// BenchmarkHNSWParameterSweep tests different HNSW parameters at large scale
func BenchmarkHNSWParameterSweep(b *testing.B) {
	useCPU := os.Getenv("USE_CPU") == "true"
	modelName := "sentence-transformers/all-MiniLM-L6-v2"
	if err := candle_binding.InitModel(modelName, useCPU); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	cacheSize := 50000 // 50K entries - good size to show differences

	// Parameter combinations to test
	// Test different M (connectivity) and efSearch (search quality) combinations
	// Fixed efConstruction=200 to focus on search-time performance
	configs := []struct {
		name     string
		m        int
		efSearch int
	}{
		// Low connectivity
		{"M8_efSearch10", 8, 10},
		{"M8_efSearch50", 8, 50},
		{"M8_efSearch100", 8, 100},
		{"M8_efSearch200", 8, 200},

		// Medium connectivity (recommended)
		{"M16_efSearch10", 16, 10},
		{"M16_efSearch50", 16, 50},
		{"M16_efSearch100", 16, 100},
		{"M16_efSearch200", 16, 200},
		{"M16_efSearch400", 16, 400},

		// High connectivity
		{"M32_efSearch50", 32, 50},
		{"M32_efSearch100", 32, 100},
		{"M32_efSearch200", 32, 200},
	}

	// Generate test data once
	b.Logf("Generating %d test queries...", cacheSize)
	testQueries := make([]string, cacheSize)
	for i := 0; i < cacheSize; i++ {
		testQueries[i] = generateQuery(MediumContent, i)
	}
	searchQuery := generateQuery(MediumContent, cacheSize/2)

	// CSV output
	resultsDir := "../../benchmark_results"
	if err := os.MkdirAll(resultsDir, 0o755); err != nil {
		b.Logf("Warning: Could not create results directory: %v", err)
	}

	csvFile, err := os.OpenFile(resultsDir+"/hnsw_parameter_sweep.csv",
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		b.Logf("Warning: Could not open CSV file: %v", err)
	} else {
		defer csvFile.Close()
		stat, _ := csvFile.Stat()
		if stat.Size() == 0 {
			header := "m,ef_search,build_time_ms,search_latency_ns,search_latency_ms,qps,memory_mb\n"
			if _, err := csvFile.WriteString(header); err != nil {
				b.Logf("Warning: failed to write CSV header: %v", err)
			}
		}
	}

	for _, config := range configs {
		b.Run(config.name, func(b *testing.B) {
			cache := NewInMemoryCache(InMemoryCacheOptions{
				Enabled:             true,
				SimilarityThreshold: 0.8,
				MaxEntries:          cacheSize,
				UseHNSW:             true,
				HNSWM:               config.m,
				HNSWEfConstruction:  200, // Fixed for consistent build quality
				HNSWEfSearch:        config.efSearch,
			})

			// Build index and measure time
			b.Logf("Building HNSW index: M=%d, efConstruction=200, efSearch=%d", config.m, config.efSearch)
			buildStart := time.Now()
			for i := 0; i < cacheSize; i++ {
				if err := cache.AddEntry(fmt.Sprintf("req-%d", i), "model",
					testQueries[i], []byte("req"), []byte("resp"), -1); err != nil {
					b.Fatalf("AddEntry failed: %v", err)
				}
				if (i+1)%10000 == 0 {
					b.Logf("  Progress: %d/%d", i+1, cacheSize)
				}
			}
			buildTime := time.Since(buildStart)

			// Estimate memory usage (rough)
			// Embeddings: cacheSize × 384 × 4 bytes
			// HNSW graph: cacheSize × M × 2 × 4 bytes (bidirectional links)
			embeddingMemMB := float64(cacheSize*384*4) / 1024 / 1024
			graphMemMB := float64(cacheSize*config.m*2*4) / 1024 / 1024
			totalMemMB := embeddingMemMB + graphMemMB

			b.Logf("Build time: %v, Est. memory: %.1f MB", buildTime, totalMemMB)

			// Benchmark search
			b.ResetTimer()
			start := time.Now()
			for i := 0; i < b.N; i++ {
				if _, _, err := cache.FindSimilar("model", searchQuery); err != nil {
					b.Fatalf("FindSimilar failed: %v", err)
				}
			}
			elapsed := time.Since(start)

			avgLatency := float64(elapsed.Nanoseconds()) / float64(b.N)
			latencyMS := avgLatency / 1e6
			qps := float64(b.N) / elapsed.Seconds()

			// Write to CSV
			if csvFile != nil {
				line := fmt.Sprintf("%d,%d,%.0f,%.0f,%.3f,%.0f,%.1f\n",
					config.m, config.efSearch, float64(buildTime.Milliseconds()),
					avgLatency, latencyMS, qps, totalMemMB)
				if _, err := csvFile.WriteString(line); err != nil {
					b.Logf("Warning: failed to write to CSV: %v", err)
				}
			}

			b.ReportMetric(latencyMS, "ms/op")
			b.ReportMetric(qps, "qps")
			b.ReportMetric(float64(buildTime.Milliseconds()), "build_ms")
			b.ReportMetric(totalMemMB, "memory_mb")
		})
	}
}

// Benchmark SIMD vs scalar dotProduct implementations
func BenchmarkDotProduct(b *testing.B) {
	// Test with different vector sizes
	sizes := []int{64, 128, 256, 384, 512, 768, 1024}

	for _, size := range sizes {
		// Generate random vectors
		a := make([]float32, size)
		vec_b := make([]float32, size)
		for i := 0; i < size; i++ {
			a[i] = rand.Float32()
			vec_b[i] = rand.Float32()
		}

		b.Run(fmt.Sprintf("SIMD/%d", size), func(b *testing.B) {
			b.ReportAllocs()
			var sum float32
			for i := 0; i < b.N; i++ {
				sum += dotProductSIMD(a, vec_b)
			}
			_ = sum
		})

		b.Run(fmt.Sprintf("Scalar/%d", size), func(b *testing.B) {
			b.ReportAllocs()
			var sum float32
			for i := 0; i < b.N; i++ {
				sum += dotProductScalar(a, vec_b)
			}
			_ = sum
		})
	}
}

// Test correctness of SIMD implementation
func TestDotProductSIMD(t *testing.T) {
	testCases := []struct {
		name string
		a    []float32
		b    []float32
		want float32
	}{
		{
			name: "empty",
			a:    []float32{},
			b:    []float32{},
			want: 0,
		},
		{
			name: "single element",
			a:    []float32{2.0},
			b:    []float32{3.0},
			want: 6.0,
		},
		{
			name: "short vector",
			a:    []float32{1, 2, 3},
			b:    []float32{4, 5, 6},
			want: 32.0, // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
		},
		{
			name: "8 elements (AVX2 boundary)",
			a:    []float32{1, 2, 3, 4, 5, 6, 7, 8},
			b:    []float32{1, 1, 1, 1, 1, 1, 1, 1},
			want: 36.0, // 1+2+3+4+5+6+7+8 = 36
		},
		{
			name: "16 elements (AVX-512 boundary)",
			a:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			b:    []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			want: 136.0, // 1+2+...+16 = 136
		},
		{
			name: "non-aligned size (17 elements)",
			a:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
			b:    []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			want: 153.0, // 1+2+...+17 = 153
		},
		{
			name: "384 dimensions (typical embedding size)",
			a:    make384Vector(),
			b:    ones(384),
			want: sum384(),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := dotProductSIMD(tc.a, tc.b)
			if abs(got-tc.want) > 0.0001 {
				t.Errorf("dotProductSIMD() = %v, want %v", got, tc.want)
			}

			// Also verify scalar produces same result
			scalar := dotProductScalar(tc.a, tc.b)
			if abs(scalar-tc.want) > 0.0001 {
				t.Errorf("dotProductScalar() = %v, want %v", scalar, tc.want)
			}

			// SIMD and scalar should match
			if abs(got-scalar) > 0.0001 {
				t.Errorf("SIMD (%v) != Scalar (%v)", got, scalar)
			}
		})
	}
}

func make384Vector() []float32 {
	v := make([]float32, 384)
	for i := range v {
		v[i] = float32(i + 1)
	}
	return v
}

func ones(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = 1.0
	}
	return v
}

func sum384() float32 {
	// Sum of 1+2+3+...+384 = 384 * 385 / 2 = 73920
	return 73920.0
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
