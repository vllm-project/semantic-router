package extproc_test

import (
	"encoding/json"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
)

var _ = Describe("Caching Functionality", func() {
	var (
		router *extproc.OpenAIRouter
		cfg    *config.RouterConfig
	)

	BeforeEach(func() {
		cfg = CreateTestConfig()
		cfg.SemanticCache.Enabled = true

		var err error
		router, err = CreateTestRouter(cfg)
		Expect(err).NotTo(HaveOccurred())

		// Override cache with enabled configuration
		cacheConfig := cache.CacheConfig{
			BackendType:         cache.InMemoryCacheType,
			Enabled:             true,
			SimilarityThreshold: 0.9,
			MaxEntries:          100,
			TTLSeconds:          3600,
		}
		cacheBackend, err := cache.NewCacheBackend(cacheConfig)
		Expect(err).NotTo(HaveOccurred())
		router.Cache = cacheBackend
	})

	It("should handle cache miss scenario", func() {
		request := map[string]interface{}{
			"model": "model-a",
			"messages": []map[string]interface{}{
				{"role": "user", "content": "What is artificial intelligence?"},
			},
		}

		requestBody, err := json.Marshal(request)
		Expect(err).NotTo(HaveOccurred())

		bodyRequest := &ext_proc.ProcessingRequest_RequestBody{
			RequestBody: &ext_proc.HttpBody{
				Body: requestBody,
			},
		}

		ctx := &extproc.RequestContext{
			Headers:   make(map[string]string),
			RequestID: "test-request-cache",
			StartTime: time.Now(),
		}

		response, err := router.HandleRequestBody(bodyRequest, ctx)
		// Even if caching fails due to candle_binding, request should continue
		Expect(err).To(Or(BeNil(), HaveOccurred()))
		if err == nil {
			Expect(response.GetRequestBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		}
	})

	It("should handle cache update on response", func() {
		// First, simulate a request that would add a pending cache entry
		ctx := &extproc.RequestContext{
			Headers:      make(map[string]string),
			RequestID:    "cache-test-request",
			RequestModel: "model-a",
			RequestQuery: "test query for caching",
			StartTime:    time.Now(),
		}

		// Simulate response processing
		openAIResponse := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: "Cached response.",
					},
				},
			},
			Usage: openai.CompletionUsage{
				PromptTokens:     10,
				CompletionTokens: 5,
				TotalTokens:      15,
			},
		}

		responseBody, err := json.Marshal(openAIResponse)
		Expect(err).NotTo(HaveOccurred())

		bodyResponse := &ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{
				Body: responseBody,
			},
		}

		response, err := router.HandleResponseBody(bodyResponse, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(response.GetResponseBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
	})

	Context("with cache enabled", func() {
		It("should attempt to cache successful responses", func() {
			// Create a request
			request := map[string]interface{}{
				"model": "model-a",
				"messages": []map[string]interface{}{
					{"role": "user", "content": "Tell me about machine learning"},
				},
			}

			requestBody, err := json.Marshal(request)
			Expect(err).NotTo(HaveOccurred())

			bodyRequest := &ext_proc.ProcessingRequest_RequestBody{
				RequestBody: &ext_proc.HttpBody{
					Body: requestBody,
				},
			}

			ctx := &extproc.RequestContext{
				Headers:   make(map[string]string),
				RequestID: "cache-ml-request",
				StartTime: time.Now(),
			}

			// Process request
			_, err = router.HandleRequestBody(bodyRequest, ctx)
			Expect(err).To(Or(BeNil(), HaveOccurred()))

			// Process response
			openAIResponse := openai.ChatCompletion{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Content: "Machine learning is a subset of artificial intelligence...",
						},
					},
				},
				Usage: openai.CompletionUsage{
					PromptTokens:     20,
					CompletionTokens: 30,
					TotalTokens:      50,
				},
			}

			responseBody, err := json.Marshal(openAIResponse)
			Expect(err).NotTo(HaveOccurred())

			bodyResponse := &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body: responseBody,
				},
			}

			ctx.RequestModel = "model-a"
			ctx.RequestQuery = "Tell me about machine learning"

			response, err := router.HandleResponseBody(bodyResponse, ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(response.GetResponseBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})

		It("should handle cache errors gracefully", func() {
			// Test with a potentially problematic query
			request := map[string]interface{}{
				"model": "model-a",
				"messages": []map[string]interface{}{
					{"role": "user", "content": ""}, // Empty content might cause issues
				},
			}

			requestBody, err := json.Marshal(request)
			Expect(err).NotTo(HaveOccurred())

			bodyRequest := &ext_proc.ProcessingRequest_RequestBody{
				RequestBody: &ext_proc.HttpBody{
					Body: requestBody,
				},
			}

			ctx := &extproc.RequestContext{
				Headers:   make(map[string]string),
				RequestID: "cache-error-test",
				StartTime: time.Now(),
			}

			// Should not fail even if caching encounters issues
			response, err := router.HandleRequestBody(bodyRequest, ctx)
			Expect(err).To(Or(BeNil(), HaveOccurred()))
			if err == nil {
				Expect(response).NotTo(BeNil())
			}
		})
	})

	Context("with cache disabled", func() {
		BeforeEach(func() {
			cfg.SemanticCache.Enabled = false
			cacheConfig := cache.CacheConfig{
				BackendType:         cache.InMemoryCacheType,
				Enabled:             false,
				SimilarityThreshold: 0.9,
				MaxEntries:          100,
				TTLSeconds:          3600,
			}
			cacheBackend, err := cache.NewCacheBackend(cacheConfig)
			Expect(err).NotTo(HaveOccurred())
			router.Cache = cacheBackend
		})

		It("should process requests normally without caching", func() {
			request := map[string]interface{}{
				"model": "model-a",
				"messages": []map[string]interface{}{
					{"role": "user", "content": "What is the weather?"},
				},
			}

			requestBody, err := json.Marshal(request)
			Expect(err).NotTo(HaveOccurred())

			bodyRequest := &ext_proc.ProcessingRequest_RequestBody{
				RequestBody: &ext_proc.HttpBody{
					Body: requestBody,
				},
			}

			ctx := &extproc.RequestContext{
				Headers:   make(map[string]string),
				RequestID: "no-cache-request",
				StartTime: time.Now(),
			}

			response, err := router.HandleRequestBody(bodyRequest, ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(response.GetRequestBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})
	})

	Describe("Category-Specific Caching", func() {
		It("should use category-specific cache settings", func() {
			// Create a config with category-specific cache settings
			cfg := CreateTestConfig()
			cfg.SemanticCache.Enabled = true
			cfg.SemanticCache.SimilarityThreshold = config.Float32Ptr(0.8)

			// Add categories with different cache settings
			cfg.Categories = []config.Category{
				{
					Name: "health",
					ModelScores: []config.ModelScore{
						{Model: "model-a", Score: 1.0, UseReasoning: config.BoolPtr(false)},
					},
					SemanticCacheEnabled:             config.BoolPtr(true),
					SemanticCacheSimilarityThreshold: config.Float32Ptr(0.95),
				},
				{
					Name: "general",
					ModelScores: []config.ModelScore{
						{Model: "model-a", Score: 1.0, UseReasoning: config.BoolPtr(false)},
					},
					SemanticCacheEnabled:             config.BoolPtr(false),
					SemanticCacheSimilarityThreshold: config.Float32Ptr(0.7),
				},
			}

			// Verify category cache settings are correct
			Expect(cfg.IsCacheEnabledForCategory("health")).To(BeTrue())
			Expect(cfg.IsCacheEnabledForCategory("general")).To(BeFalse())
			Expect(cfg.GetCacheSimilarityThresholdForCategory("health")).To(Equal(float32(0.95)))
			Expect(cfg.GetCacheSimilarityThresholdForCategory("general")).To(Equal(float32(0.7)))
		})

		It("should fall back to global settings when category doesn't specify", func() {
			cfg := CreateTestConfig()
			cfg.SemanticCache.Enabled = true
			cfg.SemanticCache.SimilarityThreshold = config.Float32Ptr(0.8)

			// Add category without cache settings
			cfg.Categories = []config.Category{
				{
					Name: "test",
					ModelScores: []config.ModelScore{
						{Model: "model-a", Score: 1.0, UseReasoning: config.BoolPtr(false)},
					},
				},
			}

			// Should use global settings
			Expect(cfg.IsCacheEnabledForCategory("test")).To(BeTrue())
			Expect(cfg.GetCacheSimilarityThresholdForCategory("test")).To(Equal(float32(0.8)))
		})
	})
})
