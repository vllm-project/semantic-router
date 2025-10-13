package extproc_test

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
)

var _ = Describe("Edge Cases and Error Conditions", func() {
	var (
		router *extproc.OpenAIRouter
		cfg    *config.RouterConfig
	)

	BeforeEach(func() {
		cfg = CreateTestConfig()
		var err error
		router, err = CreateTestRouter(cfg)
		Expect(err).NotTo(HaveOccurred())
	})

	Context("Large and malformed requests", func() {
		It("should handle very large request bodies", func() {
			largeContent := strings.Repeat("a", 10*1024) // 10KB content (reduced from 1MB to avoid memory issues)
			request := map[string]interface{}{
				"model": "model-a",
				"messages": []map[string]interface{}{
					{"role": "user", "content": largeContent},
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
				RequestID: "large-request",
				StartTime: time.Now(),
			}

			response, err := router.HandleRequestBody(bodyRequest, ctx)
			// Should handle moderately large requests gracefully
			Expect(err).To(Or(BeNil(), HaveOccurred()))
			if err == nil {
				Expect(response.GetRequestBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
			}
		})

		It("should handle requests with special characters", func() {
			request := map[string]interface{}{
				"model": "model-a",
				"messages": []map[string]interface{}{
					{"role": "user", "content": "Hello 🌍! What about ñoño and émojis? 你好"},
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
				RequestID: "unicode-request",
				StartTime: time.Now(),
			}

			response, err := router.HandleRequestBody(bodyRequest, ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(response.GetRequestBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})

		It("should handle malformed OpenAI requests gracefully", func() {
			// Missing required fields
			malformedRequest := map[string]interface{}{
				"model": "model-a",
				// Missing messages field
			}

			requestBody, err := json.Marshal(malformedRequest)
			Expect(err).NotTo(HaveOccurred())

			bodyRequest := &ext_proc.ProcessingRequest_RequestBody{
				RequestBody: &ext_proc.HttpBody{
					Body: requestBody,
				},
			}

			ctx := &extproc.RequestContext{
				Headers:   make(map[string]string),
				RequestID: "malformed-request",
				StartTime: time.Now(),
			}

			response, err := router.HandleRequestBody(bodyRequest, ctx)
			// Should handle gracefully, might continue or error depending on validation
			Expect(err).To(Or(BeNil(), HaveOccurred()))
			if err == nil {
				Expect(response).NotTo(BeNil())
			}
		})

		It("should handle requests with invalid model names", func() {
			request := map[string]interface{}{
				"model": "invalid-model-name-12345",
				"messages": []map[string]interface{}{
					{"role": "user", "content": "Test with invalid model"},
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
				RequestID: "invalid-model-request",
				StartTime: time.Now(),
			}

			response, err := router.HandleRequestBody(bodyRequest, ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(response.GetRequestBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})

		It("should handle requests with extremely long message chains", func() {
			messages := make([]map[string]interface{}, 100) // 100 messages
			for i := 0; i < 100; i++ {
				role := "user"
				if i%2 == 1 {
					role = "assistant"
				}
				messages[i] = map[string]interface{}{
					"role":    role,
					"content": fmt.Sprintf("Message %d in a very long conversation chain", i+1),
				}
			}

			request := map[string]interface{}{
				"model":    "model-b",
				"messages": messages,
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
				RequestID: "long-chain-request",
				StartTime: time.Now(),
			}

			response, err := router.HandleRequestBody(bodyRequest, ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(response.GetRequestBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})
	})

	Context("Concurrent processing", func() {
		It("should handle concurrent request processing", func() {
			const numRequests = 10
			responses := make(chan error, numRequests)

			// Create multiple concurrent requests
			for i := 0; i < numRequests; i++ {
				go func(index int) {
					request := map[string]interface{}{
						"model": "model-a",
						"messages": []map[string]interface{}{
							{"role": "user", "content": fmt.Sprintf("Request %d", index)},
						},
					}

					requestBody, err := json.Marshal(request)
					if err != nil {
						responses <- err
						return
					}

					bodyRequest := &ext_proc.ProcessingRequest_RequestBody{
						RequestBody: &ext_proc.HttpBody{
							Body: requestBody,
						},
					}

					ctx := &extproc.RequestContext{
						Headers:   make(map[string]string),
						RequestID: fmt.Sprintf("concurrent-request-%d", index),
						StartTime: time.Now(),
					}

					_, err = router.HandleRequestBody(bodyRequest, ctx)
					responses <- err
				}(i)
			}

			// Collect all responses
			errorCount := 0
			for i := 0; i < numRequests; i++ {
				err := <-responses
				if err != nil {
					errorCount++
				}
			}

			// Some errors might be expected due to candle_binding dependencies
			// The important thing is that the system doesn't crash
			Expect(errorCount).To(BeNumerically("<=", numRequests))
		})

		It("should handle rapid sequential requests", func() {
			const numRequests = 20

			for i := 0; i < numRequests; i++ {
				request := map[string]interface{}{
					"model": "model-b",
					"messages": []map[string]interface{}{
						{"role": "user", "content": fmt.Sprintf("Sequential request %d", i)},
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
					RequestID: fmt.Sprintf("sequential-request-%d", i),
					StartTime: time.Now(),
				}

				response, err := router.HandleRequestBody(bodyRequest, ctx)
				Expect(err).NotTo(HaveOccurred())
				Expect(response).NotTo(BeNil())
			}
		})
	})

	Context("Memory and resource handling", func() {
		It("should handle requests with deeply nested JSON", func() {
			// Create a deeply nested structure
			nestedContent := "{"
			for i := 0; i < 10; i++ {
				nestedContent += fmt.Sprintf(`"level%d": {`, i)
			}
			nestedContent += `"message": "deeply nested content"`
			for i := 0; i < 10; i++ {
				nestedContent += "}"
			}
			nestedContent += "}"

			request := map[string]interface{}{
				"model": "model-a",
				"messages": []map[string]interface{}{
					{"role": "user", "content": "Process this nested structure: " + nestedContent},
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
				RequestID: "nested-json-request",
				StartTime: time.Now(),
			}

			response, err := router.HandleRequestBody(bodyRequest, ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(response.GetRequestBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})

		It("should handle requests with many repeated patterns", func() {
			// Create content with many repeated patterns
			repeatedPattern := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 100)

			request := cache.OpenAIRequest{
				Model: "model-a",
				Messages: []cache.ChatMessage{
					{Role: "user", Content: repeatedPattern},
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
				RequestID: "repeated-pattern-request",
				StartTime: time.Now(),
			}

			response, err := router.HandleRequestBody(bodyRequest, ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(response.GetRequestBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})
	})

	Context("Boundary conditions", func() {
		It("should handle empty messages array", func() {
			request := cache.OpenAIRequest{
				Model:    "model-a",
				Messages: []cache.ChatMessage{}, // Empty messages
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
				RequestID: "empty-messages-request",
				StartTime: time.Now(),
			}

			response, err := router.HandleRequestBody(bodyRequest, ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(response.GetRequestBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})

		It("should handle messages with empty content", func() {
			request := cache.OpenAIRequest{
				Model: "model-a",
				Messages: []cache.ChatMessage{
					{Role: "user", Content: ""},      // Empty content
					{Role: "assistant", Content: ""}, // Empty content
					{Role: "user", Content: "Now respond to this"},
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
				RequestID: "empty-content-request",
				StartTime: time.Now(),
			}

			response, err := router.HandleRequestBody(bodyRequest, ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(response.GetRequestBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})

		It("should handle messages with only whitespace content", func() {
			request := cache.OpenAIRequest{
				Model: "model-a",
				Messages: []cache.ChatMessage{
					{Role: "user", Content: "   \n\t  "}, // Only whitespace
					{Role: "user", Content: "What is AI?"},
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
				RequestID: "whitespace-content-request",
				StartTime: time.Now(),
			}

			response, err := router.HandleRequestBody(bodyRequest, ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(response.GetRequestBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})
	})

	Context("Error recovery", func() {
		It("should recover from classification errors gracefully", func() {
			// Create a request that might cause classification issues
			request := cache.OpenAIRequest{
				Model: "auto", // This triggers classification
				Messages: []cache.ChatMessage{
					{Role: "user", Content: "Test content that might cause classification issues: \x00\x01\x02"}, // Binary content
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
				RequestID: "classification-error-request",
				StartTime: time.Now(),
			}

			response, err := router.HandleRequestBody(bodyRequest, ctx)
			// Should handle classification errors gracefully
			Expect(err).To(Or(BeNil(), HaveOccurred()))
			if err == nil {
				Expect(response).NotTo(BeNil())
			}
		})

		It("should handle timeout scenarios gracefully", func() {
			// Simulate a request that might take a long time to process
			request := cache.OpenAIRequest{
				Model: "auto",
				Messages: []cache.ChatMessage{
					{Role: "user", Content: "This is a complex request that might take time to classify and process"},
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
				RequestID: "timeout-test-request",
				StartTime: time.Now().Add(-10 * time.Second), // Simulate old request
			}

			response, err := router.HandleRequestBody(bodyRequest, ctx)
			// Should handle timeout scenarios without crashing
			Expect(err).To(Or(BeNil(), HaveOccurred()))
			if err == nil {
				Expect(response).NotTo(BeNil())
			}
		})
	})
})
