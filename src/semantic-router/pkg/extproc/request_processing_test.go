package extproc_test

import (
	"encoding/json"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

var _ = Describe("Request Processing", func() {
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

	Describe("handleRequestHeaders", func() {
		It("should process request headers successfully", func() {
			headers := &ext_proc.ProcessingRequest_RequestHeaders{
				RequestHeaders: &ext_proc.HttpHeaders{
					Headers: &core.HeaderMap{
						Headers: []*core.HeaderValue{
							{Key: "content-type", Value: "application/json"},
							{Key: "x-request-id", Value: "test-request-123"},
							{Key: "authorization", Value: "Bearer token"},
						},
					},
				},
			}

			ctx := &extproc.RequestContext{
				Headers: make(map[string]string),
			}

			response, err := router.HandleRequestHeaders(headers, ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(response).NotTo(BeNil())

			// Check that headers were stored
			Expect(ctx.Headers).To(HaveKeyWithValue("content-type", "application/json"))
			Expect(ctx.Headers).To(HaveKeyWithValue("x-request-id", "test-request-123"))
			Expect(ctx.RequestID).To(Equal("test-request-123"))

			// Check response status
			headerResp := response.GetRequestHeaders()
			Expect(headerResp).NotTo(BeNil())
			Expect(headerResp.Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})

		It("should handle missing x-request-id header", func() {
			headers := &ext_proc.ProcessingRequest_RequestHeaders{
				RequestHeaders: &ext_proc.HttpHeaders{
					Headers: &core.HeaderMap{
						Headers: []*core.HeaderValue{
							{Key: "content-type", Value: "application/json"},
						},
					},
				},
			}

			ctx := &extproc.RequestContext{
				Headers: make(map[string]string),
			}

			response, err := router.HandleRequestHeaders(headers, ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(ctx.RequestID).To(BeEmpty())
			Expect(response.GetRequestHeaders().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})

		It("should handle case-insensitive header matching", func() {
			headers := &ext_proc.ProcessingRequest_RequestHeaders{
				RequestHeaders: &ext_proc.HttpHeaders{
					Headers: &core.HeaderMap{
						Headers: []*core.HeaderValue{
							{Key: "X-Request-ID", Value: "test-case-insensitive"},
						},
					},
				},
			}

			ctx := &extproc.RequestContext{
				Headers: make(map[string]string),
			}

			_, err := router.HandleRequestHeaders(headers, ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(ctx.RequestID).To(Equal("test-case-insensitive"))
		})
	})

	Describe("handleRequestBody", func() {
		Context("with valid OpenAI request", func() {
			It("should process auto model routing successfully", func() {
				request := cache.OpenAIRequest{
					Model: "auto",
					Messages: []cache.ChatMessage{
						{Role: "user", Content: "Write a Python function to sort a list"},
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
					RequestID: "test-request",
					StartTime: time.Now(),
				}

				response, err := router.HandleRequestBody(bodyRequest, ctx)
				Expect(err).NotTo(HaveOccurred())
				Expect(response).NotTo(BeNil())

				// Should continue processing
				bodyResp := response.GetRequestBody()
				Expect(bodyResp).NotTo(BeNil())
				Expect(bodyResp.Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
			})

			It("should handle non-auto model without modification", func() {
				request := cache.OpenAIRequest{
					Model: "model-a",
					Messages: []cache.ChatMessage{
						{Role: "user", Content: "Hello world"},
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
					RequestID: "test-request",
					StartTime: time.Now(),
				}

				response, err := router.HandleRequestBody(bodyRequest, ctx)
				Expect(err).NotTo(HaveOccurred())

				bodyResp := response.GetRequestBody()
				Expect(bodyResp.Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
			})

			It("should handle empty user content", func() {
				request := cache.OpenAIRequest{
					Model: "auto",
					Messages: []cache.ChatMessage{
						{Role: "system", Content: "You are a helpful assistant"},
						{Role: "assistant", Content: "Hello! How can I help you?"},
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
					RequestID: "test-request",
					StartTime: time.Now(),
				}

				response, err := router.HandleRequestBody(bodyRequest, ctx)
				Expect(err).NotTo(HaveOccurred())
				Expect(response.GetRequestBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
			})
		})

		Context("with invalid request body", func() {
			It("should return error for malformed JSON", func() {
				bodyRequest := &ext_proc.ProcessingRequest_RequestBody{
					RequestBody: &ext_proc.HttpBody{
						Body: []byte(`{"model": "model-a", "messages": [invalid json}`),
					},
				}

				ctx := &extproc.RequestContext{
					Headers:   make(map[string]string),
					RequestID: "test-request",
					StartTime: time.Now(),
				}

				response, err := router.HandleRequestBody(bodyRequest, ctx)
				Expect(err).To(HaveOccurred())
				Expect(response).To(BeNil())
				Expect(err.Error()).To(ContainSubstring("invalid request body"))
			})

			It("should handle empty request body", func() {
				bodyRequest := &ext_proc.ProcessingRequest_RequestBody{
					RequestBody: &ext_proc.HttpBody{
						Body: []byte{},
					},
				}

				ctx := &extproc.RequestContext{
					Headers:   make(map[string]string),
					RequestID: "test-request",
					StartTime: time.Now(),
				}

				response, err := router.HandleRequestBody(bodyRequest, ctx)
				Expect(err).To(HaveOccurred())
				Expect(response).To(BeNil())
			})

			It("should handle nil request body", func() {
				bodyRequest := &ext_proc.ProcessingRequest_RequestBody{
					RequestBody: &ext_proc.HttpBody{
						Body: nil,
					},
				}

				ctx := &extproc.RequestContext{
					Headers:   make(map[string]string),
					RequestID: "test-request",
					StartTime: time.Now(),
				}

				response, err := router.HandleRequestBody(bodyRequest, ctx)
				Expect(err).To(HaveOccurred())
				Expect(response).To(BeNil())
			})
		})

		Context("with tools auto-selection", func() {
			BeforeEach(func() {
				cfg.Tools.Enabled = true
				router.ToolsDatabase = tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
					Enabled: true,
				})
			})

			It("should handle tools auto-selection", func() {
				request := map[string]interface{}{
					"model": "model-a",
					"messages": []map[string]interface{}{
						{"role": "user", "content": "Calculate the square root of 16"},
					},
					"tools": "auto",
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
					RequestID: "test-request",
					StartTime: time.Now(),
				}

				response, err := router.HandleRequestBody(bodyRequest, ctx)
				Expect(err).NotTo(HaveOccurred())

				// Should process successfully even if tools selection fails
				bodyResp := response.GetRequestBody()
				Expect(bodyResp.Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
			})

			It("should fallback to empty tools on error", func() {
				cfg.Tools.FallbackToEmpty = true

				request := map[string]interface{}{
					"model": "model-a",
					"messages": []map[string]interface{}{
						{"role": "user", "content": "Test query"},
					},
					"tools": "auto",
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
					RequestID: "test-request",
					StartTime: time.Now(),
				}

				response, err := router.HandleRequestBody(bodyRequest, ctx)
				Expect(err).NotTo(HaveOccurred())
				Expect(response.GetRequestBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
			})
		})
	})

	Describe("handleResponseHeaders", func() {
		It("should process response headers successfully", func() {
			responseHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
				ResponseHeaders: &ext_proc.HttpHeaders{
					Headers: &core.HeaderMap{
						Headers: []*core.HeaderValue{
							{Key: "content-type", Value: "application/json"},
							{Key: "x-response-id", Value: "resp-123"},
						},
					},
				},
			}

			ctx := &extproc.RequestContext{
				Headers:             make(map[string]string),
				RequestModel:        "model-a",
				ProcessingStartTime: time.Now().Add(-50 * time.Millisecond),
			}

			response, err := router.HandleResponseHeaders(responseHeaders, ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(response).NotTo(BeNil())

			respHeaders := response.GetResponseHeaders()
			Expect(respHeaders).NotTo(BeNil())
			Expect(respHeaders.Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})
	})

	Describe("handleResponseBody", func() {
		It("should process response body with token parsing", func() {
			openAIResponse := openai.ChatCompletion{
				ID:      "chatcmpl-123",
				Object:  "chat.completion",
				Created: time.Now().Unix(),
				Model:   "model-a",
				Usage: openai.CompletionUsage{
					PromptTokens:     150,
					CompletionTokens: 50,
					TotalTokens:      200,
				},
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    "assistant",
							Content: "This is a test response",
						},
						FinishReason: "stop",
					},
				},
			}

			responseBody, err := json.Marshal(openAIResponse)
			Expect(err).NotTo(HaveOccurred())

			bodyResponse := &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body: responseBody,
				},
			}

			ctx := &extproc.RequestContext{
				Headers:      make(map[string]string),
				RequestID:    "test-request",
				RequestModel: "model-a",
				RequestQuery: "test query",
				StartTime:    time.Now().Add(-2 * time.Second),
			}

			response, err := router.HandleResponseBody(bodyResponse, ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(response).NotTo(BeNil())

			respBody := response.GetResponseBody()
			Expect(respBody).NotTo(BeNil())
			Expect(respBody.Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})

		It("should handle invalid response JSON gracefully", func() {
			bodyResponse := &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body: []byte(`{invalid json}`),
				},
			}

			ctx := &extproc.RequestContext{
				Headers:      make(map[string]string),
				RequestID:    "test-request",
				RequestModel: "model-a",
				StartTime:    time.Now(),
			}

			response, err := router.HandleResponseBody(bodyResponse, ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(response.GetResponseBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})

		It("should handle empty response body", func() {
			bodyResponse := &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body: nil,
				},
			}

			ctx := &extproc.RequestContext{
				Headers:   make(map[string]string),
				RequestID: "test-request",
				StartTime: time.Now(),
			}

			response, err := router.HandleResponseBody(bodyResponse, ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(response.GetResponseBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})

		Context("with category-specific system prompt", func() {
			BeforeEach(func() {
				// Add a category with system prompt to the config
				cfg.Categories = append(cfg.Categories, config.Category{
					Name:         "math",
					Description:  "Mathematical queries and calculations",
					SystemPrompt: "You are a helpful assistant specialized in mathematics. Please provide step-by-step solutions.",
					ModelScores: []config.ModelScore{
						{Model: "model-a", Score: 0.9, UseReasoning: config.BoolPtr(false)},
					},
				})

				// Recreate router with updated config
				var err error
				router, err = CreateTestRouter(cfg)
				Expect(err).NotTo(HaveOccurred())
			})

			It("should add category-specific system prompt to auto model requests", func() {
				request := cache.OpenAIRequest{
					Model: "auto",
					Messages: []cache.ChatMessage{
						{Role: "user", Content: "What is the derivative of x^2 + 3x + 1?"},
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
					RequestID: "system-prompt-test-request",
					StartTime: time.Now(),
				}

				response, err := router.HandleRequestBody(bodyRequest, ctx)
				Expect(err).NotTo(HaveOccurred())

				bodyResp := response.GetRequestBody()
				Expect(bodyResp.Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))

				// Check if the request body was modified with system prompt
				if bodyResp.Response.BodyMutation != nil {
					modifiedBody := bodyResp.Response.BodyMutation.GetBody()
					Expect(modifiedBody).NotTo(BeNil())

					var modifiedRequest map[string]interface{}
					err = json.Unmarshal(modifiedBody, &modifiedRequest)
					Expect(err).NotTo(HaveOccurred())

					messages, ok := modifiedRequest["messages"].([]interface{})
					Expect(ok).To(BeTrue())
					Expect(len(messages)).To(BeNumerically(">=", 2))

					// Check that system message was added
					firstMessage, ok := messages[0].(map[string]interface{})
					Expect(ok).To(BeTrue())
					Expect(firstMessage["role"]).To(Equal("system"))
					Expect(firstMessage["content"]).To(ContainSubstring("mathematics"))
					Expect(firstMessage["content"]).To(ContainSubstring("step-by-step"))
				}
			})

			It("should replace existing system prompt with category-specific one", func() {
				request := cache.OpenAIRequest{
					Model: "auto",
					Messages: []cache.ChatMessage{
						{Role: "system", Content: "You are a general assistant."},
						{Role: "user", Content: "Solve the equation 2x + 5 = 15"},
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
					RequestID: "system-prompt-replace-test-request",
					StartTime: time.Now(),
				}

				response, err := router.HandleRequestBody(bodyRequest, ctx)
				Expect(err).NotTo(HaveOccurred())

				bodyResp := response.GetRequestBody()
				Expect(bodyResp.Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))

				// Check if the request body was modified with system prompt
				if bodyResp.Response.BodyMutation != nil {
					modifiedBody := bodyResp.Response.BodyMutation.GetBody()
					Expect(modifiedBody).NotTo(BeNil())

					var modifiedRequest map[string]interface{}
					err = json.Unmarshal(modifiedBody, &modifiedRequest)
					Expect(err).NotTo(HaveOccurred())

					messages, ok := modifiedRequest["messages"].([]interface{})
					Expect(ok).To(BeTrue())
					Expect(len(messages)).To(Equal(2))

					// Check that system message was replaced
					firstMessage, ok := messages[0].(map[string]interface{})
					Expect(ok).To(BeTrue())
					Expect(firstMessage["role"]).To(Equal("system"))
					Expect(firstMessage["content"]).To(ContainSubstring("mathematics"))
					Expect(firstMessage["content"]).NotTo(ContainSubstring("general assistant"))
				}
			})
		})
	})
})
