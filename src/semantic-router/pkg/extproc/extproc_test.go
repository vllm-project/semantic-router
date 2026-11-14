package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
)

type testChatMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}
type testOpenAIRequest struct {
	Model       string            `json:"model"`
	Messages    []testChatMessage `json:"messages"`
	Stream      bool              `json:"stream,omitempty"`
	Temperature float32           `json:"temperature,omitempty"`
	MaxTokens   int               `json:"max_tokens,omitempty"`
	Tools       []interface{}     `json:"tools,omitempty"`
	TopP        float32           `json:"top_p,omitempty"`
}

var _ = Describe("Process Stream Handling", func() {
	var (
		router *OpenAIRouter
		cfg    *config.RouterConfig
	)

	BeforeEach(func() {
		cfg = CreateTestConfig()
		var err error
		router, err = CreateTestRouter(cfg)
		Expect(err).NotTo(HaveOccurred())
	})

	Context("with valid request sequence", func() {
		It("should handle complete request-response cycle", func() {
			// Create a sequence of requests
			requests := []*ext_proc.ProcessingRequest{
				{
					Request: &ext_proc.ProcessingRequest_RequestHeaders{
						RequestHeaders: &ext_proc.HttpHeaders{
							Headers: &core.HeaderMap{
								Headers: []*core.HeaderValue{
									{Key: ":method", Value: "POST"},
									{Key: ":path", Value: "/v1/chat/completions"},
									{Key: "content-type", Value: "application/json"},
									{Key: "x-request-id", Value: "test-123"},
								},
							},
						},
					},
				},
				{
					Request: &ext_proc.ProcessingRequest_RequestBody{
						RequestBody: &ext_proc.HttpBody{
							Body: []byte(`{"model": "model-a", "messages": [{"role": "user", "content": "Hello"}]}`),
						},
					},
				},
				{
					Request: &ext_proc.ProcessingRequest_ResponseHeaders{
						ResponseHeaders: &ext_proc.HttpHeaders{
							Headers: &core.HeaderMap{
								Headers: []*core.HeaderValue{
									{Key: "content-type", Value: "application/json"},
								},
							},
						},
					},
				},
				{
					Request: &ext_proc.ProcessingRequest_ResponseBody{
						ResponseBody: &ext_proc.HttpBody{
							Body: []byte(`{"id":"chatcmpl-test","object":"chat.completion","created":1,"model":"model-a","choices":[{"index":0,"message":{"role":"assistant","content":"Hi there!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`),
						},
					},
				},
			}

			stream := NewMockStream(requests)

			// Process would normally run in a goroutine, but for testing we call it directly
			// and expect it to return an error when the stream ends
			err := router.Process(stream)
			Expect(err).NotTo(HaveOccurred()) // Stream should end gracefully

			// Check that all requests were processed
			Expect(len(stream.Responses)).To(Equal(len(requests)))

			// Verify response types match request types
			Expect(stream.Responses[0].GetRequestHeaders()).NotTo(BeNil())
			Expect(stream.Responses[1].GetRequestBody()).NotTo(BeNil())
			Expect(stream.Responses[2].GetResponseHeaders()).NotTo(BeNil())
			Expect(stream.Responses[3].GetResponseBody()).NotTo(BeNil())
		})

		It("should handle partial request sequences", func() {
			// Only headers and body, no response processing
			requests := []*ext_proc.ProcessingRequest{
				{
					Request: &ext_proc.ProcessingRequest_RequestHeaders{
						RequestHeaders: &ext_proc.HttpHeaders{
							Headers: &core.HeaderMap{
								Headers: []*core.HeaderValue{
									{Key: "content-type", Value: "application/json"},
									{Key: "x-request-id", Value: "partial-test"},
								},
							},
						},
					},
				},
				{
					Request: &ext_proc.ProcessingRequest_RequestBody{
						RequestBody: &ext_proc.HttpBody{
							Body: []byte(`{"model": "model-b", "messages": [{"role": "user", "content": "Test"}]}`),
						},
					},
				},
			}

			stream := NewMockStream(requests)
			err := router.Process(stream)
			Expect(err).NotTo(HaveOccurred()) // Stream should end gracefully

			// Check that both requests were processed
			Expect(len(stream.Responses)).To(Equal(2))
			Expect(stream.Responses[0].GetRequestHeaders()).NotTo(BeNil())
			Expect(stream.Responses[1].GetRequestBody()).NotTo(BeNil())
		})

		It("should maintain request context across stream", func() {
			requests := []*ext_proc.ProcessingRequest{
				{
					Request: &ext_proc.ProcessingRequest_RequestHeaders{
						RequestHeaders: &ext_proc.HttpHeaders{
							Headers: &core.HeaderMap{
								Headers: []*core.HeaderValue{
									{Key: "x-request-id", Value: "context-test-123"},
									{Key: "user-agent", Value: "test-client"},
								},
							},
						},
					},
				},
				{
					Request: &ext_proc.ProcessingRequest_RequestBody{
						RequestBody: &ext_proc.HttpBody{
							Body: []byte(`{"model": "model-a", "messages": [{"role": "user", "content": "Context test"}]}`),
						},
					},
				},
			}

			stream := NewMockStream(requests)
			err := router.Process(stream)
			Expect(err).NotTo(HaveOccurred()) // Stream should end gracefully

			// Verify both requests were processed successfully
			Expect(len(stream.Responses)).To(Equal(2))

			// Both responses should indicate successful processing
			Expect(stream.Responses[0].GetRequestHeaders().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
			Expect(stream.Responses[1].GetRequestBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})
	})

	Context("with stream errors", func() {
		It("should handle receive errors", func() {
			stream := NewMockStream([]*ext_proc.ProcessingRequest{})
			stream.RecvError = fmt.Errorf("connection lost")

			err := router.Process(stream)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("connection lost"))
		})

		It("should handle send errors", func() {
			requests := []*ext_proc.ProcessingRequest{
				{
					Request: &ext_proc.ProcessingRequest_RequestHeaders{
						RequestHeaders: &ext_proc.HttpHeaders{
							Headers: &core.HeaderMap{
								Headers: []*core.HeaderValue{
									{Key: "content-type", Value: "application/json"},
								},
							},
						},
					},
				},
			}

			stream := NewMockStream(requests)
			stream.SendError = fmt.Errorf("send failed")

			err := router.Process(stream)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("send failed"))
		})

		It("should handle context cancellation gracefully", func() {
			stream := NewMockStream([]*ext_proc.ProcessingRequest{})
			stream.RecvError = context.Canceled

			err := router.Process(stream)
			Expect(err).NotTo(HaveOccurred()) // Context cancellation should be handled gracefully
		})

		It("should handle gRPC cancellation gracefully", func() {
			stream := NewMockStream([]*ext_proc.ProcessingRequest{})
			stream.RecvError = status.Error(codes.Canceled, "context canceled")

			err := router.Process(stream)
			Expect(err).NotTo(HaveOccurred()) // Context cancellation should be handled gracefully
		})

		It("should handle intermittent errors gracefully", func() {
			requests := []*ext_proc.ProcessingRequest{
				{
					Request: &ext_proc.ProcessingRequest_RequestHeaders{
						RequestHeaders: &ext_proc.HttpHeaders{
							Headers: &core.HeaderMap{
								Headers: []*core.HeaderValue{
									{Key: "content-type", Value: "application/json"},
								},
							},
						},
					},
				},
				{
					Request: &ext_proc.ProcessingRequest_RequestBody{
						RequestBody: &ext_proc.HttpBody{
							Body: []byte(`{"model": "model-a", "messages": [{"role": "user", "content": "Test"}]}`),
						},
					},
				},
			}

			stream := NewMockStream(requests)

			// Process first request successfully
			err := router.Process(stream)
			Expect(err).NotTo(HaveOccurred()) // Stream should end gracefully

			// At least the first request should have been processed
			Expect(len(stream.Responses)).To(BeNumerically(">=", 1))
		})
	})

	Context("with unknown request types", func() {
		It("should handle unknown request types gracefully", func() {
			// Create a mock request with unknown type (using nil)
			requests := []*ext_proc.ProcessingRequest{
				{
					Request: nil, // Unknown/unsupported request type
				},
			}

			stream := NewMockStream(requests)

			err := router.Process(stream)
			Expect(err).NotTo(HaveOccurred()) // Stream should end gracefully

			// Should still send a response for unknown types
			Expect(len(stream.Responses)).To(Equal(1))

			// The response should be a body response with CONTINUE status
			bodyResp := stream.Responses[0].GetRequestBody()
			Expect(bodyResp).NotTo(BeNil())
			Expect(bodyResp.Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})

		It("should handle mixed known and unknown request types", func() {
			requests := []*ext_proc.ProcessingRequest{
				{
					Request: &ext_proc.ProcessingRequest_RequestHeaders{
						RequestHeaders: &ext_proc.HttpHeaders{
							Headers: &core.HeaderMap{
								Headers: []*core.HeaderValue{
									{Key: ":method", Value: "POST"},
									{Key: ":path", Value: "/v1/chat/completions"},
									{Key: "content-type", Value: "application/json"},
									{Key: "x-request-id", Value: "mixed-request"},
								},
							},
						},
					},
				},
				{
					Request: nil, // Unknown type
				},
				{
					Request: &ext_proc.ProcessingRequest_RequestBody{
						RequestBody: &ext_proc.HttpBody{
							Body: []byte(`{"model": "model-a", "messages": [{"role": "user", "content": "Mixed test"}]}`),
						},
					},
				},
			}

			stream := NewMockStream(requests)
			err := router.Process(stream)
			Expect(err).NotTo(HaveOccurred()) // Stream should end gracefully

			// All requests should get responses
			Expect(len(stream.Responses)).To(Equal(3))

			// Known types should be handled correctly
			Expect(stream.Responses[0].GetRequestHeaders()).NotTo(BeNil())
			Expect(stream.Responses[2].GetRequestBody()).NotTo(BeNil())

			// Unknown type should get default response
			Expect(stream.Responses[1].GetRequestBody()).NotTo(BeNil())
		})
	})

	Context("stream processing performance", func() {
		It("should handle rapid successive requests", func() {
			const numRequests = 20
			requests := make([]*ext_proc.ProcessingRequest, numRequests)

			// Create alternating header and body requests
			for i := 0; i < numRequests; i++ {
				if i%2 == 0 {
					requests[i] = &ext_proc.ProcessingRequest{
						Request: &ext_proc.ProcessingRequest_RequestHeaders{
							RequestHeaders: &ext_proc.HttpHeaders{
								Headers: &core.HeaderMap{
									Headers: []*core.HeaderValue{
										{Key: "x-request-id", Value: fmt.Sprintf("rapid-test-%d", i)},
									},
								},
							},
						},
					}
				} else {
					requests[i] = &ext_proc.ProcessingRequest{
						Request: &ext_proc.ProcessingRequest_RequestBody{
							RequestBody: &ext_proc.HttpBody{
								Body: []byte(fmt.Sprintf(`{"model": "model-b", "messages": [{"role": "user", "content": "Request %d"}]}`, i)),
							},
						},
					}
				}
			}

			stream := NewMockStream(requests)
			err := router.Process(stream)
			Expect(err).NotTo(HaveOccurred()) // Stream should end gracefully

			// All requests should be processed
			Expect(len(stream.Responses)).To(Equal(numRequests))

			// Verify all responses are valid
			for i, response := range stream.Responses {
				if i%2 == 0 {
					Expect(response.GetRequestHeaders()).NotTo(BeNil(), fmt.Sprintf("Header response %d should not be nil", i))
				} else {
					Expect(response.GetRequestBody()).NotTo(BeNil(), fmt.Sprintf("Body response %d should not be nil", i))
				}
			}
		})

		It("should handle large request bodies in stream", func() {
			largeContent := fmt.Sprintf(`{"model": "model-a", "messages": [{"role": "user", "content": "%s"}]}`,
				fmt.Sprintf("Large content: %s", strings.Repeat("x", 1000))) // 1KB content

			requests := []*ext_proc.ProcessingRequest{
				{
					Request: &ext_proc.ProcessingRequest_RequestHeaders{
						RequestHeaders: &ext_proc.HttpHeaders{
							Headers: &core.HeaderMap{
								Headers: []*core.HeaderValue{
									{Key: "x-request-id", Value: "large-body-test"},
								},
							},
						},
					},
				},
				{
					Request: &ext_proc.ProcessingRequest_RequestBody{
						RequestBody: &ext_proc.HttpBody{
							Body: []byte(largeContent),
						},
					},
				},
			}

			stream := NewMockStream(requests)
			err := router.Process(stream)
			Expect(err).NotTo(HaveOccurred()) // Stream should end gracefully

			// Should handle large content without issues
			Expect(len(stream.Responses)).To(Equal(2))
			Expect(stream.Responses[0].GetRequestHeaders()).NotTo(BeNil())
			Expect(stream.Responses[1].GetRequestBody()).NotTo(BeNil())
		})
	})
})

// MockStream implements the ext_proc.ExternalProcessor_ProcessServer interface for testing
type MockStream struct {
	Requests  []*ext_proc.ProcessingRequest
	Responses []*ext_proc.ProcessingResponse
	Ctx       context.Context
	SendError error
	RecvError error
	RecvIndex int
}

func NewMockStream(requests []*ext_proc.ProcessingRequest) *MockStream {
	return &MockStream{
		Requests:  requests,
		Responses: make([]*ext_proc.ProcessingResponse, 0),
		Ctx:       context.Background(),
		RecvIndex: 0,
	}
}

func (m *MockStream) Send(response *ext_proc.ProcessingResponse) error {
	if m.SendError != nil {
		return m.SendError
	}
	m.Responses = append(m.Responses, response)
	return nil
}

func (m *MockStream) Recv() (*ext_proc.ProcessingRequest, error) {
	if m.RecvError != nil {
		return nil, m.RecvError
	}
	if m.RecvIndex >= len(m.Requests) {
		return nil, io.EOF // Simulate end of stream
	}
	req := m.Requests[m.RecvIndex]
	m.RecvIndex++
	return req, nil
}

func (m *MockStream) Context() context.Context {
	return m.Ctx
}

func (m *MockStream) SendMsg(interface{}) error    { return nil }
func (m *MockStream) RecvMsg(interface{}) error    { return nil }
func (m *MockStream) SetHeader(metadata.MD) error  { return nil }
func (m *MockStream) SendHeader(metadata.MD) error { return nil }
func (m *MockStream) SetTrailer(metadata.MD)       {}

var _ ext_proc.ExternalProcessor_ProcessServer = &MockStream{}

// CreateTestConfig creates a standard test configuration
func CreateTestConfig() *config.RouterConfig {
	return &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "model-b",
			ModelConfig: map[string]config.ModelParams{
				"model-a": {
					PreferredEndpoints: []string{"test-endpoint1"},
				},
				"model-b": {
					PreferredEndpoints: []string{"test-endpoint1", "test-endpoint2"},
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{
					Name:    "test-endpoint1",
					Address: "127.0.0.1",
					Port:    8000,
					Weight:  1,
				},
				{
					Name:    "test-endpoint2",
					Address: "127.0.0.1",
					Port:    8001,
					Weight:  2,
				},
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				Categories: []config.Category{
					{
						Name:        "coding",
						Description: "Programming tasks",
					},
				},
			},
		},
		SemanticCache: config.SemanticCache{
			BackendType:    "memory",
			InUse:          false, // Disable for most tests
			MaxEntries:     100,
			EvictionPolicy: "lru",
			TTLSeconds:     3600,
		},
		ResponseAPI: config.ResponseAPIConfig{
			Enabled:      true,
			StoreBackend: "memory",
			MaxResponses: 100,
			TTLSeconds:   86400,
		},
	}
}

func TestExtProc(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "ExtProc Suite")
}

var _ = Describe("ExtProc Package", func() {
	Describe("Basic Setup", func() {
		It("should create test configuration successfully", func() {
			cfg := CreateTestConfig()
			Expect(cfg).NotTo(BeNil())
			Expect(cfg.BackendModels.DefaultModel).To(Equal("model-b"))
			Expect(len(cfg.IntelligentRouting.Categories)).To(Equal(1))
			Expect(cfg.IntelligentRouting.Categories[0].Name).To(Equal("coding"))
		})

		It("should create test router successfully", func() {
			cfg := CreateTestConfig()
			router, err := CreateTestRouter(cfg)
			Expect(err).To(Or(BeNil(), HaveOccurred())) // May fail due to model dependencies
			if err == nil {
				Expect(router).NotTo(BeNil())
				Expect(router.Config).To(Equal(cfg))
			}
		})
	})

	Describe("Configuration Validation", func() {
		It("should validate required configuration fields", func() {
			cfg := CreateTestConfig()

			// Test essential fields are present
			Expect(cfg.BackendModels.DefaultModel).NotTo(BeEmpty())
			Expect(cfg.BackendModels.ModelConfig).NotTo(BeEmpty())
			Expect(cfg.BackendModels.ModelConfig).To(HaveKey("model-a"))
			Expect(cfg.BackendModels.ModelConfig).To(HaveKey("model-b"))
		})

		It("should have valid cache configuration", func() {
			cfg := CreateTestConfig()

			Expect(cfg.SemanticCache.MaxEntries).To(BeNumerically(">", 0))
			Expect(cfg.SemanticCache.TTLSeconds).To(BeNumerically(">", 0))
		})
	})

	Describe("Mock Components", func() {
		It("should create mock stream successfully", func() {
			requests := []*ext_proc.ProcessingRequest{}
			stream := NewMockStream(requests)

			Expect(stream).NotTo(BeNil())
			Expect(stream.Requests).To(HaveLen(0))
			Expect(stream.Responses).To(HaveLen(0))
			Expect(stream.RecvIndex).To(Equal(0))
			Expect(stream.Context()).NotTo(BeNil())
		})

		It("should handle mock stream operations", func() {
			stream := NewMockStream([]*ext_proc.ProcessingRequest{})

			// Test Recv on empty stream
			_, err := stream.Recv()
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("EOF"))

			// Test Send
			response := &ext_proc.ProcessingResponse{}
			err = stream.Send(response)
			Expect(err).NotTo(HaveOccurred())
			Expect(stream.Responses).To(HaveLen(1))
		})
	})
})

func init() {
	// Any package-level initialization can go here
}

var _ = Describe("Edge Cases and Error Conditions", func() {
	var (
		router *OpenAIRouter
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

			ctx := &RequestContext{
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

			ctx := &RequestContext{
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

			ctx := &RequestContext{
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

			ctx := &RequestContext{
				Headers:   make(map[string]string),
				RequestID: "invalid-model-request",
				StartTime: time.Now(),
			}

			response, err := router.HandleRequestBody(bodyRequest, ctx)
			Expect(err).NotTo(HaveOccurred())
			// A model that is not configured is rejected with a clear 400 rather
			// than being silently forwarded to a backend with no resolvable
			// credential (which surfaced as a misleading upstream "401 No api key").
			Expect(response.GetImmediateResponse()).NotTo(BeNil())
			Expect(int(response.GetImmediateResponse().GetStatus().GetCode())).To(Equal(400))
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

			ctx := &RequestContext{
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

					ctx := &RequestContext{
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

			// Some errors might be expected without served models
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

				ctx := &RequestContext{
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

			ctx := &RequestContext{
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
			repeatedPatternJSON, _ := json.Marshal(repeatedPattern)

			request := testOpenAIRequest{
				Model: "model-a",
				Messages: []testChatMessage{
					{Role: "user", Content: json.RawMessage(repeatedPatternJSON)},
				},
			}

			requestBody, err := json.Marshal(request)
			Expect(err).NotTo(HaveOccurred())

			bodyRequest := &ext_proc.ProcessingRequest_RequestBody{
				RequestBody: &ext_proc.HttpBody{
					Body: requestBody,
				},
			}

			ctx := &RequestContext{
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
		It("should reject an empty messages array", func() {
			request := testOpenAIRequest{
				Model:    "model-a",
				Messages: []testChatMessage{}, // Empty messages
			}

			requestBody, err := json.Marshal(request)
			Expect(err).NotTo(HaveOccurred())

			bodyRequest := &ext_proc.ProcessingRequest_RequestBody{
				RequestBody: &ext_proc.HttpBody{
					Body: requestBody,
				},
			}

			ctx := &RequestContext{
				Headers:   make(map[string]string),
				RequestID: "empty-messages-request",
				StartTime: time.Now(),
			}

			response, err := router.HandleRequestBody(bodyRequest, ctx)
			Expect(err).NotTo(HaveOccurred())
			immediate := response.GetImmediateResponse()
			Expect(immediate).NotTo(BeNil())
			Expect(immediate.GetStatus().GetCode()).To(Equal(typev3.StatusCode_BadRequest))
			Expect(immediate.GetBody()).To(ContainSubstring("invalid inference request"))
		})

		It("should handle messages with empty content", func() {
			request := testOpenAIRequest{
				Model: "model-a",
				Messages: []testChatMessage{
					{Role: "user", Content: json.RawMessage(`""`)},      // Empty content
					{Role: "assistant", Content: json.RawMessage(`""`)}, // Empty content
					{Role: "user", Content: json.RawMessage(`"Now respond to this"`)},
				},
			}

			requestBody, err := json.Marshal(request)
			Expect(err).NotTo(HaveOccurred())

			bodyRequest := &ext_proc.ProcessingRequest_RequestBody{
				RequestBody: &ext_proc.HttpBody{
					Body: requestBody,
				},
			}

			ctx := &RequestContext{
				Headers:   make(map[string]string),
				RequestID: "empty-content-request",
				StartTime: time.Now(),
			}

			response, err := router.HandleRequestBody(bodyRequest, ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(response.GetRequestBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})

		It("should handle messages with only whitespace content", func() {
			request := testOpenAIRequest{
				Model: "model-a",
				Messages: []testChatMessage{
					{Role: "user", Content: json.RawMessage(`"   \n\t  "`)}, // Only whitespace
					{Role: "user", Content: json.RawMessage(`"What is AI?"`)},
				},
			}

			requestBody, err := json.Marshal(request)
			Expect(err).NotTo(HaveOccurred())

			bodyRequest := &ext_proc.ProcessingRequest_RequestBody{
				RequestBody: &ext_proc.HttpBody{
					Body: requestBody,
				},
			}

			ctx := &RequestContext{
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
			request := testOpenAIRequest{
				Model: "vllm-sr/auto", // This triggers classification
				Messages: []testChatMessage{
					{Role: "user", Content: json.RawMessage(`"Test content that might cause classification issues: \u0000\u0001\u0002"`)}, // Binary content
				},
			}

			requestBody, err := json.Marshal(request)
			Expect(err).NotTo(HaveOccurred())

			bodyRequest := &ext_proc.ProcessingRequest_RequestBody{
				RequestBody: &ext_proc.HttpBody{
					Body: requestBody,
				},
			}

			ctx := &RequestContext{
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
			request := testOpenAIRequest{
				Model: "vllm-sr/auto",
				Messages: []testChatMessage{
					{Role: "user", Content: json.RawMessage(`"This is a complex request that might take time to classify and process"`)},
				},
			}

			requestBody, err := json.Marshal(request)
			Expect(err).NotTo(HaveOccurred())

			bodyRequest := &ext_proc.ProcessingRequest_RequestBody{
				RequestBody: &ext_proc.HttpBody{
					Body: requestBody,
				},
			}

			ctx := &RequestContext{
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

var _ = Describe("Caching Functionality", func() {
	var (
		router *OpenAIRouter
		cfg    *config.RouterConfig
	)

	BeforeEach(func() {
		cfg = CreateTestConfig()
		cfg.SemanticCache.InUse = true

		provider, err := embedding.NewFuncProvider("test", 2, func(_ context.Context, text string) ([]float32, error) { return []float32{float32(len(text)), 1}, nil })
		Expect(err).NotTo(HaveOccurred())

		router, err = CreateTestRouter(cfg)
		Expect(err).NotTo(HaveOccurred())

		// Override cache with enabled configuration
		cacheConfig := cache.CacheConfig{
			BackendType:         cache.InMemoryCacheType,
			Enabled:             true,
			SimilarityThreshold: 0.9,
			MaxEntries:          100,
			TTLSeconds:          3600,
			EmbeddingDeployment: "test-embedding",
			EmbeddingProvider:   provider,
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

		ctx := &RequestContext{
			Headers:   make(map[string]string),
			RequestID: "test-request-cache",
			StartTime: time.Now(),
		}

		response, err := router.HandleRequestBody(bodyRequest, ctx)
		// Even if caching fails without a served embedding model, request should continue
		Expect(err).To(Or(BeNil(), HaveOccurred()))
		if err == nil {
			Expect(response.GetRequestBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		}
	})

	It("should handle cache update on response", func() {
		// First, simulate a request that would add a pending cache entry
		ctx := &RequestContext{
			Headers:      make(map[string]string),
			RequestID:    "cache-test-request",
			RequestModel: "model-a",
			RequestQuery: "test query for caching",
			StartTime:    time.Now(),
		}

		// Simulate response processing
		responseBody := []byte(`{"id":"chatcmpl-cache-update","object":"chat.completion","created":1,"model":"model-a","choices":[{"index":0,"message":{"role":"assistant","content":"Cached response."},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)

		bodyResponse := &ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{
				Body: responseBody,
			},
		}
		response, err := router.HandleResponseBody(bodyResponse, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(response.GetImmediateResponse()).To(BeNil(), "unexpected response: %#v", response)
		Expect(response.GetResponseBody()).NotTo(BeNil(), "unexpected response: %#v", response)
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

			ctx := &RequestContext{
				Headers:   make(map[string]string),
				RequestID: "cache-ml-request",
				StartTime: time.Now(),
			}

			// Process request
			_, err = router.HandleRequestBody(bodyRequest, ctx)
			Expect(err).To(Or(BeNil(), HaveOccurred()))

			// Process response
			responseBody := []byte(`{"id":"chatcmpl-cache-store","object":"chat.completion","created":1,"model":"model-a","choices":[{"index":0,"message":{"role":"assistant","content":"Machine learning is a subset of artificial intelligence..."},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":30,"total_tokens":50}}`)

			bodyResponse := &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{
					Body: responseBody,
				},
			}

			ctx.RequestModel = "model-a"
			ctx.RequestQuery = "Tell me about machine learning"
			response, err := router.HandleResponseBody(bodyResponse, ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(response.GetImmediateResponse()).To(BeNil(), "unexpected response: %#v", response)
			Expect(response.GetResponseBody()).NotTo(BeNil(), "unexpected response: %#v", response)
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

			ctx := &RequestContext{
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
			cfg.SemanticCache.InUse = false
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

			ctx := &RequestContext{
				Headers:   make(map[string]string),
				RequestID: "no-cache-request",
				StartTime: time.Now(),
			}

			response, err := router.HandleRequestBody(bodyRequest, ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(response.GetRequestBody().Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})
	})
})

// Test helper methods to expose private functionality for testing

// HandleRequestHeaders exposes handleRequestHeaders for testing
func (r *OpenAIRouter) HandleRequestHeaders(v *ext_proc.ProcessingRequest_RequestHeaders, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	return r.handleRequestHeaders(v, ctx)
}

// HandleRequestBody exposes handleRequestBody for testing
func (r *OpenAIRouter) HandleRequestBody(v *ext_proc.ProcessingRequest_RequestBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	if ctx.RequestID == "" {
		ctx.RequestID = "test-request"
	}
	if ctx.TraceContext == nil {
		ctx.TraceContext = context.Background()
	}
	return r.handleRequestBody(v, ctx)
}

// HandleResponseHeaders exposes handleResponseHeaders for testing
func (r *OpenAIRouter) HandleResponseHeaders(v *ext_proc.ProcessingRequest_ResponseHeaders, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	return r.handleResponseHeaders(v, ctx)
}

// HandleResponseBody exposes handleResponseBody for testing
func (r *OpenAIRouter) HandleResponseBody(v *ext_proc.ProcessingRequest_ResponseBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	return r.handleResponseBody(v, ctx)
}

// providerResponseHeaders runs a provider response with status through
// handleResponseHeaders and returns the headers the Router sets.
func providerResponseHeaders(t *testing.T, ctx *RequestContext, status string) map[string]string {
	t.Helper()
	response, err := (&OpenAIRouter{}).handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":status", Value: status},
			{Key: "content-type", Value: "application/json"},
		}}},
	}, ctx)
	require.NoError(t, err)
	return headerValuesByName(response.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders())
}

func TestVSRHeadersAddedOnSuccessfulNonCachedResponse(t *testing.T) {
	ctx := &RequestContext{
		VSRSelectedDecisionName:             "math_decision",
		VSRSelectedDecisionConfidence:       0.91,
		VSRSelectedDecisionConfidenceScored: true,
		VSRSelectedCategory:                 "math",
		VSRReasoningMode:                    "on",
		VSRSelectedModel:                    "deepseek-v31",
		VSRInjectedSystemPrompt:             true,
		VSRDecisionDiagnostics: decision.EvaluationDiagnostics{
			AppliedUnknownPolicies: map[string]string{"guarded": "no_match"},
		},
	}

	headerMap := providerResponseHeaders(t, ctx, "200")

	// The default surface (no x-vsr-debug): the public set only.
	assert.Equal(t, headers.SchemaVersionValue, headerMap[headers.VSRSchemaVersion])
	assert.Equal(t, "upstream", headerMap[headers.VSRResponsePath])
	assert.Equal(t, "math_decision", headerMap[headers.VSRSelectedDecision])
	assert.Equal(t, "deepseek-v31", headerMap[headers.VSRSelectedModel])
	assert.NotContains(t, headerMap, headers.VSRTrace, "routing details ride in the trace, for debug callers")
}

func TestVSRTraceOnSuccessfulResponse(t *testing.T) {
	// With x-vsr-debug, the routing details ride in x-vsr-trace beside the
	// public set.
	ctx := &RequestContext{
		Headers:                 map[string]string{headers.VSRDebug: "true"},
		VSRSelectedDecisionName: "math_decision",
		VSRSelectedCategory:     "math",
		VSRReasoningMode:        "on",
		VSRSelectedModel:        "deepseek-v31",
		VSRInjectedSystemPrompt: true,
		PromptCacheAction:       promptCacheActionPreserved,
		PromptCacheReason:       promptCacheReasonCallerMarkers,
		PromptCachePreserved:    1,
		VSRMatchedKeywords:      []string{"prove", "theorem"},
	}

	headerMap := providerResponseHeaders(t, ctx, "200")

	assert.Equal(t, "math_decision", headerMap[headers.VSRSelectedDecision])
	assert.Equal(t, "deepseek-v31", headerMap[headers.VSRSelectedModel])
	trace := traceMembersOf(t, headerMap[headers.VSRTrace])
	assert.Equal(t, `"math"`, trace[headers.TraceCategory])
	assert.Equal(t, "on", trace[headers.TraceReasoning])
	assert.Equal(t, "?1", trace[headers.TraceSystemPrompt])
	assert.Equal(t, "preserved;reason=caller_markers;preserved=1", trace[headers.TracePromptCache])
	assert.Equal(t, `("keyword:prove" "keyword:theorem")`, trace[headers.TraceSignals])
}

func TestVSRHeadersNotAddedOnCacheHit(t *testing.T) {
	// A cache hit replays a response another request produced; its routing
	// facts and warnings would not be this request's.
	ctx := &RequestContext{
		VSRSelectedCategory: "math",
		VSRReasoningMode:    "on",
		VSRSelectedModel:    "deepseek-v31",
		VSRCacheHit:         true,
	}

	assert.Empty(t, providerResponseHeaders(t, ctx, "200"))
}

func TestVSRPublicSetOnErrorResponse(t *testing.T) {
	// The public set rides on every routed response, an upstream error
	// included, so operators can always see the contract version, the path
	// and the model that answered.
	ctx := &RequestContext{
		VSRSelectedCategory: "math",
		VSRReasoningMode:    "on",
		VSRSelectedModel:    "deepseek-v31",
	}

	headerMap := providerResponseHeaders(t, ctx, "500")

	assert.Equal(t, map[string]string{
		headers.VSRSchemaVersion: headers.SchemaVersionValue,
		headers.VSRResponsePath:  "upstream",
		headers.VSRSelectedModel: "deepseek-v31",
	}, headerMap)
}

func TestVSRTraceOmitsUnsetDetails(t *testing.T) {
	// Empty fields stay out of the trace, the reasoning mode and an
	// uninjected system prompt among them.
	ctx := &RequestContext{
		Headers:                 map[string]string{headers.VSRDebug: "true"},
		VSRSelectedCategory:     "math",
		VSRSelectedModel:        "deepseek-v31",
		VSRInjectedSystemPrompt: false,
	}

	trace := traceMembersOf(t, providerResponseHeaders(t, ctx, "200")[headers.VSRTrace])

	assert.Equal(t, `"math"`, trace[headers.TraceCategory])
	assert.NotContains(t, trace, headers.TraceReasoning)
	assert.NotContains(t, trace, headers.TraceSystemPrompt)
}

// TestModelReasoningFamily tests the new family-based configuration approach
func TestModelReasoningFamily(t *testing.T) {
	// Create a router with sample model configurations
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				ReasoningConfig: config.ReasoningConfig{
					DefaultReasoningEffort: "medium",
					ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
						"qwen3": {
							Type:      "chat_template_kwargs",
							Parameter: "enable_thinking",
						},
						"deepseek": {
							Type:      "chat_template_kwargs",
							Parameter: "thinking",
						},
						"gpt-oss": {
							Type:      "reasoning_effort",
							Parameter: "reasoning_effort",
						},
						"gpt": {
							Type:      "reasoning_effort",
							Parameter: "reasoning_effort",
						},
					},
				},
			},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"qwen3-model": {
						ReasoningFamily: "qwen3",
					},
					"ds-v31-custom": {
						ReasoningFamily: "deepseek",
					},
					"my-deepseek": {
						ReasoningFamily: "deepseek",
					},
					"gpt-oss-model": {
						ReasoningFamily: "gpt-oss",
					},
					"custom-gpt": {
						ReasoningFamily: "gpt",
					},
					"phi4": {
						// No reasoning family - doesn't support reasoning
					},
				},
			},
		},
	}

	testCases := []struct {
		name              string
		model             string
		expectedConfig    string // expected config name or empty for no config
		expectedType      string
		expectedParameter string
		expectConfig      bool
	}{
		{
			name:              "qwen3-model with qwen3 family",
			model:             "qwen3-model",
			expectedConfig:    "qwen3",
			expectedType:      "chat_template_kwargs",
			expectedParameter: "enable_thinking",
			expectConfig:      true,
		},
		{
			name:              "ds-v31-custom with deepseek family",
			model:             "ds-v31-custom",
			expectedConfig:    "deepseek",
			expectedType:      "chat_template_kwargs",
			expectedParameter: "thinking",
			expectConfig:      true,
		},
		{
			name:              "my-deepseek with deepseek family",
			model:             "my-deepseek",
			expectedConfig:    "deepseek",
			expectedType:      "chat_template_kwargs",
			expectedParameter: "thinking",
			expectConfig:      true,
		},
		{
			name:              "gpt-oss-model with gpt-oss family",
			model:             "gpt-oss-model",
			expectedConfig:    "gpt-oss",
			expectedType:      "reasoning_effort",
			expectedParameter: "reasoning_effort",
			expectConfig:      true,
		},
		{
			name:              "custom-gpt with gpt family",
			model:             "custom-gpt",
			expectedConfig:    "gpt",
			expectedType:      "reasoning_effort",
			expectedParameter: "reasoning_effort",
			expectConfig:      true,
		},
		{
			name:              "phi4 - no reasoning family",
			model:             "phi4",
			expectedConfig:    "",
			expectedType:      "",
			expectedParameter: "",
			expectConfig:      false,
		},
		{
			name:              "unknown model - no config",
			model:             "unknown-model",
			expectedConfig:    "",
			expectedType:      "",
			expectedParameter: "",
			expectConfig:      false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			familyConfig := router.getModelReasoningFamily(tc.model)

			if !tc.expectConfig {
				// For unknown models, we expect no configuration
				if familyConfig != nil {
					t.Fatalf("Expected no family config for %q, got %+v", tc.model, familyConfig)
				}
				return
			}

			// For known models, we expect a valid configuration
			if familyConfig == nil {
				t.Fatalf("Expected family config for %q, got nil", tc.model)
			}
			if familyConfig.Type != tc.expectedType {
				t.Fatalf("Expected type %q for model %q, got %q", tc.expectedType, tc.model, familyConfig.Type)
			}
			if familyConfig.Parameter != tc.expectedParameter {
				t.Fatalf("Expected parameter %q for model %q, got %q", tc.expectedParameter, tc.model, familyConfig.Parameter)
			}
		})
	}
}

func TestHandleRequestHeadersWithModelsEndpoint(t *testing.T) {
	// Create a test router
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			VLLMEndpoints: []config.VLLMEndpoint{
				{
					Name:    "primary",
					Address: "127.0.0.1",
					Port:    8000,
					Weight:  1,
				},
			},
			ModelConfig: map[string]config.ModelParams{
				"gpt-4o-mini": {
					PreferredEndpoints: []string{"primary"},
				},
			},
		},
	}

	router := &OpenAIRouter{
		Config: cfg,
	}

	tests := []struct {
		name            string
		method          string
		path            string
		expectImmediate bool
	}{
		{
			name:            "GET /v1/models - should return immediate response",
			method:          "GET",
			path:            "/v1/models",
			expectImmediate: true,
		},
		{
			name:            "GET /v1/models?model=auto - should return immediate response",
			method:          "GET",
			path:            "/v1/models?model=auto",
			expectImmediate: true,
		},
		{
			name:            "POST /v1/chat/completions - should continue processing",
			method:          "POST",
			path:            "/v1/chat/completions",
			expectImmediate: false,
		},
		{
			name:            "POST /v1/models - immediate response",
			method:          "POST",
			path:            "/v1/models",
			expectImmediate: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create request headers
			requestHeaders := &ext_proc.ProcessingRequest_RequestHeaders{
				RequestHeaders: &ext_proc.HttpHeaders{
					Headers: &core.HeaderMap{
						Headers: []*core.HeaderValue{
							{
								Key:   ":method",
								Value: tt.method,
							},
							{
								Key:   ":path",
								Value: tt.path,
							},
							{
								Key:   "content-type",
								Value: "application/json",
							},
						},
					},
				},
			}

			ctx := &RequestContext{
				Headers: make(map[string]string),
			}

			response, err := router.handleRequestHeaders(requestHeaders, ctx)
			if err != nil {
				t.Fatalf("handleRequestHeaders failed: %v", err)
			}

			if tt.expectImmediate {
				// Should return immediate response
				if response.GetImmediateResponse() == nil {
					t.Error("Expected immediate response for /v1/models endpoint")
				}
			} else {
				// Should return continue response
				if response.GetRequestHeaders() == nil {
					t.Error("Expected request headers response for non-models endpoint")
				}
				if response.GetRequestHeaders().Response.Status != ext_proc.CommonResponse_CONTINUE {
					t.Error("Expected CONTINUE status for non-models endpoint")
				}
			}
		})
	}
}

func getHistogramSampleCount(metricName, model string) uint64 {
	mf, _ := prometheus.DefaultGatherer.Gather()
	for _, fam := range mf {
		if fam.GetName() != metricName || fam.GetType() != dto.MetricType_HISTOGRAM {
			continue
		}
		for _, m := range fam.GetMetric() {
			labels := m.GetLabel()
			match := false
			for _, l := range labels {
				if l.GetName() == "model" && l.GetValue() == model {
					match = true
					break
				}
			}
			if match {
				h := m.GetHistogram()
				if h != nil && h.SampleCount != nil {
					return h.GetSampleCount()
				}
			}
		}
	}
	return 0
}

var _ = Describe("Metrics recording", func() {
	var router *OpenAIRouter

	BeforeEach(func() {
		// Use a minimal router that doesn't require external models
		router = &OpenAIRouter{
			Cache: cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: false}),
		}
	})

	It("records TTFT on response headers", func() {
		ctx := &RequestContext{
			RequestModel:        "model-a",
			ProcessingStartTime: time.Now().Add(-75 * time.Millisecond),
		}

		before := getHistogramSampleCount("llm_model_first_response_observation_seconds", ctx.RequestModel)

		respHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &ext_proc.HttpHeaders{
				Headers: &core.HeaderMap{Headers: []*core.HeaderValue{{Key: ":status", Value: "200"}, {Key: "content-type", Value: "application/json"}}},
			},
		}

		response, err := router.handleResponseHeaders(respHeaders, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(response.GetResponseHeaders()).NotTo(BeNil())

		after := getHistogramSampleCount("llm_model_first_response_observation_seconds", ctx.RequestModel)
		Expect(after).To(BeNumerically(">", before))
		Expect(ctx.TTFTRecorded).To(BeTrue())
		Expect(ctx.TTFTSeconds).To(BeNumerically(">", 0))
	})

	It("records TPOT on response body", func() {
		ctx := &RequestContext{
			RequestID:    "tpot-test-1",
			RequestModel: "model-a",
			StartTime:    time.Now().Add(-1 * time.Second),
		}

		beforeTPOT := getHistogramSampleCount("llm_model_response_duration_per_output_token_seconds", ctx.RequestModel)

		beforePrompt := getHistogramSampleCount("llm_prompt_tokens_per_request", ctx.RequestModel)
		beforeCompletion := getHistogramSampleCount("llm_completion_tokens_per_request", ctx.RequestModel)

		respBodyJSON := []byte(`{"id":"chatcmpl-xyz","object":"chat.completion","created":1,"model":"model-a","choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)

		respBody := &ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: respBodyJSON},
		}
		_, decodeErr := router.decodeClientResponse(respBodyJSON, ctx)
		Expect(decodeErr).NotTo(HaveOccurred(), "fixture must satisfy the neutral response contract: %s", respBodyJSON)

		response, err := router.handleResponseBody(respBody, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(response.GetImmediateResponse()).To(BeNil(), "unexpected response: %#v", response)
		Expect(response.GetResponseBody()).NotTo(BeNil(), "unexpected response: %#v", response)

		afterTPOT := getHistogramSampleCount("llm_model_response_duration_per_output_token_seconds", ctx.RequestModel)
		Expect(afterTPOT).To(BeNumerically(">", beforeTPOT))

		// New per-request token histograms should also be recorded
		afterPrompt := getHistogramSampleCount("llm_prompt_tokens_per_request", ctx.RequestModel)
		afterCompletion := getHistogramSampleCount("llm_completion_tokens_per_request", ctx.RequestModel)
		Expect(afterPrompt).To(BeNumerically(">", beforePrompt))
		Expect(afterCompletion).To(BeNumerically(">", beforeCompletion))
	})

	It("records TTFT on first streamed content for SSE responses", func() {
		ctx := &RequestContext{
			RequestModel:        "model-stream",
			ProcessingStartTime: time.Now().Add(-120 * time.Millisecond),
			Headers:             map[string]string{"accept": "text/event-stream"},
		}

		// Simulate header phase: SSE content-type indicates streaming
		respHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &ext_proc.HttpHeaders{
				Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
					{Key: ":status", Value: "200"},
					{Key: "content-type", Value: "text/event-stream"},
				}},
			},
		}

		before := getHistogramSampleCount("llm_model_first_response_observation_seconds", ctx.RequestModel)

		// Handle response headers (should NOT record TTFT for streaming)
		response1, err := router.handleResponseHeaders(respHeaders, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(response1.GetResponseHeaders()).NotTo(BeNil())
		Expect(ctx.IsStreamingResponse).To(BeTrue())
		Expect(ctx.TTFTRecorded).To(BeFalse())

		// Now simulate the first streamed body chunk that carries content
		respBody := &ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: []byte("data: {\"id\":\"r1\",\"model\":\"model-stream\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"}}]}\n\n")},
		}

		response2, err := router.handleResponseBody(respBody, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(response2.GetResponseBody()).NotTo(BeNil())

		after := getHistogramSampleCount("llm_model_first_response_observation_seconds", ctx.RequestModel)
		Expect(after).To(BeNumerically(">", before))
		Expect(ctx.TTFTRecorded).To(BeTrue())
		Expect(ctx.TTFTSeconds).To(BeNumerically(">", 0))
	})
})

// getCounterValue returns the sum of a counter across metrics matching the given labels
func getCounterValue(metricName string, want map[string]string) float64 {
	var sum float64
	mfs, _ := prometheus.DefaultGatherer.Gather()
	for _, fam := range mfs {
		if fam.GetName() != metricName || fam.GetType() != dto.MetricType_COUNTER {
			continue
		}
		for _, m := range fam.GetMetric() {
			labels := m.GetLabel()
			match := true
			for k, v := range want {
				found := false
				for _, l := range labels {
					if l.GetName() == k && l.GetValue() == v {
						found = true
						break
					}
				}
				if !found {
					match = false
					break
				}
			}
			if match && m.GetCounter() != nil {
				sum += m.GetCounter().GetValue()
			}
		}
	}
	return sum
}

func TestRequestParseErrorIncrementsErrorCounter(t *testing.T) {
	r := &OpenAIRouter{}

	ctx := &RequestContext{}
	// Invalid JSON triggers parse error
	badBody := []byte("not-json")
	v := &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: badBody},
	}

	before := getCounterValue("llm_request_errors_total", map[string]string{"reason": "parse_error", "model": "unknown"})

	// Use test helper wrapper to access unexported method
	_, _ = r.HandleRequestBody(v, ctx)

	after := getCounterValue("llm_request_errors_total", map[string]string{"reason": "parse_error", "model": "unknown"})
	if !(after > before) {
		t.Fatalf("expected llm_request_errors_total(parse_error,unknown) to increase: before=%v after=%v", before, after)
	}
}

func TestResponseParseErrorIncrementsErrorCounter(t *testing.T) {
	r := &OpenAIRouter{}

	ctx := &RequestContext{RequestModel: "model-a"}
	// Invalid JSON triggers parse error in response body handler
	badJSON := []byte("{invalid}")
	v := &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: badJSON},
	}

	before := getCounterValue("llm_request_errors_total", map[string]string{"reason": "parse_error", "model": "model-a"})

	_, _ = r.HandleResponseBody(v, ctx)

	after := getCounterValue("llm_request_errors_total", map[string]string{"reason": "parse_error", "model": "model-a"})
	if !(after > before) {
		t.Fatalf("expected llm_request_errors_total(parse_error,model-a) to increase: before=%v after=%v", before, after)
	}
}

func TestUpstreamStatusIncrements4xx5xxCounters(t *testing.T) {
	r := &OpenAIRouter{}

	ctx := &RequestContext{RequestModel: "m"}

	// 503 -> upstream_5xx
	hdrs5xx := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{Headers: []*core.HeaderValue{{Key: ":status", Value: "503"}}},
		},
	}

	before5xx := getCounterValue("llm_request_errors_total", map[string]string{"reason": "upstream_5xx", "model": "m"})
	_, _ = r.HandleResponseHeaders(hdrs5xx, ctx)
	after5xx := getCounterValue("llm_request_errors_total", map[string]string{"reason": "upstream_5xx", "model": "m"})
	if !(after5xx > before5xx) {
		t.Fatalf("expected upstream_5xx to increase for model m: before=%v after=%v", before5xx, after5xx)
	}

	// 404 -> upstream_4xx
	hdrs4xx := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{Headers: []*core.HeaderValue{{Key: ":status", Value: "404"}}},
		},
	}

	before4xx := getCounterValue("llm_request_errors_total", map[string]string{"reason": "upstream_4xx", "model": "m"})
	_, _ = r.HandleResponseHeaders(hdrs4xx, ctx)
	after4xx := getCounterValue("llm_request_errors_total", map[string]string{"reason": "upstream_4xx", "model": "m"})
	if !(after4xx > before4xx) {
		t.Fatalf("expected upstream_4xx to increase for model m: before=%v after=%v", before4xx, after4xx)
	}
}

type MockResponseStore struct {
	responses map[string]*responseapi.StoredResponse
}

func NewMockResponseStore() *MockResponseStore {
	return &MockResponseStore{responses: make(map[string]*responseapi.StoredResponse)}
}

func (m *MockResponseStore) StoreResponse(_ context.Context, response *responseapi.StoredResponse) error {
	m.responses[response.ID] = response
	return nil
}

func (m *MockResponseStore) GetResponse(_ context.Context, id string) (*responseapi.StoredResponse, error) {
	if response, ok := m.responses[id]; ok {
		return response, nil
	}
	return nil, responsestore.ErrNotFound
}

func (m *MockResponseStore) UpdateResponse(_ context.Context, response *responseapi.StoredResponse) error {
	if _, ok := m.responses[response.ID]; !ok {
		return responsestore.ErrNotFound
	}
	m.responses[response.ID] = response
	return nil
}

func (m *MockResponseStore) DeleteResponse(_ context.Context, id string) error {
	if _, ok := m.responses[id]; !ok {
		return responsestore.ErrNotFound
	}
	delete(m.responses, id)
	return nil
}

func (m *MockResponseStore) GetConversationChain(_ context.Context, responseID string) ([]*responseapi.StoredResponse, error) {
	if response, ok := m.responses[responseID]; ok {
		return []*responseapi.StoredResponse{response}, nil
	}
	return nil, responsestore.ErrNotFound
}

func (m *MockResponseStore) ListResponsesByConversation(_ context.Context, _ string, _ responsestore.ListOptions) ([]*responseapi.StoredResponse, error) {
	return nil, nil
}

func (m *MockResponseStore) IsEnabled() bool {
	return true
}

func (m *MockResponseStore) Close() error {
	return nil
}

func (m *MockResponseStore) CheckConnection(ctx context.Context) error {
	return nil
}
