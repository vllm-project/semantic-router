package extproc_test

import (
	"context"
	"fmt"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
)

var _ = Describe("Process Stream Handling", func() {
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

	Context("with valid request sequence", func() {
		It("should handle complete request-response cycle", func() {
			// Create a sequence of requests
			requests := []*ext_proc.ProcessingRequest{
				{
					Request: &ext_proc.ProcessingRequest_RequestHeaders{
						RequestHeaders: &ext_proc.HttpHeaders{
							Headers: &core.HeaderMap{
								Headers: []*core.HeaderValue{
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
							Body: []byte(`{"choices": [{"message": {"content": "Hi there!"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}`),
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
									{Key: "content-type", Value: "application/json"},
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
