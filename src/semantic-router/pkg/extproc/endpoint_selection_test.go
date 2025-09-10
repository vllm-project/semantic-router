package extproc_test

import (
	"encoding/json"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
)

var _ = Describe("Endpoint Selection", func() {
	var (
		router *extproc.OpenAIRouter
		cfg    *config.RouterConfig
	)

	BeforeEach(func() {
		cfg = CreateTestConfig()
		var err error
		router, err = CreateTestRouter(cfg)
		if err != nil {
			Skip("Skipping test due to model initialization failure: " + err.Error())
		}
	})

	Describe("Model Routing with Endpoint Selection", func() {
		Context("when model is 'auto'", func() {
			It("should select appropriate endpoint for automatically selected model", func() {
				// Create a request with model "auto"
				openAIRequest := map[string]interface{}{
					"model": "auto",
					"messages": []map[string]interface{}{
						{
							"role":    "user",
							"content": "Write a Python function to sort a list",
						},
					},
				}

				requestBody, err := json.Marshal(openAIRequest)
				Expect(err).NotTo(HaveOccurred())

				// Create processing request
				processingRequest := &ext_proc.ProcessingRequest{
					Request: &ext_proc.ProcessingRequest_RequestBody{
						RequestBody: &ext_proc.HttpBody{
							Body: requestBody,
						},
					},
				}

				// Create mock stream
				stream := NewMockStream([]*ext_proc.ProcessingRequest{processingRequest})

				// Process the request
				err = router.Process(stream)
				Expect(err).NotTo(HaveOccurred())

				// Verify response was sent
				Expect(stream.Responses).To(HaveLen(1))
				response := stream.Responses[0]

				// Check if headers were set for endpoint selection
				requestBodyResponse := response.GetRequestBody()
				Expect(requestBodyResponse).NotTo(BeNil())

				headerMutation := requestBodyResponse.GetResponse().GetHeaderMutation()
				if headerMutation != nil && len(headerMutation.SetHeaders) > 0 {
					// Verify that endpoint selection header is present
					var endpointHeaderFound bool
					var modelHeaderFound bool

					for _, header := range headerMutation.SetHeaders {
						if header.Header.Key == "x-semantic-destination-endpoint" {
							endpointHeaderFound = true
							// Should be one of the configured endpoint addresses
							// Check both Value and RawValue since implementation uses RawValue
							headerValue := header.Header.Value
							if headerValue == "" && len(header.Header.RawValue) > 0 {
								headerValue = string(header.Header.RawValue)
							}
							Expect(headerValue).To(BeElementOf("127.0.0.1:8000", "127.0.0.1:8001"))
						}
						if header.Header.Key == "x-selected-model" {
							modelHeaderFound = true
							// Should be one of the configured models
							// Check both Value and RawValue since implementation may use either
							headerValue := header.Header.Value
							if headerValue == "" && len(header.Header.RawValue) > 0 {
								headerValue = string(header.Header.RawValue)
							}
							Expect(headerValue).To(BeElementOf("model-a", "model-b"))
						}
					}

					// At least one of these should be true (endpoint header should be set when model routing occurs)
					Expect(endpointHeaderFound || modelHeaderFound).To(BeTrue())
				}
			})
		})

		Context("when model is explicitly specified", func() {
			It("should select appropriate endpoint for specified model", func() {
				// Create a request with explicit model
				openAIRequest := map[string]interface{}{
					"model": "model-a",
					"messages": []map[string]interface{}{
						{
							"role":    "user",
							"content": "Hello, world!",
						},
					},
				}

				requestBody, err := json.Marshal(openAIRequest)
				Expect(err).NotTo(HaveOccurred())

				// Create processing request
				processingRequest := &ext_proc.ProcessingRequest{
					Request: &ext_proc.ProcessingRequest_RequestBody{
						RequestBody: &ext_proc.HttpBody{
							Body: requestBody,
						},
					},
				}

				// Create mock stream
				stream := NewMockStream([]*ext_proc.ProcessingRequest{processingRequest})

				// Process the request
				err = router.Process(stream)
				Expect(err).NotTo(HaveOccurred())

				// Verify response was sent
				Expect(stream.Responses).To(HaveLen(1))
				response := stream.Responses[0]

				// Check if headers were set for endpoint selection
				requestBodyResponse := response.GetRequestBody()
				Expect(requestBodyResponse).NotTo(BeNil())

				headerMutation := requestBodyResponse.GetResponse().GetHeaderMutation()
				if headerMutation != nil && len(headerMutation.SetHeaders) > 0 {
					var endpointHeaderFound bool
					var selectedEndpoint string

					for _, header := range headerMutation.SetHeaders {
						if header.Header.Key == "x-semantic-destination-endpoint" {
							endpointHeaderFound = true
							// Check both Value and RawValue since implementation uses RawValue
							selectedEndpoint = header.Header.Value
							if selectedEndpoint == "" && len(header.Header.RawValue) > 0 {
								selectedEndpoint = string(header.Header.RawValue)
							}
							break
						}
					}

					if endpointHeaderFound {
						// model-a should be routed to test-endpoint1 based on preferred endpoints
						Expect(selectedEndpoint).To(Equal("127.0.0.1:8000"))
					}
				}
			})

			It("should handle model with multiple preferred endpoints", func() {
				// Create a request with model-b which has multiple preferred endpoints
				openAIRequest := map[string]interface{}{
					"model": "model-b",
					"messages": []map[string]interface{}{
						{
							"role":    "user",
							"content": "Test message",
						},
					},
				}

				requestBody, err := json.Marshal(openAIRequest)
				Expect(err).NotTo(HaveOccurred())

				// Create processing request
				processingRequest := &ext_proc.ProcessingRequest{
					Request: &ext_proc.ProcessingRequest_RequestBody{
						RequestBody: &ext_proc.HttpBody{
							Body: requestBody,
						},
					},
				}

				// Create mock stream
				stream := NewMockStream([]*ext_proc.ProcessingRequest{processingRequest})

				// Process the request
				err = router.Process(stream)
				Expect(err).NotTo(HaveOccurred())

				// Verify response was sent
				Expect(stream.Responses).To(HaveLen(1))
				response := stream.Responses[0]

				// Check if headers were set for endpoint selection
				requestBodyResponse := response.GetRequestBody()
				Expect(requestBodyResponse).NotTo(BeNil())

				headerMutation := requestBodyResponse.GetResponse().GetHeaderMutation()
				if headerMutation != nil && len(headerMutation.SetHeaders) > 0 {
					var endpointHeaderFound bool
					var selectedEndpoint string

					for _, header := range headerMutation.SetHeaders {
						if header.Header.Key == "x-semantic-destination-endpoint" {
							endpointHeaderFound = true
							// Check both Value and RawValue since implementation uses RawValue
							selectedEndpoint = header.Header.Value
							if selectedEndpoint == "" && len(header.Header.RawValue) > 0 {
								selectedEndpoint = string(header.Header.RawValue)
							}
							break
						}
					}

					if endpointHeaderFound {
						// model-b should be routed to test-endpoint2 (higher weight) or test-endpoint1
						Expect(selectedEndpoint).To(BeElementOf("127.0.0.1:8000", "127.0.0.1:8001"))
					}
				}
			})
		})
	})

	Describe("Endpoint Configuration Validation", func() {
		It("should have valid endpoint configuration in test config", func() {
			Expect(cfg.VLLMEndpoints).To(HaveLen(2))

			// Verify first endpoint
			endpoint1 := cfg.VLLMEndpoints[0]
			Expect(endpoint1.Name).To(Equal("test-endpoint1"))
			Expect(endpoint1.Address).To(Equal("127.0.0.1"))
			Expect(endpoint1.Port).To(Equal(8000))
			Expect(endpoint1.Models).To(ContainElements("model-a", "model-b"))
			Expect(endpoint1.Weight).To(Equal(1))

			// Verify second endpoint
			endpoint2 := cfg.VLLMEndpoints[1]
			Expect(endpoint2.Name).To(Equal("test-endpoint2"))
			Expect(endpoint2.Address).To(Equal("127.0.0.1"))
			Expect(endpoint2.Port).To(Equal(8001))
			Expect(endpoint2.Models).To(ContainElement("model-b"))
			Expect(endpoint2.Weight).To(Equal(2))
		})

		It("should pass endpoint validation", func() {
			err := cfg.ValidateEndpoints()
			Expect(err).NotTo(HaveOccurred())
		})

		It("should find correct endpoints for models", func() {
			// Test model-a (should find test-endpoint1)
			endpoints := cfg.GetEndpointsForModel("model-a")
			Expect(endpoints).To(HaveLen(1))
			Expect(endpoints[0].Name).To(Equal("test-endpoint1"))

			// Test model-b (should find both endpoints, but prefer test-endpoint2 due to weight)
			endpoints = cfg.GetEndpointsForModel("model-b")
			Expect(endpoints).To(HaveLen(2))
			endpointNames := []string{endpoints[0].Name, endpoints[1].Name}
			Expect(endpointNames).To(ContainElements("test-endpoint1", "test-endpoint2"))

			// Test best endpoint selection
			bestEndpoint, found := cfg.SelectBestEndpointForModel("model-b")
			Expect(found).To(BeTrue())
			Expect(bestEndpoint).To(BeElementOf("test-endpoint1", "test-endpoint2"))

			// Test best endpoint address selection
			bestEndpointAddress, found := cfg.SelectBestEndpointAddressForModel("model-b")
			Expect(found).To(BeTrue())
			Expect(bestEndpointAddress).To(BeElementOf("127.0.0.1:8000", "127.0.0.1:8001"))
		})
	})

	Describe("Request Context Processing", func() {
		It("should handle request headers properly", func() {
			// Create request headers
			requestHeaders := &ext_proc.ProcessingRequest{
				Request: &ext_proc.ProcessingRequest_RequestHeaders{
					RequestHeaders: &ext_proc.HttpHeaders{
						Headers: &core.HeaderMap{
							Headers: []*core.HeaderValue{
								{
									Key:   "content-type",
									Value: "application/json",
								},
								{
									Key:   "x-request-id",
									Value: "test-request-123",
								},
							},
						},
					},
				},
			}

			// Create mock stream with headers
			stream := NewMockStream([]*ext_proc.ProcessingRequest{requestHeaders})

			// Process the request
			err := router.Process(stream)
			Expect(err).NotTo(HaveOccurred())

			// Should have received a response
			Expect(stream.Responses).To(HaveLen(1))

			// Headers should be processed and allowed to continue
			response := stream.Responses[0]
			headersResponse := response.GetRequestHeaders()
			Expect(headersResponse).NotTo(BeNil())
			Expect(headersResponse.Response.Status).To(Equal(ext_proc.CommonResponse_CONTINUE))
		})
	})
})
