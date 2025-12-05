//go:build !windows && cgo

package benchmarks

import (
	"context"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
)

var (
	testRouter *extproc.OpenAIRouter
)

func setupRouter(b *testing.B) {
	if testRouter != nil {
		return
	}

	// Load config
	cfg, err := config.LoadConfig("../config/testing/config.e2e.yaml")
	if err != nil {
		b.Fatalf("Failed to load config: %v", err)
	}

	// Initialize router
	router, err := extproc.NewOpenAIRouter(cfg)
	if err != nil {
		b.Fatalf("Failed to create router: %v", err)
	}

	testRouter = router
	b.ResetTimer()
}

// mockStream implements a minimal ext_proc stream for testing
type mockStream struct {
	grpc.ServerStream
	ctx      context.Context
	requests []*ext_proc.ProcessingRequest
	recvIdx  int
	sent     []*ext_proc.ProcessingResponse
}

func newMockStream(ctx context.Context, requests []*ext_proc.ProcessingRequest) *mockStream {
	return &mockStream{
		ctx:      ctx,
		requests: requests,
		sent:     make([]*ext_proc.ProcessingResponse, 0),
	}
}

func (m *mockStream) Context() context.Context {
	return m.ctx
}

func (m *mockStream) Recv() (*ext_proc.ProcessingRequest, error) {
	if m.recvIdx >= len(m.requests) {
		return nil, nil
	}
	req := m.requests[m.recvIdx]
	m.recvIdx++
	return req, nil
}

func (m *mockStream) Send(resp *ext_proc.ProcessingResponse) error {
	m.sent = append(m.sent, resp)
	return nil
}

func (m *mockStream) SetHeader(metadata.MD) error  { return nil }
func (m *mockStream) SendHeader(metadata.MD) error { return nil }
func (m *mockStream) SetTrailer(metadata.MD)       {}
func (m *mockStream) SendMsg(interface{}) error    { return nil }
func (m *mockStream) RecvMsg(interface{}) error    { return nil }

// BenchmarkProcessRequest benchmarks basic request processing
func BenchmarkProcessRequest(b *testing.B) {
	setupRouter(b)

	ctx := context.Background()

	// Create a simple request headers message
	requests := []*ext_proc.ProcessingRequest{
		{
			Request: &ext_proc.ProcessingRequest_RequestHeaders{
				RequestHeaders: &ext_proc.HttpHeaders{
					Headers: &ext_proc.HeaderMap{
						Headers: []*ext_proc.HeaderValue{
							{Key: "content-type", Value: "application/json"},
							{Key: ":path", Value: "/v1/chat/completions"},
							{Key: ":method", Value: "POST"},
						},
					},
				},
			},
		},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		stream := newMockStream(ctx, requests)
		_ = testRouter.Process(stream)
	}
}

// BenchmarkProcessRequestBody benchmarks request body processing
func BenchmarkProcessRequestBody(b *testing.B) {
	setupRouter(b)

	ctx := context.Background()

	// Simulate request with headers and body
	body := []byte(`{"model":"auto","messages":[{"role":"user","content":"What is 2+2?"}]}`)

	requests := []*ext_proc.ProcessingRequest{
		{
			Request: &ext_proc.ProcessingRequest_RequestHeaders{
				RequestHeaders: &ext_proc.HttpHeaders{
					Headers: &ext_proc.HeaderMap{
						Headers: []*ext_proc.HeaderValue{
							{Key: "content-type", Value: "application/json"},
							{Key: ":path", Value: "/v1/chat/completions"},
						},
					},
				},
			},
		},
		{
			Request: &ext_proc.ProcessingRequest_RequestBody{
				RequestBody: &ext_proc.HttpBody{
					Body: body,
				},
			},
		},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		stream := newMockStream(ctx, requests)
		_ = testRouter.Process(stream)
	}
}

// BenchmarkHeaderProcessing benchmarks header processing overhead
func BenchmarkHeaderProcessing(b *testing.B) {
	setupRouter(b)

	ctx := context.Background()

	requests := []*ext_proc.ProcessingRequest{
		{
			Request: &ext_proc.ProcessingRequest_RequestHeaders{
				RequestHeaders: &ext_proc.HttpHeaders{
					Headers: &ext_proc.HeaderMap{
						Headers: []*ext_proc.HeaderValue{
							{Key: "content-type", Value: "application/json"},
							{Key: ":path", Value: "/v1/chat/completions"},
							{Key: ":method", Value: "POST"},
							{Key: "authorization", Value: "Bearer test-token"},
							{Key: "user-agent", Value: "test-client/1.0"},
						},
					},
				},
			},
		},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		stream := newMockStream(ctx, requests)
		_ = testRouter.Process(stream)
	}
}

// BenchmarkFullRequestFlow benchmarks complete request flow
func BenchmarkFullRequestFlow(b *testing.B) {
	setupRouter(b)

	ctx := context.Background()

	// Complete request flow: headers + body + response headers + response body
	body := []byte(`{"model":"auto","messages":[{"role":"user","content":"Solve this equation: x^2 + 5x + 6 = 0"}]}`)

	requests := []*ext_proc.ProcessingRequest{
		{
			Request: &ext_proc.ProcessingRequest_RequestHeaders{
				RequestHeaders: &ext_proc.HttpHeaders{
					Headers: &ext_proc.HeaderMap{
						Headers: []*ext_proc.HeaderValue{
							{Key: "content-type", Value: "application/json"},
							{Key: ":path", Value: "/v1/chat/completions"},
							{Key: ":method", Value: "POST"},
						},
					},
				},
			},
		},
		{
			Request: &ext_proc.ProcessingRequest_RequestBody{
				RequestBody: &ext_proc.HttpBody{
					Body: body,
				},
			},
		},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		stream := newMockStream(ctx, requests)
		_ = testRouter.Process(stream)
	}
}

// BenchmarkDifferentRequestTypes benchmarks various request types
func BenchmarkDifferentRequestTypes(b *testing.B) {
	setupRouter(b)

	testCases := []struct {
		name string
		body string
	}{
		{"Math", `{"model":"auto","messages":[{"role":"user","content":"What is the derivative of x^2?"}]}`},
		{"Code", `{"model":"auto","messages":[{"role":"user","content":"Write a Python function to reverse a string"}]}`},
		{"Business", `{"model":"auto","messages":[{"role":"user","content":"Analyze this business strategy"}]}`},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			ctx := context.Background()

			requests := []*ext_proc.ProcessingRequest{
				{
					Request: &ext_proc.ProcessingRequest_RequestHeaders{
						RequestHeaders: &ext_proc.HttpHeaders{
							Headers: &ext_proc.HeaderMap{
								Headers: []*ext_proc.HeaderValue{
									{Key: "content-type", Value: "application/json"},
									{Key: ":path", Value: "/v1/chat/completions"},
								},
							},
						},
					},
				},
				{
					Request: &ext_proc.ProcessingRequest_RequestBody{
						RequestBody: &ext_proc.HttpBody{
							Body: []byte(tc.body),
						},
					},
				},
			}

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				stream := newMockStream(ctx, requests)
				_ = testRouter.Process(stream)
			}
		})
	}
}

// BenchmarkConcurrentRequests benchmarks concurrent request processing
func BenchmarkConcurrentRequests(b *testing.B) {
	setupRouter(b)

	body := []byte(`{"model":"auto","messages":[{"role":"user","content":"Test message"}]}`)

	requests := []*ext_proc.ProcessingRequest{
		{
			Request: &ext_proc.ProcessingRequest_RequestHeaders{
				RequestHeaders: &ext_proc.HttpHeaders{
					Headers: &ext_proc.HeaderMap{
						Headers: []*ext_proc.HeaderValue{
							{Key: "content-type", Value: "application/json"},
							{Key: ":path", Value: "/v1/chat/completions"},
						},
					},
				},
			},
		},
		{
			Request: &ext_proc.ProcessingRequest_RequestBody{
				RequestBody: &ext_proc.HttpBody{
					Body: body,
				},
			},
		},
	}

	b.ResetTimer()
	b.ReportAllocs()

	b.RunParallel(func(pb *testing.PB) {
		ctx := context.Background()
		for pb.Next() {
			stream := newMockStream(ctx, requests)
			_ = testRouter.Process(stream)
		}
	})
}
