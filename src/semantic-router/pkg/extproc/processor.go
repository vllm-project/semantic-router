package extproc

import (
	"context"
	"errors"
	"io"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Process implements the ext_proc calls
func (r *OpenAIRouter) Process(stream ext_proc.ExternalProcessor_ProcessServer) error {
	observability.Infof("Started processing a new request")

	// Initialize request context
	ctx := &RequestContext{
		Headers: make(map[string]string),
	}

	for {
		req, err := stream.Recv()
		if err != nil {
			// Handle EOF - this indicates the client has closed the stream gracefully
			if err == io.EOF {
				observability.Infof("Stream ended gracefully")
				return nil
			}

			// Handle gRPC status-based cancellations/timeouts
			if s, ok := status.FromError(err); ok {
				switch s.Code() {
				case codes.Canceled:
					observability.Infof("Stream canceled gracefully")
					metrics.RecordRequestError(ctx.RequestModel, "cancellation")
					return nil
				case codes.DeadlineExceeded:
					observability.Infof("Stream deadline exceeded")
					metrics.RecordRequestError(ctx.RequestModel, "timeout")
					return nil
				}
			}

			// Handle context cancellation from the server-side context
			if errors.Is(err, context.Canceled) {
				observability.Infof("Stream canceled gracefully")
				metrics.RecordRequestError(ctx.RequestModel, "cancellation")
				return nil
			}
			if errors.Is(err, context.DeadlineExceeded) {
				observability.Infof("Stream deadline exceeded")
				metrics.RecordRequestError(ctx.RequestModel, "timeout")
				return nil
			}

			observability.Errorf("Error receiving request: %v", err)
			return err
		}

		switch v := req.Request.(type) {
		case *ext_proc.ProcessingRequest_RequestHeaders:
			response, err := r.handleRequestHeaders(v, ctx)
			if err != nil {
				observability.Errorf("handleRequestHeaders failed: %v", err)
				return err
			}
			if err := sendResponse(stream, response, "request header"); err != nil {
				observability.Errorf("sendResponse for headers failed: %v", err)
				return err
			}

		case *ext_proc.ProcessingRequest_RequestBody:
			response, err := r.handleRequestBody(v, ctx)
			if err != nil {
				observability.Errorf("handleRequestBody failed: %v", err)
				return err
			}
			if err := sendResponse(stream, response, "request body"); err != nil {
				observability.Errorf("sendResponse for body failed: %v", err)
				return err
			}

		case *ext_proc.ProcessingRequest_ResponseHeaders:
			response, err := r.handleResponseHeaders(v, ctx)
			if err != nil {
				return err
			}
			if err := sendResponse(stream, response, "response header"); err != nil {
				return err
			}

		case *ext_proc.ProcessingRequest_ResponseBody:
			response, err := r.handleResponseBody(v, ctx)
			if err != nil {
				return err
			}
			if err := sendResponse(stream, response, "response body"); err != nil {
				return err
			}

		default:
			observability.Warnf("Unknown request type: %v", v)

			// For unknown message types, create a body response with CONTINUE status
			response := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_RequestBody{
					RequestBody: &ext_proc.BodyResponse{
						Response: &ext_proc.CommonResponse{
							Status: ext_proc.CommonResponse_CONTINUE,
						},
					},
				},
			}

			if err := sendResponse(stream, response, "unknown"); err != nil {
				return err
			}
		}
	}
}
