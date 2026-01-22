package extproc

import (
	"context"
	"errors"
	"io"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// Process implements the ext_proc calls
func (r *OpenAIRouter) Process(stream ext_proc.ExternalProcessor_ProcessServer) error {
	logging.Infof("Processing at stage [init]")

	// Initialize request context
	ctx := &RequestContext{
		Headers: make(map[string]string),
	}

	for {
		req, err := stream.Recv()
		if err != nil {
			// Mark streaming as aborted if it was a streaming response
			// This prevents caching incomplete responses
			if ctx.IsStreamingResponse && !ctx.StreamingComplete {
				ctx.StreamingAborted = true
				logging.Infof("Streaming response aborted before completion, will not cache")
			}

			// Handle EOF - this indicates the client has closed the stream gracefully
			if errors.Is(err, io.EOF) {
				logging.Infof("Stream ended gracefully")
				return nil
			}

			// Handle gRPC status-based cancellations/timeouts
			if s, ok := status.FromError(err); ok {
				switch s.Code() {
				case codes.Canceled:
					return nil
				case codes.DeadlineExceeded:
					logging.Infof("Stream deadline exceeded")
					metrics.RecordRequestError(ctx.RequestModel, "timeout")
					return nil
				}
			}

			// Handle context cancellation from the server-side context
			if errors.Is(err, context.Canceled) {
				logging.Infof("Stream canceled gracefully")
				return nil
			}
			if errors.Is(err, context.DeadlineExceeded) {
				logging.Infof("Stream deadline exceeded")
				metrics.RecordRequestError(ctx.RequestModel, "timeout")
				return nil
			}

			logging.Errorf("Error receiving request: %v", err)
			return err
		}

		switch v := req.Request.(type) {
		case *ext_proc.ProcessingRequest_RequestHeaders:
			response, err := r.handleRequestHeaders(v, ctx)
			if err != nil {
				logging.Errorf("handleRequestHeaders failed: %v", err)
				return err
			}
			if err := sendResponse(stream, response, "request header"); err != nil {
				logging.Errorf("sendResponse for headers failed: %v", err)
				return err
			}

		case *ext_proc.ProcessingRequest_RequestBody:
			response, err := r.handleRequestBody(v, ctx)
			if err != nil {
				logging.Errorf("handleRequestBody failed: %v", err)
				return err
			}
			logging.Infof("Modified request body: %+v", response)
			if err := sendResponse(stream, response, "request body"); err != nil {
				logging.Errorf("sendResponse for body failed: %v", err)
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
			logging.Warnf("Unknown request type: %v", v)

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
