package extproc

import (
	"context"
	"errors"
	"io"
	"log"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/metrics"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Process implements the ext_proc calls
func (r *OpenAIRouter) Process(stream ext_proc.ExternalProcessor_ProcessServer) error {
	log.Println("Started processing a new request")

	// Initialize request context
	ctx := &RequestContext{
		Headers: make(map[string]string),
	}

	for {
		req, err := stream.Recv()
		if err != nil {
			// Handle EOF - this indicates the client has closed the stream gracefully
			if err == io.EOF {
				log.Println("Stream ended gracefully")
				return nil
			}

			// Handle gRPC status-based cancellations/timeouts
			if s, ok := status.FromError(err); ok {
				switch s.Code() {
				case codes.Canceled:
					log.Println("Stream canceled gracefully")
					metrics.RecordRequestError(ctx.RequestModel, "cancellation")
					return nil
				case codes.DeadlineExceeded:
					log.Println("Stream deadline exceeded")
					metrics.RecordRequestError(ctx.RequestModel, "timeout")
					return nil
				}
			}

			// Handle context cancellation from the server-side context
			if errors.Is(err, context.Canceled) {
				log.Println("Stream canceled gracefully")
				metrics.RecordRequestError(ctx.RequestModel, "cancellation")
				return nil
			}
			if errors.Is(err, context.DeadlineExceeded) {
				log.Println("Stream deadline exceeded")
				metrics.RecordRequestError(ctx.RequestModel, "timeout")
				return nil
			}

			log.Printf("Error receiving request: %v", err)
			return err
		}

		switch v := req.Request.(type) {
		case *ext_proc.ProcessingRequest_RequestHeaders:
			response, err := r.handleRequestHeaders(v, ctx)
			if err != nil {
				log.Printf("ERROR: handleRequestHeaders failed: %v", err)
				return err
			}
			if err := sendResponse(stream, response, "request header"); err != nil {
				log.Printf("ERROR: sendResponse for headers failed: %v", err)
				return err
			}

		case *ext_proc.ProcessingRequest_RequestBody:
			log.Printf("DEBUG: Processing Request Body - THIS IS WHERE ROUTING HAPPENS")

			response, err := r.handleRequestBody(v, ctx)
			if err != nil {
				log.Printf("ERROR: handleRequestBody failed: %v", err)
				return err
			}
			if err := sendResponse(stream, response, "request body"); err != nil {
				log.Printf("ERROR: sendResponse for body failed: %v", err)
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
			log.Printf("Unknown request type: %v", v)

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
