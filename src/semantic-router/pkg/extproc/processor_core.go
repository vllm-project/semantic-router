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

// handleRequestBodyDispatch routes body messages to the correct handler.
//
// BUFFERED mode (default): the message goes straight to handleRequestBody.
//
// STREAMED mode (streamed_body_mode: true in config): Envoy sends multiple
// body messages. A StreamedBodyHandler accumulates chunks, detects the model
// from the first few KB, and either passes through or accumulates for the
// full pipeline on end_of_stream.
func (r *OpenAIRouter) handleRequestBodyDispatch(v *ext_proc.ProcessingRequest_RequestBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	eos := v.RequestBody.GetEndOfStream()

	// If we already have a handler from a previous chunk, continue streaming
	if ctx.StreamedBody != nil {
		resp, err := ctx.StreamedBody.HandleChunk(v.RequestBody, ctx)
		if eos {
			ctx.StreamedBody.Release()
			ctx.StreamedBody = nil
		}
		return resp, err
	}

	// Decide mode based on config: only use streaming handler when explicitly enabled
	streamedMode := r.Config != nil && r.Config.StreamedBodyMode
	if streamedMode && !eos {
		ctx.StreamedBody = newStreamedBodyHandler(r, ctx)
		return ctx.StreamedBody.HandleChunk(v.RequestBody, ctx)
	}

	// BUFFERED mode or single-message STREAMED — use classic pipeline
	return r.handleRequestBody(v, ctx)
}

// Process implements the ext_proc calls
func (r *OpenAIRouter) Process(stream ext_proc.ExternalProcessor_ProcessServer) error {
	logging.Debugf("Processing at stage [init]")

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
					logging.Debugf("Streaming response aborted before completion, will not cache")
				}

				// Handle EOF - this indicates the client has closed the stream gracefully
				if errors.Is(err, io.EOF) {
					logging.Debugf("Stream ended gracefully")
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
				logging.Debugf("Stream canceled gracefully")
				return nil
			}
			if errors.Is(err, context.DeadlineExceeded) {
				logging.Debugf("Stream deadline exceeded")
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
			response, err := r.handleRequestBodyDispatch(v, ctx)
			if err != nil {
				logging.Errorf("handleRequestBody failed: %v", err)
				return err
			}
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
