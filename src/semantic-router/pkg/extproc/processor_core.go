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
func (r *OpenAIRouter) Process(stream ext_proc.ExternalProcessor_ProcessServer) (retErr error) {
	logging.Debugf("Processing at stage [init]")

	// Recover from any panic (including OOM kills surfaced as runtime panics from
	// CGO inference calls) so a single bad request cannot take down the gRPC server.
	defer func() {
		if rec := recover(); rec != nil {
			logging.Errorf("Process: recovered panic: %v", rec)
			retErr = status.Errorf(codes.Internal, "internal error: %v", rec)
		}
	}()

	// Initialize request context
	ctx := &RequestContext{
		Headers: make(map[string]string),
	}

	for {
		req, err := stream.Recv()
		if err != nil {
			return r.handleProcessReceiveError(ctx, err)
		}

		if err := r.handleProcessRequest(stream, req, ctx); err != nil {
			return err
		}
	}
}

func (r *OpenAIRouter) handleProcessReceiveError(ctx *RequestContext, err error) error {
	if ctx.IsStreamingResponse && !ctx.StreamingComplete {
		ctx.StreamingAborted = true
		logging.Debugf("Streaming response aborted before completion, will not cache")
	}

	if errors.Is(err, io.EOF) {
		logging.Debugf("Stream ended gracefully")
		return nil
	}

	if handled := handleProcessStatusError(ctx, err); handled {
		return nil
	}

	if handled := handleProcessContextError(ctx, err); handled {
		return nil
	}

	logging.Errorf("Error receiving request: %v", err)
	return err
}

func handleProcessStatusError(ctx *RequestContext, err error) bool {
	s, ok := status.FromError(err)
	if !ok {
		return false
	}

	switch s.Code() {
	case codes.Canceled:
		return true
	case codes.DeadlineExceeded:
		recordProcessTimeout(ctx)
		return true
	default:
		return false
	}
}

func handleProcessContextError(ctx *RequestContext, err error) bool {
	if errors.Is(err, context.Canceled) {
		logging.Debugf("Stream canceled gracefully")
		return true
	}
	if errors.Is(err, context.DeadlineExceeded) {
		recordProcessTimeout(ctx)
		return true
	}
	return false
}

func recordProcessTimeout(ctx *RequestContext) {
	logging.Infof("Stream deadline exceeded")
	metrics.RecordRequestError(ctx.RequestModel, "timeout")
}

func (r *OpenAIRouter) handleProcessRequest(
	stream ext_proc.ExternalProcessor_ProcessServer,
	req *ext_proc.ProcessingRequest,
	ctx *RequestContext,
) error {
	switch v := req.Request.(type) {
	case *ext_proc.ProcessingRequest_RequestHeaders:
		return r.processRequestHeaders(stream, v, ctx)
	case *ext_proc.ProcessingRequest_RequestBody:
		return r.processRequestBody(stream, v, ctx)
	case *ext_proc.ProcessingRequest_ResponseHeaders:
		return r.processResponseHeaders(stream, v, ctx)
	case *ext_proc.ProcessingRequest_ResponseBody:
		return r.processResponseBody(stream, v, ctx)
	default:
		return processUnknownRequest(stream, v)
	}
}

func (r *OpenAIRouter) processRequestHeaders(
	stream ext_proc.ExternalProcessor_ProcessServer,
	v *ext_proc.ProcessingRequest_RequestHeaders,
	ctx *RequestContext,
) error {
	response, err := r.handleRequestHeaders(v, ctx)
	if err != nil {
		logging.Errorf("handleRequestHeaders failed: %v", err)
		return err
	}
	if err := sendResponse(stream, response, "request header"); err != nil {
		logging.Errorf("sendResponse for headers failed: %v", err)
		return err
	}
	return nil
}

func (r *OpenAIRouter) processRequestBody(
	stream ext_proc.ExternalProcessor_ProcessServer,
	v *ext_proc.ProcessingRequest_RequestBody,
	ctx *RequestContext,
) error {
	response, err := r.handleRequestBodyDispatch(v, ctx)
	if err != nil {
		logging.Errorf("handleRequestBody failed: %v", err)
		return err
	}
	if err := sendResponse(stream, response, "request body"); err != nil {
		logging.Errorf("sendResponse for body failed: %v", err)
		return err
	}
	return nil
}

func (r *OpenAIRouter) processResponseHeaders(
	stream ext_proc.ExternalProcessor_ProcessServer,
	v *ext_proc.ProcessingRequest_ResponseHeaders,
	ctx *RequestContext,
) error {
	response, err := r.handleResponseHeaders(v, ctx)
	if err != nil {
		return err
	}
	return sendResponse(stream, response, "response header")
}

func (r *OpenAIRouter) processResponseBody(
	stream ext_proc.ExternalProcessor_ProcessServer,
	v *ext_proc.ProcessingRequest_ResponseBody,
	ctx *RequestContext,
) error {
	response, err := r.handleResponseBody(v, ctx)
	if err != nil {
		return err
	}
	return sendResponse(stream, response, "response body")
}

func processUnknownRequest(
	stream ext_proc.ExternalProcessor_ProcessServer,
	request interface{},
) error {
	logging.Warnf("Unknown request type: %v", request)

	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
				},
			},
		},
	}

	return sendResponse(stream, response, "unknown")
}
