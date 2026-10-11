package extproc

import (
	"context"
	"errors"
	"io"
	"net/http"
	"runtime/debug"

	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// handleRequestBodyDispatch routes body messages to the correct handler.
//
// BUFFERED mode (default): the message goes straight to handleRequestBody.
//
// STREAMED and FULL_DUPLEX_STREAMED modes (global.router.streamed_body.enabled):
// Envoy sends multiple body messages. A StreamedBodyHandler accumulates every
// chunk and runs the full pipeline on end_of_stream. Requests that name a
// concrete model are accumulated too, because dispatch can still rewrite the
// model ID, translate the wire format, and add stream_options.include_usage.
func (r *OpenAIRouter) handleRequestBodyDispatch(v *ext_proc.ProcessingRequest_RequestBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	// Honor x-vsr-skip-processing before allocating a streamed-body handler.
	// This guarantees no chunk accumulation, model detection, or buffered
	// pipeline runs for opted-out requests, regardless of streamed_body_mode.
	if ctx.SkipProcessing {
		if ctx.FullDuplexRequestBody {
			return newFullDuplexRequestBodyResponse(v.RequestBody.GetBody(), v.RequestBody.GetEndOfStream()), nil
		}
		return newContinueRequestBodyResponse(), nil
	}

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
	if ctx.FullDuplexRequestBody && !streamedMode {
		// Envoy negotiated FULL_DUPLEX_STREAMED while streamed_body is
		// disabled. Relaying the chunks would deliver the request upstream
		// without decoding, classification, or any other routing policy, so
		// the mismatch must fail closed instead of silently bypassing the
		// router. Official Envoy templates ship BUFFERED bodies; this state
		// only arises from a hand-edited Envoy config.
		logging.ComponentWarnEvent("extproc", "full_duplex_without_streamed_body", map[string]interface{}{
			"request_id": ctx.RequestID,
		})
		return r.createErrorResponse(503, "Router streamed_body is disabled but Envoy negotiated full-duplex request bodies; fix the Envoy processing mode or enable global.router.streamed_body"), nil
	}
	// STREAMED may contain just one EOS body message; it still needs the guards.
	if streamedMode {
		ctx.StreamedBody = newStreamedBodyHandler(r, ctx)
		resp, err := ctx.StreamedBody.HandleChunk(v.RequestBody, ctx)
		if eos {
			ctx.StreamedBody.Release()
			ctx.StreamedBody = nil
		}
		return resp, err
	}

	// BUFFERED mode uses the classic pipeline.
	return r.handleRequestBody(v, ctx)
}

// Process implements the ext_proc calls
func (r *OpenAIRouter) Process(stream ext_proc.ExternalProcessor_ProcessServer) (retErr error) {
	logging.Debugf("Processing at stage [init]")
	var ctx *RequestContext

	// Recover from any panic (including OOM kills surfaced as runtime panics from
	// CGO inference calls) so a single bad request cannot take down the gRPC server.
	defer func() {
		if rec := recover(); rec != nil {
			r.finalizeRouterReplay(ctx, routerreplay.LifecycleFailed, "processor_panic")
			logging.Errorf("Process: recovered panic: %v\n%s", rec, debug.Stack())
			retErr = status.Errorf(codes.Internal, "internal error: %v", rec)
		}
		// Error and panic returns skip the receive-error cleanup, so an
		// in-flight admission taken before a dispatch error would otherwise
		// inflate the model's in-flight count until the tracker's max age.
		// Requests that ended through a normal path already zeroed the token,
		// making this a no-op.
		releaseInflightAdmission(ctx)
		if retErr != nil && ctx != nil {
			sendHeldHeaderReplyBeforeError(stream, ctx)
		}
		finishRequestTrace(ctx, retErr)
	}()

	// Initialize request context
	ctx = &RequestContext{
		Headers:       make(map[string]string),
		TraceContext:  stream.Context(),
		ConfigVersion: r.configVersion.Load(),
	}

	for {
		req, err := stream.Recv()
		if err != nil {
			// grpc-go has already sent the status after any receive error but
			// EOF, and Process sends nothing at EOF, so drop a held header reply.
			ctx.fullDuplexHold = nil
			return r.handleProcessReceiveError(ctx, err)
		}

		if err := r.handleProcessRequest(stream, req, ctx); err != nil {
			state, reason := replayLifecycleForProcessError(err)
			r.finalizeRouterReplay(ctx, state, reason)
			return err
		}
	}
}

func (r *OpenAIRouter) handleProcessReceiveError(ctx *RequestContext, err error) error {
	if !errors.Is(err, io.EOF) {
		ctx.TraceReceiveError = err
	} else if ctx.RequestSpan != nil {
		// A terminal response closes the span before the next Recv. EOF with
		// an open request therefore means the response was never completed.
		ctx.TraceReceiveError = io.ErrUnexpectedEOF
	}
	if ctx.IsStreamingResponse && !ctx.StreamingComplete {
		ctx.StreamingAborted = true
		// The evidence window is count-bounded, so a turn that never reaches EOS
		// must still land as a fact. Without it the newest failed turns cannot
		// displace older regressions and a later request could switch on
		// evidence that is no longer from the latest turns. The recorder is
		// idempotent and empty usage stays non-attributable.
		recordSessionTurnOutcome(ctx, responseUsageMetrics{})
		logging.Debugf("Streaming response aborted before completion, will not cache")
	}
	if ctx.InflightToken != 0 {
		releaseInflightAdmission(ctx)
	}

	state, reason := replayLifecycleForReceiveError(err)
	r.finalizeRouterReplay(ctx, state, reason)

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

func replayLifecycleForProcessError(err error) (string, string) {
	if errors.Is(err, context.Canceled) || status.Code(err) == codes.Canceled {
		return routerreplay.LifecycleAborted, "request_canceled"
	}
	if errors.Is(err, context.DeadlineExceeded) || status.Code(err) == codes.DeadlineExceeded {
		return routerreplay.LifecycleAborted, "request_deadline_exceeded"
	}
	return routerreplay.LifecycleFailed, "request_processing_failed"
}

func replayLifecycleForReceiveError(err error) (string, string) {
	if errors.Is(err, io.EOF) {
		return routerreplay.LifecycleAborted, "stream_ended_before_terminal_response"
	}
	if errors.Is(err, context.Canceled) || status.Code(err) == codes.Canceled {
		return routerreplay.LifecycleAborted, "client_canceled"
	}
	if errors.Is(err, context.DeadlineExceeded) || status.Code(err) == codes.DeadlineExceeded {
		return routerreplay.LifecycleAborted, "deadline_exceeded"
	}
	return routerreplay.LifecycleFailed, "extproc_receive_failed"
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
	if protocolConfig := req.GetProtocolConfig(); protocolConfig != nil {
		mode := protocolConfig.GetRequestBodyMode()
		ctx.FullDuplexRequestBody = mode == http_ext.ProcessingMode_FULL_DUPLEX_STREAMED
		ctx.BufferedRequestBody = mode == http_ext.ProcessingMode_BUFFERED || mode == http_ext.ProcessingMode_BUFFERED_PARTIAL
	}

	switch v := req.Request.(type) {
	case *ext_proc.ProcessingRequest_RequestHeaders:
		return r.processRequestHeaders(stream, v, ctx)
	case *ext_proc.ProcessingRequest_RequestBody:
		return r.processRequestBody(stream, v, ctx)
	case *ext_proc.ProcessingRequest_RequestTrailers:
		if ctx.FullDuplexRequestBody {
			return r.processRequestTrailers(stream, ctx)
		}
		return processUnknownRequest(stream, v)
	case *ext_proc.ProcessingRequest_ResponseHeaders:
		return r.processResponseHeaders(stream, req, v, ctx)
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
	response, err := r.requestHeadersReply(v, ctx)
	if err != nil {
		return err
	}
	if r.holdFullDuplexHeaderReply(v, response, ctx) {
		return nil
	}
	if err := sendResponse(stream, response, "request header"); err != nil {
		logging.Errorf("sendResponse for headers failed: %v", err)
		return err
	}
	finishImmediateResponseTrace(ctx, response)
	return nil
}

func (r *OpenAIRouter) processRequestBody(
	stream ext_proc.ExternalProcessor_ProcessServer,
	v *ext_proc.ProcessingRequest_RequestBody,
	ctx *RequestContext,
) error {
	response, err := r.handleRequestBodyDispatch(v, ctx)
	_, err = r.sendRequestBodyResult(stream, response, err, ctx)
	return err
}

// sendRequestBodyResult sends the reply to a request body message, or to the
// request trailers that end a full-duplex body, and reports whether it was an
// immediate response.
func (r *OpenAIRouter) sendRequestBodyResult(
	stream ext_proc.ExternalProcessor_ProcessServer,
	response *ext_proc.ProcessingResponse,
	err error,
	ctx *RequestContext,
) (bool, error) {
	response, err = r.requestBodyReply(response, err, ctx)
	if err != nil {
		return false, err
	}
	r.addEnvoyReliabilityHeaders(response, ctx)
	// FULL_DUPLEX_STREAMED explicitly permits the processor to buffer any
	// number of input chunks before sending a StreamedBodyResponse. A nil
	// response here means this chunk was retained for the eventual EOS reply.
	if response == nil && ctx.FullDuplexRequestBody {
		if ctx.StreamedBody == nil && ctx.fullDuplexHold != nil {
			return false, status.Error(codes.Internal, "full-duplex request body ended without a reply")
		}
		return false, nil
	}
	if err := flushHeldRequestHeaderReply(stream, response, ctx); err != nil {
		return false, err
	}
	if err := sendResponse(stream, response, "request body"); err != nil {
		logging.Errorf("sendResponse for body failed: %v", err)
		return false, err
	}
	finishImmediateResponseTrace(ctx, response)
	return response.GetImmediateResponse() != nil, nil
}

// processBodyRoutingError converts a *llmprotocol.ProtocolError raised during
// routing or dispatch into an immediate client-facing response. Capability
// mismatches (a request requiring capabilities the chosen backend wire cannot
// express, e.g. image output on chat completions) are client errors, not
// server failures; every other error keeps the caller's generic path.
func (r *OpenAIRouter) processBodyRoutingError(err error, ctx *RequestContext) (*ext_proc.ProcessingResponse, bool) {
	if err == nil {
		return nil, false
	}
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) {
		return nil, false
	}
	if ctx != nil {
		ctx.ImmediateProtocolError = protocolError
	}
	response := r.createErrorResponse(http.StatusBadRequest, protocolError.Message)
	addPromptCacheReceiptToImmediateResponse(response, ctx)
	return response, true
}

func (r *OpenAIRouter) processResponseHeaders(
	stream ext_proc.ExternalProcessor_ProcessServer,
	req *ext_proc.ProcessingRequest,
	v *ext_proc.ProcessingRequest_ResponseHeaders,
	ctx *RequestContext,
) error {
	if v != nil {
		var attributes map[string]*structpb.Struct
		if req != nil {
			attributes = req.GetAttributes()
		}
		captureUpstreamEndpointAddress(ctx, v.ResponseHeaders.GetHeaders(), attributes)
	}
	response, err := r.responseHeadersReply(v, ctx)
	if err != nil {
		return err
	}
	if err := sendResponse(stream, response, "response header"); err != nil {
		return err
	}
	responseHeadersReplySent(ctx, response, v.ResponseHeaders.GetEndOfStream())
	return nil
}

func (r *OpenAIRouter) processResponseBody(
	stream ext_proc.ExternalProcessor_ProcessServer,
	v *ext_proc.ProcessingRequest_ResponseBody,
	ctx *RequestContext,
) error {
	response, err := r.responseBodyReply(v, ctx)
	if err != nil {
		return err
	}
	if err := sendResponse(stream, response, "response body"); err != nil {
		return err
	}
	responseBodyReplySent(ctx, response, v.ResponseBody.GetEndOfStream())
	return nil
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

// releaseInflightAdmission ends the request's in-flight admission, if one is
// still open. The token and its model key travel together on the context, so
// every cleanup path — normal response completion, receive errors, dispatch
// errors, and panics — releases the exact bucket the request was admitted to
// even when the routing model was rewritten in between. Double release is a
// no-op because every caller clears the token afterwards.
func releaseInflightAdmission(ctx *RequestContext) {
	if ctx == nil || ctx.InflightToken == 0 {
		return
	}
	inflight.End(ctx.InflightModel, ctx.InflightToken)
	ctx.InflightToken = 0
}
