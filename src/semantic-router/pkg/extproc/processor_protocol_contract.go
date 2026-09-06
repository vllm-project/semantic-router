package extproc

import (
	"errors"
	"fmt"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func (r *OpenAIRouter) encodeSyntheticTextResponse(
	ctx *RequestContext,
	text string,
	streaming bool,
) ([]byte, string, *llmprotocol.Response, error) {
	if ctx == nil {
		return nil, "", nil, fmt.Errorf("request context is unavailable")
	}
	// A synthetic response terminates at the Router and is never sent to the
	// selected backend. Its wire contract therefore belongs exclusively to the
	// client-facing ingress protocol; TargetFormat must not leak into it.
	format := ctx.SourceFormat
	if format == "" {
		format = llmprotocol.OpenAIChatV1
	}
	responseID := "resp_" + ctx.RequestID
	itemID := "item_" + ctx.RequestID
	usage := authoritativeZeroUsage()
	response := &llmprotocol.Response{
		Generation: 1,
		ID:         responseID,
		CreatedAt:  time.Now().UTC(),
		Model:      ctx.RequestModel,
		Output: []llmprotocol.OutputItem{{
			ID: itemID, Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}},
		}},
		StopReason: llmprotocol.StopEndTurn,
		Usage:      usage,
	}
	engine, err := r.protocolEngine()
	if err != nil {
		return nil, "", nil, err
	}
	if !streaming {
		envelope := llmprotocol.Envelope{}
		envelope.ResponseRender.PreviousResponseID = responseObjectPreviousID(ctx)
		encoded, encodeErr := engine.EncodeResponse(format, *response, envelope)
		if encodeErr != nil {
			return nil, "", nil, encodeErr
		}
		ctx.SemanticResponse = response
		ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, encoded.Diagnostics...)
		return encoded.Body, "application/json", response, nil
	}
	body, diagnostics, err := engine.EncodeResponseStream(format, *response, llmprotocol.StreamContext{
		Context: ctx.TraceContext, Source: format, Target: format,
		Options:     clientStreamOptions(ctx),
		PublicModel: response.Model, ResponseID: response.ID,
		PreviousResponseID: responseObjectPreviousID(ctx),
	})
	if err != nil {
		return nil, "", nil, err
	}
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, diagnostics...)
	ctx.SemanticResponse = response
	return body, "text/event-stream", response, nil
}

func authoritativeZeroUsage() llmprotocol.Usage {
	zero := int64(0)
	authoritative := func() llmprotocol.TokenCount {
		return llmprotocol.TokenCount{Value: &zero, Provenance: llmprotocol.UsageAuthoritative}
	}
	return llmprotocol.Usage{
		State:         llmprotocol.UsageAvailable,
		InputUncached: authoritative(), OutputOther: authoritative(),
		InputTotal: authoritative(), OutputTotal: authoritative(), Total: authoritative(),
	}
}

func (r *OpenAIRouter) protocolEngine() (*protocolcodec.Engine, error) {
	if r == nil {
		return nil, fmt.Errorf("protocol runtime is unavailable")
	}
	registry := r.ProtocolCodecs
	if registry == nil {
		registry = protocolcodec.NewBuiltinRegistry()
	}
	return protocolcodec.NewEngine(registry, llmprotocol.DefaultPolicy())
}

// prepareProtocolRequest decodes every public wire format exactly once. The
// neutral request is the sole mutable request contract after this boundary.
func (r *OpenAIRouter) prepareProtocolRequest(
	body []byte,
	ctx *RequestContext,
) (*llmprotocol.Request, *ext_proc.ProcessingResponse) {
	if ctx.SourceFormat == "" {
		ctx.SourceFormat = llmprotocol.OpenAIChatV1
	}
	engine, err := r.protocolEngine()
	if err != nil {
		return nil, r.createErrorResponse(503, "protocol runtime unavailable")
	}
	request, envelope, diagnostics, err := engine.DecodeRequestForMutation(ctx.SourceFormat, body)
	if err != nil {
		recordIngressProtocolError(ctx, err)
		var protocolError *llmprotocol.ProtocolError
		if errors.As(err, &protocolError) {
			copy := *protocolError
			ctx.ImmediateProtocolError = &copy
		}
		return nil, r.createErrorResponse(400, "invalid inference request")
	}
	request.Trusted.SourceFormat = ctx.SourceFormat
	request.Trusted.CorrelationID = ctx.RequestID
	ctx.IngressBodyBytes = len(body)
	ctx.SemanticRequest = &request
	ctx.ProtocolEnvelope = envelope
	ctx.ProtocolDiagnostics = append(llmprotocol.Diagnostics(nil), diagnostics...)
	ctx.ExpectStreamingResponse = ctx.ExpectStreamingResponse || request.Stream
	if ctx.SourceFormat == llmprotocol.OpenAIResponsesV1 {
		state, stateErr := r.ResponseAPIFilter.PrepareObjectState(ctx.TraceContext, request, body)
		if stateErr != nil {
			var protocolError *llmprotocol.ProtocolError
			if errors.As(stateErr, &protocolError) {
				copy := *protocolError
				ctx.ImmediateProtocolError = &copy
				return nil, r.createErrorResponse(responseObjectStateHTTPStatus(protocolError), protocolError.Message)
			}
			return nil, r.createErrorResponse(503, "retained response history is unavailable")
		}
		ctx.ResponseObjectState = state
	}
	populateSessionTransitionFields(ctx)
	return &request, nil
}

func responseObjectStateHTTPStatus(protocolError *llmprotocol.ProtocolError) int {
	if protocolError != nil && protocolError.Category == llmprotocol.ErrorNotFound {
		return 404
	}
	return 503
}

func recordIngressProtocolError(ctx *RequestContext, err error) {
	model := ""
	if ctx != nil {
		model = ctx.RequestModel
	}
	reason := "invalid_request"
	var protocolError *llmprotocol.ProtocolError
	if errors.As(err, &protocolError) {
		switch protocolError.Code {
		case "body_limit", "duplicate_json_field", "invalid_json", "trailing_json":
			reason = "parse_error"
		}
	}
	metrics.RecordRequestError(model, reason)
	metrics.RecordModelRequest(model)
}

func (r *OpenAIRouter) encodeDispatchRequest(ctx *RequestContext) ([]byte, error) {
	if ctx == nil || ctx.SemanticRequest == nil {
		return nil, fmt.Errorf("neutral inference request is unavailable")
	}
	engine, err := r.protocolEngine()
	if err != nil {
		return nil, err
	}
	format := ctx.TargetFormat
	if format == "" {
		format = ctx.SourceFormat
	}
	if format == "" {
		format = llmprotocol.OpenAIChatV1
	}
	dispatchRequest := *ctx.SemanticRequest
	if format == llmprotocol.OpenAIChatV1 && dispatchRequest.Stream &&
		!streamUsageAlreadyRequested(dispatchRequest.StreamOptions) {
		// The Router always asks Chat backends for the final usage chunk so
		// accounting observes authoritative tokens. Public stream rendering uses
		// the original client preference retained in ctx.SemanticRequest.
		includeUsage := true
		dispatchRequest.StreamOptions.IncludeUsage = &includeUsage
		// Source bytes no longer describe the dispatch request, so retire the
		// replay claim on them. Without this the encoder forwards the original
		// client bytes and the forced flag never reaches the backend.
		dispatchRequest.Generation++
	}
	encoded, err := engine.EncodeRequest(format, dispatchRequest, ctx.ProtocolEnvelope)
	if err != nil {
		return nil, err
	}
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, encoded.Diagnostics...)
	return encoded.Body, nil
}

func streamUsageAlreadyRequested(options llmprotocol.StreamOptions) bool {
	return options.IncludeUsage != nil && *options.IncludeUsage
}

func clientStreamOptions(ctx *RequestContext) llmprotocol.StreamOptions {
	if ctx == nil || ctx.SemanticRequest == nil {
		return llmprotocol.StreamOptions{}
	}
	return ctx.SemanticRequest.StreamOptions
}

func (r *OpenAIRouter) decodeClientResponse(
	body []byte,
	ctx *RequestContext,
) (*llmprotocol.Response, error) {
	if ctx == nil {
		return nil, fmt.Errorf("request context is unavailable")
	}
	engine, err := r.protocolEngine()
	if err != nil {
		return nil, err
	}
	source, target := responseWireFormats(ctx)
	var mutation protocolcodec.ResponseMutation
	if responseID := responseObjectPublicID(ctx); responseID != "" {
		mutation = func(response *llmprotocol.Response) error {
			response.ID = responseID
			return nil
		}
	}
	decoded, err := engine.TranslateResponse(source, target, body, mutation)
	if err != nil {
		return nil, err
	}
	ctx.SemanticResponse = &decoded.Response
	ctx.ResponseEnvelope = decoded.Envelope
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, decoded.Diagnostics...)
	return ctx.SemanticResponse, nil
}

// decodeCachedClientResponse decodes the cache's public response contract.
// Cache partitions include the ingress protocol and cache writes persist the
// client-facing buffered body, so the selected backend format must not affect
// replay decoding.
func (r *OpenAIRouter) decodeCachedClientResponse(
	body []byte,
	ctx *RequestContext,
) (*llmprotocol.Response, error) {
	if ctx == nil {
		return nil, fmt.Errorf("request context is unavailable")
	}
	engine, err := r.protocolEngine()
	if err != nil {
		return nil, err
	}
	format := ctx.SourceFormat
	if format == "" {
		format = llmprotocol.OpenAIChatV1
	}
	decoded, err := engine.TranslateResponse(format, format, body, nil)
	if err != nil {
		return nil, err
	}
	ctx.SemanticResponse = &decoded.Response
	ctx.ResponseEnvelope = decoded.Envelope
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, decoded.Diagnostics...)
	return ctx.SemanticResponse, nil
}

func (r *OpenAIRouter) encodeClientResponse(
	response llmprotocol.Response,
	ctx *RequestContext,
) ([]byte, error) {
	engine, err := r.protocolEngine()
	if err != nil {
		return nil, err
	}
	format := llmprotocol.OpenAIChatV1
	envelope := llmprotocol.Envelope{}
	if ctx != nil {
		if ctx.SourceFormat != "" {
			format = ctx.SourceFormat
		}
		envelope = ctx.ResponseEnvelope
		envelope.ResponseRender.PreviousResponseID = responseObjectPreviousID(ctx)
	}
	encoded, err := engine.EncodeResponse(format, response, envelope)
	if err != nil {
		return nil, err
	}
	if ctx != nil {
		ctx.SemanticResponse = &encoded.Response
		ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, encoded.Diagnostics...)
	}
	return encoded.Body, nil
}

func requestWirePath(format llmprotocol.WireFormat) string {
	switch format {
	case llmprotocol.OpenAIResponsesV1:
		return "/v1/responses"
	case llmprotocol.AnthropicMessagesV1:
		return "/v1/messages"
	default:
		return "/v1/chat/completions"
	}
}

//nolint:cyclop // Immediate responses exhaustively normalize the supported body and protocol combinations.
func (r *OpenAIRouter) encodeImmediateResponseForClient(
	response *ext_proc.ProcessingResponse,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if response == nil || ctx == nil || ctx.ImmediateResponseEncoded {
		return response
	}
	immediate := response.GetImmediateResponse()
	if immediate == nil || len(immediate.Body) == 0 {
		return response
	}
	format := ctx.SourceFormat
	if format == "" {
		format = llmprotocol.OpenAIChatV1
	}
	engine, err := r.protocolEngine()
	if err != nil {
		return response
	}
	status := int(immediate.GetStatus().GetCode())
	if status >= 400 {
		protocolError := immediateProtocolError(status)
		if ctx.ImmediateProtocolError != nil {
			protocolError = ctx.ImmediateProtocolError
		}
		if body, encodeErr := engine.EncodeError(format, protocolError); encodeErr == nil {
			immediate.Body = body
			setImmediateContentType(immediate, "application/json")
			ctx.ImmediateResponseEncoded = true
		}
		return response
	}
	if ctx.SemanticRequest == nil {
		// Successful management/object endpoints are not inference responses and
		// therefore do not participate in the LLM wire matrix.
		ctx.ImmediateResponseEncoded = true
		return response
	}

	// Every successful short-circuit must be created from neutral semantics or
	// explicitly mark an object/management response as already encoded. Treat a
	// missed producer as an internal contract violation instead of guessing that
	// its body is Chat Completions.
	protocolError := llmprotocol.NewError(llmprotocol.ErrorInternal, "unencoded_response", "request failed", nil)
	if body, encodeErr := engine.EncodeError(format, protocolError); encodeErr == nil {
		immediate.Body = body
		if immediate.Status == nil {
			immediate.Status = &typev3.HttpStatus{}
		}
		immediate.Status.Code = statusCodeToEnum(500)
		setImmediateContentType(immediate, "application/json")
		ctx.ImmediateResponseEncoded = true
	}
	return response
}

func immediateProtocolError(status int) *llmprotocol.ProtocolError {
	category, code, message := llmprotocol.ErrorInternal, "request_failed", "request failed"
	switch status {
	case 400, 413, 422:
		category, code, message = llmprotocol.ErrorInvalidRequest, "invalid_request", "invalid inference request"
	case 401:
		category, code, message = llmprotocol.ErrorAuthentication, "authentication_failed", "authentication failed"
	case 403:
		category, code, message = llmprotocol.ErrorPermission, "permission_denied", "permission denied"
	case 404:
		category, code, message = llmprotocol.ErrorNotFound, "not_found", "resource not found"
	case 405, 409:
		category, code, message = llmprotocol.ErrorConflict, "request_conflict", "request cannot be completed"
	case 429:
		category, code, message = llmprotocol.ErrorRateLimited, "rate_limited", "rate limit exceeded"
	case 499:
		category, code, message = llmprotocol.ErrorInternal, "request_canceled", "request canceled"
	case 502, 503, 504:
		category, code, message = llmprotocol.ErrorUpstreamUnavailable, "upstream_unavailable", "model service unavailable"
	}
	return llmprotocol.NewError(category, code, message, nil)
}

func setImmediateContentType(response *ext_proc.ImmediateResponse, value string) {
	if response.Headers == nil {
		response.Headers = &ext_proc.HeaderMutation{}
	}
	for _, option := range response.Headers.SetHeaders {
		if option != nil && option.Header != nil && option.Header.Key == "content-type" {
			option.Header.Value = ""
			option.Header.RawValue = []byte(value)
			return
		}
	}
	response.Headers.SetHeaders = append(
		response.Headers.SetHeaders,
		newHeaderValueOption("content-type", value),
	)
}
