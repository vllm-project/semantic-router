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
		encoded, encodeErr := engine.EncodeResponse(format, *response, llmprotocol.Envelope{})
		if encodeErr != nil {
			return nil, "", nil, encodeErr
		}
		ctx.SemanticResponse = response
		ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, encoded.Diagnostics...)
		return encoded.Body, "application/json", response, nil
	}
	body, diagnostics, err := engine.EncodeResponseStream(format, *response, llmprotocol.StreamContext{
		Context: ctx.TraceContext, Source: format, Target: format,
		PublicModel: response.Model, ResponseID: response.ID,
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
	request, envelope, diagnostics, err := engine.DecodeRequest(ctx.SourceFormat, body)
	if err != nil {
		recordIngressProtocolError(ctx, err)
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
		owner, _ := r.responseObjectOwner(ctx)
		if r.ResponseAPIFilter != nil {
			ctx.ResponseObjectState = r.ResponseAPIFilter.PrepareObjectState(ctx.TraceContext, owner, request, body)
		}
	}
	populateSessionTransitionFields(ctx)
	return &request, nil
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
	format := ctx.SourceFormat
	if format == "" {
		format = llmprotocol.OpenAIChatV1
	}
	encoded, err := engine.EncodeRequest(format, *ctx.SemanticRequest, ctx.ProtocolEnvelope)
	if err != nil {
		return nil, err
	}
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, encoded.Diagnostics...)
	return encoded.Body, nil
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
		if body, encodeErr := engine.EncodeError(format, immediateProtocolError(status)); encodeErr == nil {
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
