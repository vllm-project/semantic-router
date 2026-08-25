package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/outcomefeedback"
)

const publicOutcomeFeedbackPath = "/v1/router/outcomes"

type OutcomeFeedbackRuntime interface {
	Submit(context.Context, outcomefeedback.Caller, string, outcomefeedback.Request) (outcomefeedback.Receipt, error)
}

type OutcomeLearningProjectionRuntime interface {
	Read(context.Context, string) (outcomefeedback.Projection, error)
}

type outcomeFeedbackRequestState struct {
	idempotencyKey string
	caller         outcomefeedback.Caller
	body           []byte
}

func (r *OpenAIRouter) handleOutcomeFeedbackRequestHeaders(
	method string,
	path string,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if normalizeRequestPath(path) != publicOutcomeFeedbackPath {
		return nil
	}
	if method != http.MethodPost {
		return r.createErrorResponse(http.StatusMethodNotAllowed, "method not allowed")
	}
	if r == nil || r.OutcomeFeedback == nil || !r.nativeAccessEnabled() ||
		ctx == nil || ctx.InferenceAccess == nil {
		return r.outcomeFeedbackError(http.StatusNotFound, "outcome endpoint not found", 0)
	}
	idempotencyKey := headerValueCI(ctx, "idempotency-key")
	if err := outcomefeedback.ValidateIdempotencyKey(idempotencyKey); err != nil {
		return r.outcomeFeedbackError(http.StatusBadRequest, "invalid outcome request", 0)
	}
	source := outcomefeedback.SourceAPIKey
	switch ctx.InferenceAccess.source {
	case accessruntime.AuthenticationSourceAPIKey:
	case accessruntime.AuthenticationSourceDelegated:
		source = outcomefeedback.SourceDelegated
	default:
		return r.outcomeFeedbackError(http.StatusUnauthorized, "invalid or missing API key", 0)
	}
	tenant := ctx.InferenceAccess.tenant
	caller := outcomefeedback.Caller{
		NamespaceID: tenant.NamespaceID, APIKeyID: tenant.APIKeyID,
		UserID: tenant.UserID, TeamID: tenant.TeamID, Source: source,
	}
	if err := caller.Validate(); err != nil {
		return r.outcomeFeedbackError(http.StatusUnauthorized, "invalid or missing API key", 0)
	}
	ctx.OutcomeFeedback = &outcomeFeedbackRequestState{
		idempotencyKey: idempotencyKey,
		caller:         caller,
		body:           make([]byte, 0, 1024),
	}
	mutation := buildIdentityEncodingRequestMutation(true)
	mutation.RemoveHeaders = append(mutation.RemoveHeaders, "idempotency-key")
	return newContinueRequestHeadersResponse(mutation)
}

func (r *OpenAIRouter) handleOutcomeFeedbackRequestBody(
	v *ext_proc.ProcessingRequest_RequestBody,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, bool) {
	if ctx == nil || ctx.OutcomeFeedback == nil {
		return nil, false
	}
	state := ctx.OutcomeFeedback
	chunk := v.RequestBody.GetBody()
	if len(state.body)+len(chunk) > outcomefeedback.MaximumBodyBytes {
		ctx.OutcomeFeedback = nil
		return r.outcomeFeedbackError(http.StatusRequestEntityTooLarge, "outcome request is too large", 0), true
	}
	state.body = append(state.body, chunk...)
	if !v.RequestBody.GetEndOfStream() {
		if ctx.FullDuplexRequestBody {
			return nil, true
		}
		return newContinueRequestBodyResponse(), true
	}
	ctx.OutcomeFeedback = nil
	request, err := outcomefeedback.DecodeRequest(state.body)
	if err != nil {
		return r.outcomeFeedbackError(http.StatusBadRequest, "invalid outcome request", 0), true
	}
	receipt, err := r.OutcomeFeedback.Submit(ctx.TraceContext, state.caller, state.idempotencyKey, request)
	if err != nil {
		return r.outcomeFeedbackResponseForError(err), true
	}
	status := http.StatusCreated
	if receipt.Duplicate {
		status = http.StatusOK
	}
	payload, marshalErr := json.Marshal(receipt)
	if marshalErr != nil {
		return r.outcomeFeedbackError(http.StatusServiceUnavailable, "outcome feedback is temporarily unavailable", 0), true
	}
	return r.createJSONResponseWithBody(status, payload, ""), true
}

func (r *OpenAIRouter) outcomeFeedbackResponseForError(err error) *ext_proc.ProcessingResponse {
	switch {
	case errors.Is(err, outcomefeedback.ErrInvalid):
		return r.outcomeFeedbackError(http.StatusBadRequest, "invalid outcome request", 0)
	case errors.Is(err, outcomefeedback.ErrNotFound):
		return r.outcomeFeedbackError(http.StatusNotFound, "inference replay not found", 0)
	case errors.Is(err, outcomefeedback.ErrIdempotencyConflict):
		return r.outcomeFeedbackError(http.StatusConflict, "idempotency key was already used", 0)
	case errors.Is(err, outcomefeedback.ErrRateLimited):
		var limited *outcomefeedback.RateLimitError
		if errors.As(err, &limited) {
			return r.outcomeFeedbackError(http.StatusTooManyRequests, "outcome feedback rate limit exceeded", limited.RetryAfter)
		}
		return r.outcomeFeedbackError(http.StatusTooManyRequests, "outcome feedback rate limit exceeded", time.Second)
	default:
		return r.outcomeFeedbackError(http.StatusServiceUnavailable, "outcome feedback is temporarily unavailable", 0)
	}
}

func (r *OpenAIRouter) outcomeFeedbackError(status int, message string, retryAfter time.Duration) *ext_proc.ProcessingResponse {
	payload, _ := json.Marshal(map[string]any{
		"error": map[string]string{"message": message, "type": "outcome_feedback_error"},
	})
	response := r.createJSONResponseWithBody(status, payload, "")
	if retryAfter > 0 {
		seconds := int64(math.Ceil(retryAfter.Seconds()))
		if seconds < 1 {
			seconds = 1
		}
		response.GetImmediateResponse().Headers.SetHeaders = append(
			response.GetImmediateResponse().Headers.SetHeaders,
			retryAfterHeader(seconds),
		)
	}
	return response
}
