package extproc

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

// failedAttempt is a fallback candidate's failed upstream answer.
type failedAttempt struct {
	status int
	body   []byte
	format llmprotocol.WireFormat
	// transport reports that no backend answered: the connection failed,
	// was reset or timed out. failure then classifies why, when known.
	transport bool
	failure   string
}

// failedCandidateAttempt describes a response-phase candidate call that
// failed: an HTTP status from its backend, or a transport error with none.
func failedCandidateAttempt(dispatch *providerDispatch, status int, body []byte, callErr error) *failedAttempt {
	attempt := &failedAttempt{status: status, body: body, format: dispatch.targetFormat}
	if status == 0 {
		attempt.transport, attempt.body = true, nil
		attempt.status, attempt.failure = http.StatusServiceUnavailable, candidateFailure(callErr)
		if errors.Is(callErr, context.DeadlineExceeded) {
			attempt.status, attempt.failure = http.StatusGatewayTimeout, "timeout"
		}
	}
	return attempt
}

// candidateFailure names why a candidate the Router called itself did not
// answer, as pkg/upstream names a native call's: a connection that could not
// be opened, or one that broke after the request went out.
func candidateFailure(err error) string {
	var connErr *connector.Error
	if !errors.As(err, &connErr) || connErr.Kind != connector.KindTransport {
		return ""
	}
	var opErr *net.OpError
	if errors.As(err, &opErr) && opErr.Op == "dial" {
		return string(upstream.KindConnectFailure)
	}
	return string(upstream.KindReset)
}

// failedOutcomeAttempt describes a native candidate call that failed, from
// what pkg/upstream saw of it.
func failedOutcomeAttempt(candidate *fallbackCandidate, outcome routing.Outcome) *failedAttempt {
	if outcome.Local {
		return &failedAttempt{status: outcome.Status, format: candidate.dispatch.targetFormat, transport: true, failure: outcome.Failure}
	}
	return &failedAttempt{status: outcome.Status, body: failedReplyBody(outcome), format: candidate.dispatch.targetFormat}
}

// failedAttemptResponse answers the client with the last attempt of a
// fallback chain that ended without a success: the backend's error, with its
// status, in the client's protocol and terms (normalizeUpstreamClientError),
// or, when no backend answered, an unavailable
// error that keeps the transport's detail out of the client's message. No
// response phase sees that answer, so the record ends with it here.
func (r *OpenAIRouter) failedAttemptResponse(ctx *RequestContext, attempt *failedAttempt) *ext_proc.ProcessingResponse {
	if attempt == nil {
		return nil
	}
	return r.answerFailedCall(ctx, attempt.status, r.failedAttemptError(ctx, attempt), attempt.replayReason())
}

// answerFailedCall answers the client with protocolError and status for an
// upstream call that gave it no answer, and ends the record with it for
// reason.
func (r *OpenAIRouter) answerFailedCall(
	ctx *RequestContext, status int, protocolError *llmprotocol.ProtocolError, reason string,
) *ext_proc.ProcessingResponse {
	ctx.ImmediateProtocolError = protocolError
	response := r.encodeImmediateResponseForClient(r.createErrorResponse(status, protocolError.Message), ctx)
	r.failAnsweredRouterReplay(ctx, status, response.GetImmediateResponse().GetBody(), reason)
	return response
}

// replayReason is why the record of a request answered with the attempt's
// failure fails.
func (attempt *failedAttempt) replayReason() string {
	if attempt.transport {
		return "upstream_" + cmp.Or(attempt.failure, "unavailable")
	}
	return "upstream_error_response"
}

// fallbackCredentialRefusal is the request's 401 when a chain ended at a
// candidate refused for want of the request's delegated credential, and ends
// the record with it; nil when no candidate was refused.
func (r *OpenAIRouter) fallbackCredentialRefusal(ctx *RequestContext) *ext_proc.ProcessingResponse {
	refused := r.encodeImmediateResponseForClient(r.delegatedCredentialRefusal(ctx), ctx)
	immediate := refused.GetImmediateResponse()
	r.failAnsweredRouterReplay(ctx, int(immediate.GetStatus().GetCode()), immediate.GetBody(), "request_rejected")
	return refused
}

func (r *OpenAIRouter) failedAttemptError(ctx *RequestContext, attempt *failedAttempt) *llmprotocol.ProtocolError {
	if attempt.transport {
		if attempt.status == http.StatusGatewayTimeout {
			return llmprotocol.NewError(llmprotocol.ErrorUpstreamTimeout, "upstream_timeout", "the model service did not answer in time", nil)
		}
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "upstream_unavailable", "the model service is unavailable", nil)
	}
	engine, err := r.protocolEngine(ctx)
	if err != nil {
		return upstreamTransportFallback(attempt.status, err)
	}
	decoded, _, err := engine.DecodeTransportError(attempt.format, attempt.body)
	if err != nil || decoded.Error == nil {
		return upstreamTransportFallback(attempt.status, err)
	}
	_ = normalizeUpstreamClientError(ctx, attempt.status)(&decoded)
	return decoded.Error
}

// noteLengthRefusal marks a failed attempt in which the backend refused the
// prompt as longer than its window, read from the error body in format. An
// attempt without a body, such as the primary's at the response-header
// phase, counts as one when it is a 400 or 413 to a request whose count fell
// within the margin of the model's window. Fallback then goes on only to
// candidates with wider windows.
func (r *OpenAIRouter) noteLengthRefusal(
	ctx *RequestContext, attempt *fallback.AttemptOutcome, format llmprotocol.WireFormat, body []byte,
) {
	if attempt.StatusCode != http.StatusBadRequest && attempt.StatusCode != http.StatusRequestEntityTooLarge {
		return
	}
	model := attemptWindowModel(ctx)
	if len(body) == 0 {
		if _, inBand := ctx.contextInBand[model]; !inBand {
			return
		}
	} else if !r.refusedForLength(format, attempt.StatusCode, body) {
		return
	}
	attempt.ContextLengthExceeded = true
	window := r.contextWindowSize(model)
	ctx.fallbackMinWindow = max(ctx.fallbackMinWindow, window)
	logging.ComponentEvent("extproc", "context_length_refused_upstream", map[string]interface{}{
		"request_id": ctx.RequestID,
		"model":      model,
		"window":     window,
		"from_body":  len(body) > 0,
	})
}

func (r *OpenAIRouter) refusedForLength(format llmprotocol.WireFormat, status int, body []byte) bool {
	engine, err := r.protocolEngine(nil)
	if err != nil {
		return false
	}
	decoded, _, err := engine.DecodeTransportError(format, body)
	return err == nil && upstreamContextLengthError(status, decoded.Error)
}

// attemptWindowModel is the configured model of the attempt in flight, whose
// window governs it.
func attemptWindowModel(ctx *RequestContext) string {
	if ctx.VSRSelectedCandidate != nil && ctx.VSRSelectedCandidate.Model != "" {
		return strings.TrimSpace(ctx.VSRSelectedCandidate.Model)
	}
	if ctx.VSRSelectedModel != "" {
		return strings.TrimSpace(ctx.VSRSelectedModel)
	}
	return strings.TrimSpace(ctx.RequestModel)
}

// fallbackWindowSkip is why a candidate cannot serve a request that a
// backend refused for its length or that the Router's count rules out for
// the candidate's window; nil when neither applies.
func (r *OpenAIRouter) fallbackWindowSkip(ctx *RequestContext, ref *config.ModelRef) error {
	window := r.contextWindowSize(ref.Model)
	if ctx.fallbackMinWindow > 0 && window <= ctx.fallbackMinWindow {
		return fmt.Errorf("fallback candidate %q has a %d-token context window, no wider than the %d-token window whose backend refused the request for its length",
			candidateModelIdentity(*ref), window, ctx.fallbackMinWindow)
	}
	if window <= 0 {
		return nil
	}
	if tokens, kind, ok := r.cachedContextTokens(ctx, ctx.VSRSelectedDecision, ref.Model); ok && r.bandVerdict(window, tokens, kind) == contextExceeds {
		return &contextLengthError{window: window, tokens: tokens, kind: kind, scope: fmt.Sprintf("fallback candidate %q", candidateModelIdentity(*ref))}
	}
	return nil
}
