package extproc

import (
	"fmt"
	"net/http"
	"strconv"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// anthropicOverloadedStatus is the status Anthropic's API answers when it is
// overloaded. Envoy's StatusCode enum has no name for it, but Envoy sends an
// immediate response's status from 200 on as it is.
const anthropicOverloadedStatus = 529

// guardOverloadResponse answers a request that rules unread on an overloaded
// decision runtime decide (VSRGuardOverload): rules that failed closed alone
// refuse it, or one that failed open would have refused it. It gets an error
// its client retries instead of the refusal or the forward: Anthropic
// Messages gets 529 overloaded_error, as Anthropic's API sends when it is
// overloaded, and OpenAI Chat Completions and Responses get 429 with the code
// overloaded, which their SDKs and agents retry with backoff where a 5xx may
// be shown as a failure. Both carry Retry-After and x-vsr-warnings:
// guard_overloaded. The request is not forwarded; nil when the request is not
// such a one.
func (r *OpenAIRouter) guardOverloadResponse(ctx *RequestContext) *ext_proc.ProcessingResponse {
	if ctx == nil || len(ctx.VSRGuardOverload) == 0 {
		return nil
	}
	retryAfter := r.Config.SignalOverloadRetryAfter()
	message := fmt.Sprintf("the guard could not check the request because its decision model is overloaded; retry after %d seconds", retryAfter)
	status := http.StatusTooManyRequests
	ctx.ImmediateProtocolError = llmprotocol.NewError(llmprotocol.ErrorRateLimited, "overloaded", message, nil)
	if ctx.SourceFormat == llmprotocol.AnthropicMessagesV1 {
		status = anthropicOverloadedStatus
		ctx.ImmediateProtocolError = llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "overloaded_error", message, nil)
	}
	ctx.guardOverloadAnswered = true
	logging.ComponentWarnEvent("extproc", "guard_overloaded", map[string]interface{}{
		"request_id": ctx.RequestID, "rules": ctx.VSRGuardOverload, "status": status, "retry_after_s": retryAfter,
	})
	metrics.RecordSignalOverloadRetryable(string(ctx.SourceFormat))
	response := r.createErrorResponse(http.StatusTooManyRequests, message)
	response.GetImmediateResponse().Status = &typev3.HttpStatus{Code: typev3.StatusCode(status)}
	appendImmediateResponseHeader(response, "retry-after", strconv.Itoa(retryAfter))
	return response
}
