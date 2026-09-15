package extproc

import (
	"fmt"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
	httputil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

// applyRateLimit preserves the existing optional data-plane limiter at the
// post-selection boundary. The selected logical model is part of the rule key,
// and actual provider usage is settled by the response pipeline.
func (r *OpenAIRouter) applyRateLimit(ctx *RequestContext, selectedModel string) *ext_proc.ProcessingResponse {
	if r == nil || r.RateLimiter == nil || ctx == nil {
		return nil
	}
	limitContext := r.buildRateLimitContext(ctx, selectedModel)
	decision, err := r.RateLimiter.Check(limitContext)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "rate_limit_check_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"model":      selectedModel,
			"error":      err.Error(),
		})
		return r.createRateLimitResponse(decision)
	}
	if decision != nil && !decision.Allowed {
		logging.ComponentEvent("extproc", "rate_limit_rejected", map[string]interface{}{
			"request_id":         ctx.RequestID,
			"model":              limitContext.Model,
			"provider":           decision.Provider,
			"remaining":          decision.Remaining,
			"user_scope_present": limitContext.UserID != "",
		})
		return r.createRateLimitResponse(decision)
	}
	ctx.RateLimitCtx = &limitContext
	return nil
}

func (r *OpenAIRouter) buildRateLimitContext(ctx *RequestContext, selectedModel string) ratelimit.Context {
	return ratelimit.Context{
		UserID:     ctx.TrustedIdentity.UserID,
		Groups:     append([]string(nil), ctx.TrustedIdentity.Groups...),
		Model:      selectedModel,
		TokenCount: ctx.VSRContextTokenCount,
	}
}

func (r *OpenAIRouter) createRateLimitResponse(decision *ratelimit.Decision) *ext_proc.ProcessingResponse {
	retryAfter := "60"
	if decision != nil && decision.RetryAfter > 0 {
		retryAfter = fmt.Sprintf("%d", int(decision.RetryAfter.Seconds()))
	}
	body := []byte(fmt.Sprintf(
		`{"error":{"message":"Rate limit exceeded. Retry after %s seconds.","type":"rate_limit_error","code":429}}`,
		retryAfter,
	))
	responseHeaders := []*core.HeaderValueOption{
		{Header: &core.HeaderValue{Key: "content-type", RawValue: []byte("application/json")}},
		{Header: &core.HeaderValue{Key: "retry-after", RawValue: []byte(retryAfter)}},
	}
	if decision != nil {
		responseHeaders = append(responseHeaders,
			&core.HeaderValueOption{Header: &core.HeaderValue{Key: "x-ratelimit-limit", RawValue: []byte(fmt.Sprintf("%d", decision.Limit))}},
			&core.HeaderValueOption{Header: &core.HeaderValue{Key: "x-ratelimit-remaining", RawValue: []byte(fmt.Sprintf("%d", decision.Remaining))}},
		)
		if !decision.ResetAt.IsZero() {
			responseHeaders = append(responseHeaders, &core.HeaderValueOption{Header: &core.HeaderValue{
				Key: "x-ratelimit-reset", RawValue: []byte(fmt.Sprintf("%d", decision.ResetAt.Unix())),
			}})
		}
	}
	responseHeaders = append(responseHeaders, httputil.KeystoneHeaderOptions(headers.ResponsePathRateLimited)...)
	return &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_ImmediateResponse{
		ImmediateResponse: &ext_proc.ImmediateResponse{
			Status:  &typev3.HttpStatus{Code: typev3.StatusCode_TooManyRequests},
			Headers: &ext_proc.HeaderMutation{SetHeaders: responseHeaders},
			Body:    body,
		},
	}}
}
