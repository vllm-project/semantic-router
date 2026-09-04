package extproc

import (
	"context"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/shadow"
)

// dispatchShadowArms replays the normalized request to every configured shadow
// arm inside a bounded, fire-and-forget goroutine. It never blocks or fails the
// primary response: the primary route has already decided its destination when
// this runs, and every arm outcome is observed then discarded on error.
func (r *OpenAIRouter) dispatchShadowArms(reqCtx *RequestContext) {
	if r == nil || reqCtx == nil || reqCtx.SemanticRequest == nil {
		return
	}
	cfg := r.Config.ShadowComparison
	if !cfg.IsEnabled() {
		return
	}

	go func() {
		// A panic in the shadow path must never take down the primary
		// request or the router process.
		defer func() {
			if rec := recover(); rec != nil {
				logging.ComponentErrorEvent("extproc", "shadow_dispatch_panic", map[string]interface{}{
					"request_id": reqCtx.RequestID,
					"panic":      rec,
				})
			}
		}()

		parent := reqCtx.TraceContext
		if parent == nil {
			parent = context.Background()
		}
		// The aggregate shadow window bounds total shadow work; cancelling the
		// stream context (client disconnect, request deadline) also cancels
		// every in-flight arm.
		ctx, cancel := context.WithTimeout(parent, cfg.GetMaxWait())
		defer cancel()

		params, ok := r.shadowRequestParams(reqCtx)
		if !ok {
			return
		}
		// C1: key resolver wiring lands with the aggregate-budget milestone;
		// arms are evaluated without Authorization today.
		results := shadow.Dispatch(ctx, cfg, params, nil)
		for _, res := range results {
			fields := map[string]interface{}{
				"request_id":        reqCtx.RequestID,
				"arm":               res.Arm,
				"model":             res.Model,
				"ok":                res.Outcome == shadow.OutcomeCompleted,
				"outcome":           string(res.Outcome),
				"latency_ms":        res.LatencyMS,
				"prompt_tokens":     res.PromptTokens,
				"completion_tokens": res.CompletionTokens,
			}
			if res.Outcome == shadow.OutcomeCompleted {
				logging.ComponentEvent("extproc", "shadow_arm_result", fields)
			} else {
				fields["error"] = res.Err
				logging.ComponentEvent("extproc", "shadow_arm_failed", fields)
			}
		}
	}()
}

// shadowRequestParams serializes the single normalized semantic request into
// the OpenAI-compatible params shared by every arm, so normalized inputs stay
// byte-identical across arms and aligned with the primary route (same codec
// encode + parse path as looper execution).
func (r *OpenAIRouter) shadowRequestParams(reqCtx *RequestContext) (*openai.ChatCompletionNewParams, bool) {
	engine, err := r.protocolEngine()
	if err != nil {
		logging.ComponentWarnEvent("extproc", "shadow_encode_engine_unavailable", map[string]interface{}{
			"request_id": reqCtx.RequestID,
			"error":      err.Error(),
		})
		return nil, false
	}
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, *reqCtx.SemanticRequest, llmprotocol.Envelope{})
	if err != nil {
		logging.ComponentWarnEvent("extproc", "shadow_encode_failed", map[string]interface{}{
			"request_id": reqCtx.RequestID,
			"error":      err.Error(),
		})
		return nil, false
	}
	params, err := parseOpenAIRequest(encoded.Body)
	if err != nil {
		logging.ComponentWarnEvent("extproc", "shadow_parse_failed", map[string]interface{}{
			"request_id": reqCtx.RequestID,
			"error":      err.Error(),
		})
		return nil, false
	}
	return params, true
}
