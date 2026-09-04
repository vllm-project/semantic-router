package extproc

import (
	"context"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
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
		// C1/C2: arms evaluate the same normalized request under the budget.
		// C3: a blinded judge compares the surviving arms (disabled by default).
		// C4: the comparison evidence lands in Replay via the outcome channel.
		results := shadow.Dispatch(ctx, cfg, params, nil)
		r.observeShadowArms(reqCtx, results)
		decision := r.judgeShadowArms(ctx, cfg, params, results, reqCtx)
		r.recordShadowEvidence(reqCtx, cfg, results, decision)
	}()
}

// observeShadowArms logs one observability event per arm; the primary response
// was already decided and is never touched here.
func (r *OpenAIRouter) observeShadowArms(reqCtx *RequestContext, results []shadow.ArmResult) {
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
}

// judgeShadowArms runs the blinded comparison when a judge is configured.
// All judge failures map to an explicit judge outcome and stay in
// observability/Replay; they never affect the primary response. A zero-value
// decision (empty outcome) means no judge is configured.
func (r *OpenAIRouter) judgeShadowArms(
	ctx context.Context,
	cfg config.ShadowComparisonConfig,
	params *openai.ChatCompletionNewParams,
	results []shadow.ArmResult,
	reqCtx *RequestContext,
) shadow.JudgeDecision {
	if !cfg.Judge.Enabled || cfg.Judge.Model == "" || cfg.Judge.Endpoint == "" {
		return shadow.JudgeDecision{}
	}
	judge := shadow.NewJudge(cfg.Judge, cfg.Arms)
	question, _ := extractUserAndNonUserContent(params)
	decision := judge.Decide(ctx, question, results)
	fields := map[string]interface{}{
		"request_id":     reqCtx.RequestID,
		"judge_outcome":  string(decision.Outcome),
		"winner_arm_id":  decision.WinnerArmID,
		"judge_model":    decision.JudgeModel,
		"rubric_version": decision.JudgeRubricVersion,
		"latency_ms":     decision.LatencyMS,
	}
	if len(decision.TieArmIDs) > 0 {
		fields["tie_arm_ids"] = decision.TieArmIDs
	}
	if decision.Reason != "" {
		fields["reason"] = decision.Reason
	}
	logging.ComponentEvent("extproc", "shadow_judge_result", fields)
	return decision
}

// recordShadowEvidence attaches the comparison evidence to the replay record
// via the outcome channel. Only opaque arm ids and compact features (outcome,
// latency, usage) are stored — no response content and no model identity, per
// the issue's non-goals and D7. The opaque->model mapping stays in the
// deployment config; arm identity lives in observability. Judge disabled
// still records the arms with a "observed" verdict.
func (r *OpenAIRouter) recordShadowEvidence(
	reqCtx *RequestContext,
	cfg config.ShadowComparisonConfig,
	results []shadow.ArmResult,
	decision shadow.JudgeDecision,
) {
	if r.ReplayRecorder == nil || reqCtx.RouterReplayID == "" {
		return
	}
	judge := shadow.NewJudge(cfg.Judge, cfg.Arms)
	outcome := buildShadowEvidence(judge, results, decision)
	// Reuse the same guard the learning runtime applies: only attach to a
	// record that still exists.
	if _, found := r.ReplayRecorder.GetRecord(reqCtx.RouterReplayID); !found {
		return
	}
	if err := r.ReplayRecorder.AppendOutcome(reqCtx.RouterReplayID, outcome); err != nil {
		logging.ComponentEvent("extproc", "shadow_evidence_failed", map[string]interface{}{
			"request_id": reqCtx.RequestID,
			"replay_id":  reqCtx.RouterReplayID,
			"error":      err.Error(),
		})
	}
}

// buildShadowEvidence packs arm results and the (optional) judge decision into
// a replay Outcome. The reusable evidence logic (blind ids, compact features,
// verdict mapping) lives in shadow.Judge.Evidence; this layer only wraps it
// into the replay outcome shape.
func buildShadowEvidence(judge *shadow.Judge, results []shadow.ArmResult, decision shadow.JudgeDecision) routerreplay.Outcome {
	verdict, metadata := judge.Evidence(results, decision)
	return routerreplay.Outcome{
		Timestamp: time.Now().UTC(),
		Source:    "shadow_comparison",
		Verdict:   verdict,
		Reason:    decision.Reason,
		Metadata:  metadata,
	}
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
