package extproc

import (
	"encoding/json"
	"fmt"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessionbudget"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// budgetTerminateStatus is the HTTP status returned when a session is terminated
// for exceeding its token budget. 429 mirrors the congestion-control / rate-limit
// paradigm the vision paper draws on.
const budgetTerminateStatus = 429

// SessionBudgetDecision is the write-once result of evaluating a session's
// cumulative token usage against the configured budget. It is stored on
// RequestContext so downstream ladder action sites and the response-header
// writer can consult it without re-reading session telemetry.
type SessionBudgetDecision struct {
	Stage      sessionbudget.Stage
	Ratio      float64
	Cumulative int64
	Budget     int64
}

// evaluateSessionBudget compares the active session's cumulative token usage
// against the configured session_token_budget and records the resulting
// graduated stage on ctx. It is a complete no-op (no read, no metrics, no ctx
// mutation) unless enforcement is enabled, the budget is positive, and the
// request resolved to a session — preserving the opt-in, tri-state contract.
func (r *OpenAIRouter) evaluateSessionBudget(ctx *RequestContext) {
	if r == nil || r.Config == nil || ctx == nil {
		return
	}
	cfg := r.Config.SessionTokenBudget
	if !cfg.Enabled || cfg.BudgetTokens <= 0 || ctx.SessionID == "" {
		return
	}

	var cumulative int64
	if snap, ok := sessiontelemetry.GetRouterSessionSnapshot(ctx.SessionID, time.Now()); ok {
		cumulative = snap.CumulativePromptTokens + snap.CumulativeCompletionTokens
	}

	thresholds := sessionbudget.ResolveThresholds(sessionbudget.Thresholds{
		ShapeTools: cfg.Thresholds.ShapeTools,
		Compress:   cfg.Thresholds.Compress,
		Downgrade:  cfg.Thresholds.Downgrade,
		Terminate:  cfg.Thresholds.Terminate,
	})
	stage, ratio := sessionbudget.Evaluate(cumulative, cfg.BudgetTokens, thresholds)

	ctx.SessionBudget = &SessionBudgetDecision{
		Stage:      stage,
		Ratio:      ratio,
		Cumulative: cumulative,
		Budget:     cfg.BudgetTokens,
	}

	metrics.RecordSessionBudgetStage(stage.String())
	metrics.ObserveSessionBudgetRatio(ratio)
	logging.ComponentDebugEvent("extproc", "session_budget_evaluated", map[string]interface{}{
		"request_id": ctx.RequestID,
		"session_id": ctx.SessionID,
		"stage":      stage.String(),
		"ratio":      ratio,
		"cumulative": cumulative,
		"budget":     cfg.BudgetTokens,
	})
}

// maybeTerminateForBudget returns an immediate budget-exceeded response when the
// session reached the terminate stage, short-circuiting the backend the way the
// fast-response path does. It returns nil for every other stage (and when no
// budget decision was made), leaving the request to proceed normally.
func (r *OpenAIRouter) maybeTerminateForBudget(ctx *RequestContext) *ext_proc.ProcessingResponse {
	if ctx == nil || ctx.SessionBudget == nil || ctx.SessionBudget.Stage != sessionbudget.StageTerminate {
		return nil
	}
	d := ctx.SessionBudget

	message := fmt.Sprintf(
		"Session token budget exceeded: %d cumulative tokens is %.2fx the configured budget of %d.",
		d.Cumulative, d.Ratio, d.Budget,
	)
	body, _ := json.Marshal(map[string]interface{}{
		"error": map[string]interface{}{
			"message": message,
			"type":    "budget_exceeded",
			"code":    budgetTerminateStatus,
		},
	})

	resp := r.createJSONResponseWithBody(budgetTerminateStatus, body)
	appendSessionBudgetHeaders(resp.GetImmediateResponse().GetHeaders(), d, true)

	// The stage counter was already incremented in evaluateSessionBudget; emit
	// only the distinct audit log here to avoid double-counting.
	logging.ComponentWarnEvent("extproc", "session_budget_terminated", map[string]interface{}{
		"request_id": ctx.RequestID,
		"session_id": ctx.SessionID,
		"ratio":      d.Ratio,
		"cumulative": d.Cumulative,
		"budget":     d.Budget,
	})
	return resp
}

// appendSessionBudgetHeaders adds the x-vsr-budget-* headers describing a budget
// decision to a header mutation. When terminated, it also sets
// x-vsr-budget-exceeded=true.
func appendSessionBudgetHeaders(mutation *ext_proc.HeaderMutation, d *SessionBudgetDecision, terminated bool) {
	if mutation == nil || d == nil {
		return
	}
	set := func(key, value string) {
		mutation.SetHeaders = append(mutation.SetHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{Key: key, RawValue: []byte(value)},
		})
	}
	set(headers.VSRBudgetStage, d.Stage.String())
	set(headers.VSRBudgetRatio, fmt.Sprintf("%.2f", d.Ratio))
	if terminated {
		set(headers.VSRBudgetExceeded, "true")
	}
}
