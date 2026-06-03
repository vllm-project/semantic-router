package extproc

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessionbudget"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

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
