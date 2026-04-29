package extproc

import (
	"math"
	"slices"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func (r *OpenAIRouter) applyModelSwitchGate(
	selCtx *selection.SelectionContext,
	result *selection.SelectionResult,
	selectedModelRef *config.ModelRef,
	ctx *RequestContext,
) (*config.ModelRef, bool) {
	if r == nil || r.Config == nil || selCtx == nil || selectedModelRef == nil {
		return selectedModelRef, false
	}
	cfg := r.Config.ModelSelection.ModelSwitchGate
	if !cfg.Enabled {
		return selectedModelRef, false
	}

	currentModel := ""
	requestID := ""
	if ctx != nil {
		currentModel = ctx.PreviousModel
		requestID = ctx.RequestID
	}

	cacheWarmth, cacheWarmthOK := estimateGateCacheWarmth(currentModel, time.Now())

	gate := selection.NewModelSwitchGate(cfg, r.LookupTable)
	decision := gate.Evaluate(selection.ModelSwitchGateInput{
		SelectionContext: selCtx,
		SelectionResult:  result,
		CurrentModel:     currentModel,
		CandidateModel:   selectedModelRef.Model,
		CacheWarmth:      cacheWarmth,
		CacheWarmthOK:    cacheWarmthOK,
	})

	fields := decision.LogFields()
	fields["request_id"] = requestID
	fields["decision"] = selCtx.DecisionName
	logging.ComponentDebugEvent("selection", "model_switch_gate_evaluated", fields)

	emitEnforceUnavailableLog(decision, requestID, selCtx.DecisionName)
	metrics.RecordModelSwitchGateDecision(decision.Mode, decision.WouldSwitch, decision.EnforcedStay, decision.Reason)
	if decision.NetSwitchAdvantageOK {
		metrics.ObserveModelSwitchGateNetAdvantage(decision.NetSwitchAdvantage)
	}
	for _, signal := range decision.MissingSignals {
		metrics.RecordModelSwitchGateMissingSignal(signal)
	}

	if !decision.EnforcedStay || decision.FinalModel == selectedModelRef.Model {
		return selectedModelRef, false
	}
	if currentModelRef := findModelRefByModel(selCtx.CandidateModels, decision.FinalModel); currentModelRef != nil {
		return currentModelRef, true
	}
	return selectedModelRef, false
}

// estimateGateCacheWarmth produces a request-time cache warmth prior for model.
// Returns (warmth, true) when latency history exists for the model; otherwise
// (0, false) so the gate records cache_warmth as a missing signal and applies
// no cache penalty (conservative — does not falsely inflate switch cost).
//
// Unlike latency.EstimateCacheProbability — which scores a current TTFT
// observation against history and is therefore only meaningful at response time
// — this helper estimates ambient cache warmth from the freshness of the most
// recent TTFT update alone, which is the only signal available before the
// current request executes.
func estimateGateCacheWarmth(model string, now time.Time) (float64, bool) {
	if model == "" {
		return 0, false
	}
	lastUpdated, ok := latency.GetTTFTLastUpdated(model)
	if !ok {
		return 0, false
	}
	if now.IsZero() {
		now = time.Now()
	}
	age := now.Sub(lastUpdated).Seconds()
	if age < 0 {
		age = 0
	}
	warmth := math.Exp(-math.Ln2 * age / latency.FreshnessHalfLifeSeconds)
	if warmth < 0 {
		warmth = 0
	}
	if warmth > 1 {
		warmth = 1
	}
	return warmth, true
}

// emitEnforceUnavailableLog warns once per request when enforce mode is
// configured but the gate fell back to audit-only because previous_model was
// missing — typically Chat Completions traffic without per-turn model history.
func emitEnforceUnavailableLog(decision selection.ModelSwitchGateDecision, requestID, decisionName string) {
	if decision.Mode != selection.ModelSwitchGateModeEnforce {
		return
	}
	if decision.Reason != "missing_signal_fallback" {
		return
	}
	if !slices.Contains(decision.MissingSignals, "previous_model") {
		return
	}
	logging.ComponentWarnEvent("selection", "model_switch_gate_enforce_unavailable", map[string]interface{}{
		"request_id":      requestID,
		"decision":        decisionName,
		"missing_signals": decision.MissingSignals,
		"hint":            "enforce mode currently requires per-turn model history; only Response API requests carry it. Chat Completions are observed in shadow until persistence ships.",
	})
}

func findModelRefByModel(modelRefs []config.ModelRef, model string) *config.ModelRef {
	for i := range modelRefs {
		if modelRefs[i].Model == model || modelRefs[i].LoRAName == model {
			return &modelRefs[i]
		}
	}
	return nil
}
