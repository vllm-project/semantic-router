package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
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
	cacheWarmth := 0.0
	requestID := ""
	if ctx != nil {
		currentModel = ctx.PreviousModel
		cacheWarmth = ctx.CacheWarmthEstimate
		requestID = ctx.RequestID
	}

	gate := selection.NewModelSwitchGate(cfg, r.LookupTable)
	decision := gate.Evaluate(selection.ModelSwitchGateInput{
		SelectionContext: selCtx,
		SelectionResult:  result,
		CurrentModel:     currentModel,
		CandidateModel:   selectedModelRef.Model,
		CacheWarmth:      cacheWarmth,
	})

	fields := decision.LogFields()
	fields["request_id"] = requestID
	fields["decision"] = selCtx.DecisionName
	logging.ComponentDebugEvent("selection", "model_switch_gate_evaluated", fields)

	if !decision.EnforcedStay || decision.FinalModel == selectedModelRef.Model {
		return selectedModelRef, false
	}
	if currentModelRef := findModelRefByModel(selCtx.CandidateModels, decision.FinalModel); currentModelRef != nil {
		return currentModelRef, true
	}
	return selectedModelRef, false
}

func findModelRefByModel(modelRefs []config.ModelRef, model string) *config.ModelRef {
	for i := range modelRefs {
		if modelRefs[i].Model == model || modelRefs[i].LoRAName == model {
			return &modelRefs[i]
		}
	}
	return nil
}
