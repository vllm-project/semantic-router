package extproc

import (
	"context"
	"errors"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

const (
	selectionFallbackInvalidContext      = "selector_invalid_context"
	selectionFallbackUnavailable         = "selector_unavailable"
	selectionFallbackCancelled           = "selector_cancelled"
	selectionFallbackTimeout             = "selector_timeout"
	selectionFallbackInvocation          = "selector_invocation_failed"
	selectionFallbackInvalidOutput       = "selector_invalid_output"
	selectionFallbackUndeclaredCandidate = "selector_undeclared_candidate"
	selectionFallbackError               = "selector_error"
	selectionFallbackInvalidResult       = "selector_invalid_result"
	selectionFallbackUnknownModel        = "selector_unknown_model"
)

// selectModelFromCandidates uses the configured selection algorithm to choose
// a model. Invalid or unavailable selection falls back to the first configured
// candidate while recording an explicit diagnostic.
func (r *OpenAIRouter) selectModelFromCandidates(
	selCtx *selection.SelectionContext,
	algorithm *config.AlgorithmConfig,
	ctx *RequestContext,
) (*config.ModelRef, string, error) {
	defaultCandidate := firstValidCandidateModelRef(selCtx)
	if defaultCandidate == nil {
		return nil, "", nil
	}
	method := r.getSelectionMethod(algorithm)
	if err := selection.ValidateSelectionContext(selCtx); err != nil {
		logging.Warnf("[ModelSelection] Invalid selection context: %v, using default candidate", err)
		selected := r.recordSelectionFallback(
			method,
			selectionFallbackInvalidContext,
			selCtx,
			nil,
			defaultCandidate,
			nil,
			ctx,
		)
		return selected, string(method), nil
	}
	if len(selCtx.CandidateModels) == 1 {
		return r.selectSingleCandidateModel(selCtx, defaultCandidate, ctx)
	}

	selector := r.selectorForDecisionMethod(method, algorithm, ctx)
	if selector == nil {
		logging.Warnf("[ModelSelection] No selector available for method %s, using default candidate", method)
		selected := r.recordSelectionFallback(
			method,
			selectionFallbackUnavailable,
			selCtx,
			nil,
			defaultCandidate,
			nil,
			ctx,
		)
		return selected, string(method), nil
	}
	return r.selectWithSelector(
		selCtx,
		method,
		selector,
		defaultCandidate,
		ctx,
	)
}

func (r *OpenAIRouter) selectWithSelector(
	selCtx *selection.SelectionContext,
	method selection.SelectionMethod,
	selector selection.Selector,
	defaultCandidate *config.ModelRef,
	ctx *RequestContext,
) (*config.ModelRef, string, error) {
	requestCtx := selectionRequestContext(ctx)
	selectionStart := time.Now()
	result, err := selector.Select(requestCtx, selCtx)
	selection.RecordSelectionDuration(
		method,
		selector.Tier(),
		time.Since(selectionStart),
	)
	if err != nil {
		if requestCtx.Err() != nil {
			return nil, string(method), requestCtx.Err()
		}
		logging.Warnf("[ModelSelection] Selection failed: %v, using default candidate", err)
		selected := r.recordSelectionFallback(
			method,
			selectionFallbackReasonForError(err),
			selCtx,
			nil,
			defaultCandidate,
			selector,
			ctx,
		)
		return selected, string(method), nil
	}
	if err := selection.ValidateSelectionResult(selCtx, result); err != nil {
		logging.Warnf("[ModelSelection] Invalid selection result: %v, using default candidate", err)
		selected := r.recordSelectionFallback(
			method,
			selectionFallbackInvalidResult,
			selCtx,
			result,
			defaultCandidate,
			selector,
			ctx,
		)
		return selected, string(method), nil
	}

	selectedModel := selectedModelRefFromResult(selCtx, result)
	if selectedModel == nil {
		logging.Warnf("[ModelSelection] Selected model %s not found in candidates, using default candidate", result.SelectedModel)
		selected := r.recordSelectionFallback(
			method,
			selectionFallbackUnknownModel,
			selCtx,
			result,
			defaultCandidate,
			selector,
			ctx,
		)
		return selected, string(method), nil
	}
	recordCtx, result, selectedModel, learningApplied := r.applyRouterLearning(
		selCtx,
		result,
		selectedModel,
		ctx,
	)
	ctx.VSRSelectionReasoning = selectionReasoningForDiagnostics(
		method,
		result.Reasoning,
	)
	recordPromptHelperTelemetry(ctx, result)
	logSelectionResult(method, result, selectedModel, learningApplied)
	selection.RecordSelection(
		string(method),
		selectionDecisionStateKey(selCtx),
		selectedModel.Model,
		result.Tier,
		result.Score,
	)
	recordAgenticSessionDecision(recordCtx, result, selectedModel, ctx)
	return selectedModel, string(method), nil
}

func recordPromptHelperTelemetry(
	ctx *RequestContext,
	result *selection.SelectionResult,
) {
	if ctx == nil || result == nil || result.Method != selection.MethodPrompt {
		return
	}
	ctx.VSRPromptHelperModel = result.HelperModel
	ctx.VSRPromptHelperPromptTokens = result.HelperPromptTokens
	ctx.VSRPromptHelperCompletionTokens = result.HelperCompletionTokens
	ctx.VSRPromptHelperTotalTokens = result.HelperTotalTokens
	ctx.VSRPromptHelperLatencyMs = result.HelperLatencyMs
}

func selectionFallbackReasonForError(err error) string {
	switch {
	case errors.Is(err, context.Canceled):
		return selectionFallbackCancelled
	case errors.Is(err, context.DeadlineExceeded):
		return selectionFallbackTimeout
	case errors.Is(err, selection.ErrPromptInvalidOutput):
		return selectionFallbackInvalidOutput
	case errors.Is(err, selection.ErrPromptUndeclaredCandidate):
		return selectionFallbackUndeclaredCandidate
	case errors.Is(err, selection.ErrPromptInvocation):
		return selectionFallbackInvocation
	default:
		return selectionFallbackError
	}
}

func selectionRequestContext(ctx *RequestContext) context.Context {
	if ctx != nil && ctx.TraceContext != nil {
		return ctx.TraceContext
	}
	return context.Background()
}

func (r *OpenAIRouter) selectSingleCandidateModel(
	selCtx *selection.SelectionContext,
	defaultCandidate *config.ModelRef,
	ctx *RequestContext,
) (*config.ModelRef, string, error) {
	result := &selection.SelectionResult{
		SelectedModel: defaultCandidate.Model,
		LoRAName:      defaultCandidate.LoRAName,
		Score:         1.0,
		Confidence:    1.0,
		Method:        selection.MethodStatic,
		Tier:          selection.TierSupported,
		Reasoning:     "single candidate",
		AllScores:     map[string]float64{defaultCandidate.Model: 1.0},
	}
	recordCtx, result, selectedModel, learningApplied := r.applyRouterLearning(
		selCtx,
		result,
		defaultCandidate,
		ctx,
	)
	ctx.VSRSelectionReasoning = boundedSelectionReasoning(result.Reasoning)
	logSelectionResult(selection.MethodStatic, result, selectedModel, learningApplied)
	recordAgenticSessionDecision(recordCtx, result, selectedModel, ctx)
	return selectedModel, "single", nil
}

func boundedSelectionReasoning(value string) string {
	const maxReasoningBytes = 1024
	if len(value) <= maxReasoningBytes {
		return value
	}
	end := 0
	for index := range value {
		if index > maxReasoningBytes {
			break
		}
		end = index
	}
	return value[:end]
}

func selectionReasoningForDiagnostics(
	method selection.SelectionMethod,
	reasoning string,
) string {
	if method == selection.MethodPrompt {
		return "prompt selector selected declared candidate"
	}
	return boundedSelectionReasoning(reasoning)
}

func (r *OpenAIRouter) recordSelectionFallback(
	method selection.SelectionMethod,
	reason string,
	selCtx *selection.SelectionContext,
	_ *selection.SelectionResult,
	fallback *config.ModelRef,
	selector selection.Selector,
	ctx *RequestContext,
) *config.ModelRef {
	tier := selection.TierSupported
	if selector != nil {
		tier = selector.Tier()
	}
	fallbackResult := &selection.SelectionResult{
		SelectedModel: fallback.Model,
		LoRAName:      fallback.LoRAName,
		Method:        method,
		Tier:          tier,
		Reasoning:     reason,
		AllScores:     map[string]float64{fallback.Model: 0},
	}
	recordCtx, fallbackResult, selected, learningApplied := r.applyRouterLearning(
		selCtx,
		fallbackResult,
		fallback,
		ctx,
	)
	if ctx != nil {
		ctx.VSRSelectionReasoning = boundedSelectionReasoning(reason)
	}
	selection.RecordSelection(
		string(method),
		selectionDecisionStateKey(selCtx),
		selected.Model,
		fallbackResult.Tier,
		0,
	)
	selection.RecordSelectionFallback(
		method,
		selectionDecisionStateKey(selCtx),
		reason,
	)
	logSelectionResult(method, fallbackResult, selected, learningApplied)
	recordAgenticSessionDecision(recordCtx, fallbackResult, selected, ctx)
	return selected
}
