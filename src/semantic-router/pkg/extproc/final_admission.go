package extproc

import (
	"errors"
	"fmt"
	"slices"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func isFinalContextOverflow(err error) bool {
	var budget *selection.RequestBudgetError
	return errors.As(err, &budget) && budget.Code == "context_length_exceeded"
}

func isDispatchBudgetError(err error) bool {
	var budget *selection.RequestBudgetError
	return errors.As(err, &budget)
}

// A candidate projection publishes plugin observations only if it is dispatched.
func (ctx *RequestContext) observeProjection(record func()) {
	if ctx == nil || ctx.projectionObservations == nil {
		record()
		return
	}
	*ctx.projectionObservations = append(*ctx.projectionObservations, record)
}

type dispatchAdmission = routerreplay.FinalAdmissionDiagnostics

func (r *OpenAIRouter) admitProviderBody(body []byte, request *llmprotocol.Request, dispatch *providerDispatch, ctx *RequestContext) ([]byte, dispatchAdmission, error) {
	model := dispatch.effectiveBackendModel()
	window := r.contextWindowSize(model)
	result := dispatchAdmission{Model: model, ContextWindow: window, Source: "unknown", Verdict: "unknown"}
	if request == nil {
		return body, result, nil
	}
	if request.Sampling.AutomaticOutput {
		if err := r.checkAutomaticBackend(request, dispatch, ctx); err != nil {
			return nil, result, err
		}
		limitField := "max_completion_tokens"
		if gjson.GetBytes(body, "max_tokens").Exists() {
			limitField = "max_tokens"
		}
		renderBody, err := sjson.DeleteBytes(body, limitField)
		if err != nil {
			return nil, result, err
		}
		result.Source = "provider_render"
		if renderErr := r.applyRenderedBudget(renderBody, request, dispatch, ctx); renderErr != nil {
			if isFinalContextOverflow(renderErr) {
				result.Verdict = "exceeds"
			}
			return nil, result, renderErr
		}
		body, err = sjson.SetBytes(body, limitField, *request.Sampling.MaxOutputTokens)
		result.Verdict = "fits"
		result.PromptTokens = int(*request.Sampling.AutomaticInputTokens)
		reserve, _ := requestOutputReserve(request)
		result.ReservedOutputTokens = &reserve
		return body, result, err
	}

	if window <= 0 || !slices.Contains([]llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1}, dispatch.targetFormat) {
		return body, result, nil
	}
	engine, err := r.protocolEngine()
	if err != nil {
		return nil, result, err
	}
	projected, _, _, err := engine.DecodeRequest(dispatch.targetFormat, body)
	if err != nil {
		return body, result, nil
	}
	reserve, reserveSource := requestOutputReserve(&projected)
	available := window - reserve
	cfg := contextOverflowConfig(r.Config, ctx)
	if cfg != nil {
		budget, budgetErr := r.contextOverflowInputBudget(model, projected.Sampling.MaxOutputTokens, cfg)
		if budgetErr != nil {
			if isDispatchBudgetError(budgetErr) {
				result.Source, result.Verdict = "output_reserve", "exceeds"
			}
			return nil, result, budgetErr
		}
		available, reserve = budget, window-budget
	}
	if cfg != nil || reserveSource == requestDemandOutputReserveExplicit {
		result.ReservedOutputTokens = &reserve
	}
	if contextErr := selectionRequestContext(ctx).Err(); contextErr != nil {
		return nil, result, contextErr
	}
	carriesMedia := requestCarriesMedia(&projected)
	if available < 0 || available == 0 && carriesMedia {
		result.Source, result.Verdict = "output_reserve", "exceeds"
		return nil, result, overflowBudgetError(fmt.Sprintf("The final request's %d-token output reserve leaves no input capacity in the %d-token context window", reserve, window))
	}
	if carriesMedia || projected.Truncation != "" && projected.Truncation != "disabled" {
		return body, result, nil
	}
	if bound := contextByteBound(&projected, nil); bound < available {
		result.Source, result.PromptTokenBound = "byte_bound", bound
		if result.ReservedOutputTokens != nil {
			result.Verdict = "fits"
		}
		return body, result, nil
	}
	tokenizer := r.requestTokenizer(ctx, model)
	if tokenizer.encoding == nil {
		return body, result, nil
	}
	count, err := countPromptTokensContext(selectionRequestContext(ctx), tokenizer.encoding, &projected)
	if err != nil {
		return nil, result, err
	}
	verdict := r.bandVerdict(available, count, tokenizer.kind)
	result.Source, result.Verdict, result.PromptTokens = string(tokenizer.kind), verdict.String(), count
	if verdict == contextExceeds {
		return nil, result, overflowBudgetError(fmt.Sprintf("The final request counts about %d input tokens with %d reserved for output in a %d-token context window", count, reserve, window))
	}
	if result.ReservedOutputTokens == nil {
		result.Verdict = "unknown"
	}
	return body, result, nil
}

func (r *OpenAIRouter) materializeAdmittedCandidate(dispatch *providerDispatch, ctx *RequestContext) (*preparedProviderRequest, dispatchAdmission, error) {
	prepared, err := r.materializePreparedProviderRequest(dispatch, ctx)
	if err != nil {
		return nil, dispatchAdmission{}, err
	}
	if pass := ctx.structuredOutputPass; pass != nil {
		view := *ctx.SemanticRequest
		body, admission, passErr := r.admitProviderBody(pass.body, &view, dispatch, ctx)
		if passErr != nil {
			admission.RequestPart = "structured_output_pass"
			return prepared, admission, passErr
		}
		pass.body = body
	}
	body, admission, err := r.admitProviderBody(prepared.body, ctx.SemanticRequest, dispatch, ctx)
	if err == nil {
		prepared = newPreparedProviderRequest(body, prepared.metadata)
	}
	return prepared, admission, err
}

func (r *OpenAIRouter) prepareFinalCandidate(dispatch *providerDispatch, ctx *RequestContext) (*preparedProviderRequest, dispatchAdmission, error) {
	model := dispatch.effectiveBackendModel()
	preparationErr := dispatch.preparationBudgetError
	if dispatch.compressionPending {
		preparationErr = r.compressAndMaskDispatch(ctx, dispatch)
	}
	if preparationErr != nil {
		verdict := "unknown"
		if isDispatchBudgetError(preparationErr) {
			verdict = "exceeds"
		} else if !dispatch.compressionPending || !errors.Is(preparationErr, selection.ErrNoEligibleCandidates) {
			return nil, dispatchAdmission{}, preparationErr
		}
		admission := dispatchAdmission{
			Model: model, ContextWindow: r.contextWindowSize(model), Source: "provider_preparation", Verdict: verdict,
		}
		metrics.RecordFinalAdmission(admission.Source, admission.Verdict)
		return nil, admission, preparationErr
	}
	diagnostics := len(ctx.ProtocolDiagnostics)
	observations := len(*ctx.projectionObservations)
	prepared, admission, err := r.materializeAdmittedCandidate(dispatch, ctx)
	if ctx.SemanticRequest.Sampling.AutomaticOutput && isFinalContextOverflow(err) {
		ctx.ProtocolDiagnostics = ctx.ProtocolDiagnostics[:diagnostics]
		*ctx.projectionObservations = (*ctx.projectionObservations)[:observations]
		err = r.compressAutomaticOverflow(ctx, ctx.SemanticRequest, dispatch, err)
		if err == nil {
			prepared, admission, err = r.materializeAdmittedCandidate(dispatch, ctx)
		}
	}
	if admission.Source != "" {
		metrics.RecordFinalAdmission(admission.Source, admission.Verdict)
	}
	if err == nil && ctx.SemanticRequest != nil {
		err = r.rejectDispatchCapabilityMismatch(ctx.SemanticRequest, dispatch, ctx)
	}
	if err != nil && selectionRequestContext(ctx).Err() == nil {
		return nil, admission, err
	}
	return prepared, admission, err
}
