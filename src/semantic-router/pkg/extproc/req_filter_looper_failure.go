/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	"errors"
	"fmt"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

type looperFailureEvidence struct {
	algorithm      string
	modelsUsed     []string
	iterations     int
	usage          looper.TokenUsage
	executionTrace looper.ExecutionTrace
	fusionQuorum   *looper.FusionQuorumOutcome
}

func (r *OpenAIRouter) looperExecutionErrorResponse(
	err error,
	originalModel string,
	decision *config.Decision,
	reqCtx *RequestContext,
) *ext_proc.ProcessingResponse {
	failureFields := map[string]interface{}{
		"request_id": reqCtx.RequestID,
		"decision":   decision.Name,
		"algorithm":  decision.Algorithm.Type,
		"error":      err.Error(),
	}

	evidence, hasEvidence := looperFailureEvidenceFromError(err)
	var evidenceArgs []looperFailureEvidence
	if hasEvidence {
		evidence.addLogFields(failureFields)
		evidenceArgs = append(evidenceArgs, evidence)
	}

	// Terminal below-quorum outcomes are counted here rather than in the
	// algorithm layer, so a single recorder owns every quorum sample. The
	// disposition is already terminal on this path, so nothing is promoted.
	finalizeLooperQuorumOutcome(reqCtx, evidence.fusionQuorum, decision, false)

	logging.ComponentErrorEvent("extproc", "looper_execution_failed", failureFields)
	// A run a spend budget stopped answers as the refused call would alone,
	// read again now that its other calls have settled.
	if refusal := reqCtx.spend.hopRefusal(); refusal != nil {
		return r.respondLooperBudgetRefused(reqCtx, reqCtx.spend.recheck(refusal), evidenceArgs...)
	}
	status, message, reason := 500, "Looper execution failed: "+err.Error(), "looper_execution_failed"
	// A model's rate limit or failure that ended the run reaches the client
	// with the model's status, as it would without a Looper.
	if upstream := looper.UpstreamStatus(err); upstream == 429 || upstream >= 500 {
		status = upstream
	}
	// A run whose every hop dispatch refused answers with that refusal, as
	// the request would without a Looper.
	var refused *looperHopsRefusedError
	if errors.As(err, &refused) {
		reqCtx.ImmediateProtocolError = refused.refusal
		status, message, reason = 400, refused.refusal.Message, refused.refusal.Code
	}
	// A hop or check refused for want of the request's delegated credential
	// answers the run 401, with the typed error the refusal records.
	if r.delegatedCredentialRefusal(reqCtx) != nil {
		status, message, reason = 401, delegatedCredentialMessage, delegatedCredentialCode
	}
	return r.recordLooperFailure(reqCtx, originalModel, decision, status, message, reason, evidenceArgs...)
}

// respondLooperBudgetRefused answers a Looper run whose call a spend budget
// refused with that refusal, and records the run as refused, with what its
// other calls used and spent.
func (r *OpenAIRouter) respondLooperBudgetRefused(
	ctx *RequestContext,
	refusal *budgetRefusal,
	executionEvidence ...looperFailureEvidence,
) *ext_proc.ProcessingResponse {
	if len(executionEvidence) > 0 {
		applyLooperFailureEvidence(ctx, executionEvidence[0])
	}
	ctx.spend.setReport(refusal.status, refusal.estimate)
	response := r.budgetRefusalResponse(ctx, refusal)
	status := int(response.GetImmediateResponse().GetStatus().GetCode())
	response = r.recordRejection(ctx, ctx.spend.model, "", status, response, refusal.code)
	if len(executionEvidence) > 0 {
		r.updateLooperReplayUsage(ctx, executionEvidence[0].usage, executionEvidence[0].iterations)
	}
	r.refreshReplayBudget(ctx)
	return response
}

func looperFailureEvidenceFromError(err error) (looperFailureEvidence, bool) {
	if evidence, ok := looper.EscalateEvidenceFromError(err); ok {
		return looperFailureEvidence{
			algorithm:      config.DecisionAlgorithmEscalate,
			modelsUsed:     append([]string(nil), evidence.ModelsUsed...),
			iterations:     evidence.Iterations,
			usage:          evidence.Usage,
			executionTrace: evidence.ExecutionTrace,
		}, true
	}
	if outcome, ok := looper.FusionQuorumOutcomeFromError(err); ok {
		evidence, _ := looper.FusionQuorumEvidenceFromError(err)
		modelsUsed := make([]string, 0, len(evidence.Attempts))
		for _, att := range evidence.Attempts {
			if att.Model != "" {
				modelsUsed = append(modelsUsed, att.Model)
			}
		}
		return looperFailureEvidence{
			algorithm:    config.DecisionAlgorithmFusion,
			modelsUsed:   modelsUsed,
			iterations:   len(modelsUsed),
			usage:        evidence.Usage,
			fusionQuorum: outcome,
		}, true
	}
	return looperFailureEvidence{}, false
}

// looperQuorumOutcomeDiagnostics builds the Replay projection for a finalized
// quorum outcome. Both the served-fallback and terminal-failure paths reach it
// with the same type, so every path persists the same bounded evidence. The
// outcome type already excludes provider error text, so nothing is stripped
// here.
func looperQuorumOutcomeDiagnostics(
	outcome *looper.FusionQuorumOutcome,
) *routerreplay.FusionQuorumDiagnostics {
	if outcome == nil {
		return nil
	}
	return &routerreplay.FusionQuorumDiagnostics{
		RequiredCount:  outcome.RequiredCount,
		UsableCount:    outcome.UsableCount,
		SelectedPolicy: outcome.SelectedPolicy,
		FallbackTarget: outcome.FallbackTarget,
		Disposition:    string(outcome.Disposition),
		Attempts:       fusionQuorumOutcomeDiagnostics(outcome.Attempts),
	}
}

func fusionQuorumOutcomeDiagnostics(
	attempts []looper.FusionQuorumAttemptOutcome,
) []routerreplay.FusionPanelAttemptDiagnostics {
	diagnostics := make([]routerreplay.FusionPanelAttemptDiagnostics, len(attempts))
	for index, attempt := range attempts {
		diagnostics[index] = routerreplay.FusionPanelAttemptDiagnostics{
			Model:            attempt.Model,
			State:            string(attempt.State),
			PromptTokens:     attempt.Usage.PromptTokens,
			CompletionTokens: attempt.Usage.CompletionTokens,
			TotalTokens:      attempt.Usage.TotalTokens,
		}
	}
	return diagnostics
}

func (evidence looperFailureEvidence) addLogFields(fields map[string]interface{}) {
	fields["prompt_tokens"] = evidence.usage.PromptTokens
	fields["completion_tokens"] = evidence.usage.CompletionTokens
	fields["total_tokens"] = evidence.usage.TotalTokens
	if evidence.algorithm == config.DecisionAlgorithmEscalate {
		fields["models_used"] = evidence.modelsUsed
		fields["iterations"] = evidence.iterations
	}
	if evidence.fusionQuorum != nil {
		fields["fusion_quorum"] = looperQuorumOutcomeDiagnostics(evidence.fusionQuorum)
	}
}

func (r *OpenAIRouter) recordLooperFailure(
	ctx *RequestContext,
	originalModel string,
	decision *config.Decision,
	statusCode int,
	message string,
	reason string,
	executionEvidence ...looperFailureEvidence,
) *ext_proc.ProcessingResponse {
	return r.recordLooperFailureWithLifecycle(
		ctx,
		originalModel,
		decision,
		statusCode,
		message,
		routerreplay.LifecycleFailed,
		reason,
		executionEvidence...,
	)
}

func (r *OpenAIRouter) recordLooperFailureWithLifecycle(
	ctx *RequestContext,
	originalModel string,
	decision *config.Decision,
	statusCode int,
	message string,
	lifecycleState string,
	reason string,
	executionEvidence ...looperFailureEvidence,
) *ext_proc.ProcessingResponse {
	response := r.createErrorResponse(statusCode, message)
	decisionName := ""
	if decision != nil {
		decisionName = decision.Name
	}
	if len(executionEvidence) > 0 {
		applyLooperFailureEvidence(ctx, executionEvidence[0])
	}

	r.startRouterReplay(ctx, originalModel, "", decisionName)
	if len(executionEvidence) > 0 {
		r.updateLooperReplayUsage(ctx, executionEvidence[0].usage, executionEvidence[0].iterations)
	}
	r.updateRouterReplayStatus(ctx, statusCode, false)
	if immediate := response.GetImmediateResponse(); immediate != nil {
		r.attachRouterReplayResponse(ctx, immediate.Body, false)
	}
	r.finalizeRouterReplay(ctx, lifecycleState, reason)
	if len(executionEvidence) > 0 {
		recordLooperIntermediateRejectedUsage(ctx, originalModel, decisionName, executionEvidence[0])
	}
	return response
}

func (r *OpenAIRouter) recordLooperImmediateDispatchFailure(
	ctx *RequestContext,
	originalModel string,
	decision *config.Decision,
	response *ext_proc.ProcessingResponse,
	evidence looperFailureEvidence,
) {
	if ctx == nil || response == nil {
		return
	}
	decisionName := ""
	if decision != nil {
		decisionName = decision.Name
	}
	applyLooperFailureEvidence(ctx, evidence)

	statusCode := 500
	if immediate := response.GetImmediateResponse(); immediate != nil {
		if immediate.GetStatus() != nil && immediate.GetStatus().GetCode() != 0 {
			statusCode = int(immediate.GetStatus().GetCode())
		}
	}

	r.startRouterReplay(ctx, originalModel, "", decisionName)
	r.updateLooperReplayUsage(ctx, evidence.usage, evidence.iterations)
	r.updateRouterReplayStatus(ctx, statusCode, false)
	if immediate := response.GetImmediateResponse(); immediate != nil {
		r.attachRouterReplayResponse(ctx, immediate.Body, false)
	}
	r.finalizeRouterReplay(ctx, routerreplay.LifecycleFailed, "looper_stream_dispatch_rejected")
	recordLooperIntermediateRejectedUsage(ctx, originalModel, decisionName, evidence)
}

func recordLooperIntermediateRejectedUsage(
	ctx *RequestContext,
	originalModel string,
	decisionName string,
	ev looperFailureEvidence,
) {
	if ev.usage.PromptTokens <= 0 && ev.usage.CompletionTokens <= 0 {
		return
	}
	recordSessionTurn(ctx, responseUsageMetrics{
		promptTokens:     int(ev.usage.PromptTokens),
		completionTokens: int(ev.usage.CompletionTokens),
	}, sessiontelemetry.TurnPricing{})
	logging.LogEvent("llm_usage", map[string]interface{}{
		"request_id":                     ctx.RequestID,
		"model":                          originalModel,
		"decision":                       decisionName,
		"prompt_tokens":                  ev.usage.PromptTokens,
		"completion_tokens":              ev.usage.CompletionTokens,
		"total_tokens":                   ev.usage.TotalTokens,
		"cost":                           0.0,
		"currency":                       "unknown",
		"pricing":                        "looper_immediate_rejected_unpriced",
		"intermediate_prompt_tokens":     ev.usage.PromptTokens,
		"intermediate_completion_tokens": ev.usage.CompletionTokens,
		"looper_models_used":             ev.modelsUsed,
		"looper_iterations":              ev.iterations,
	})
}

func applyLooperFailureEvidence(ctx *RequestContext, evidence looperFailureEvidence) {
	switch evidence.algorithm {
	case config.DecisionAlgorithmEscalate:
		ctx.VSRLooperDiagnostics = looperReplayDiagnostics(evidence.executionTrace)
		if evidence.iterations <= 0 {
			return
		}
		ctx.VSRSelectionMethod = evidence.algorithm
		ctx.VSRSelectionReasoning = boundedSelectionReasoning(fmt.Sprintf(
			"%s execution failed after %d call attempts; completed models: %s",
			evidence.algorithm,
			evidence.iterations,
			strings.Join(evidence.modelsUsed, ","),
		))
	case config.DecisionAlgorithmFusion:
		if evidence.fusionQuorum == nil {
			return
		}
		ctx.VSRSelectionMethod = evidence.algorithm
		ctx.VSRSelectionReasoning = boundedSelectionReasoning(fmt.Sprintf(
			"fusion panel quorum not met: %d/%d usable responses",
			evidence.fusionQuorum.UsableCount,
			evidence.fusionQuorum.RequiredCount,
		))
	}
}
