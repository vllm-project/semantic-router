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
	"fmt"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

type looperFailureEvidence struct {
	algorithm    string
	modelsUsed   []string
	iterations   int
	usage        looper.TokenUsage
	fusionQuorum *routerreplay.FusionQuorumDiagnostics
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

	logging.ComponentErrorEvent("extproc", "looper_execution_failed", failureFields)
	return r.recordLooperFailure(
		reqCtx,
		originalModel,
		decision,
		500,
		"Looper execution failed: "+err.Error(),
		"looper_execution_failed",
		evidenceArgs...,
	)
}

func looperFailureEvidenceFromError(err error) (looperFailureEvidence, bool) {
	if evidence, ok := looper.ConfidenceEvidenceFromError(err); ok {
		return looperFailureEvidence{
			algorithm:  config.DecisionAlgorithmConfidence,
			modelsUsed: append([]string(nil), evidence.ModelsUsed...),
			iterations: evidence.Iterations,
			usage:      evidence.Usage,
		}, true
	}
	if evidence, ok := looper.FusionQuorumEvidenceFromError(err); ok {
		return looperFailureEvidence{
			algorithm:    config.DecisionAlgorithmFusion,
			usage:        evidence.Usage,
			fusionQuorum: fusionQuorumDiagnostics(evidence),
		}, true
	}
	return looperFailureEvidence{}, false
}

func fusionQuorumDiagnostics(evidence looper.FusionQuorumEvidence) *routerreplay.FusionQuorumDiagnostics {
	attempts := make([]routerreplay.FusionPanelAttemptDiagnostics, len(evidence.Attempts))
	for index, attempt := range evidence.Attempts {
		attempts[index] = routerreplay.FusionPanelAttemptDiagnostics{
			Model:            attempt.Model,
			State:            string(attempt.State),
			PromptTokens:     attempt.Usage.PromptTokens,
			CompletionTokens: attempt.Usage.CompletionTokens,
			TotalTokens:      attempt.Usage.TotalTokens,
		}
	}
	return &routerreplay.FusionQuorumDiagnostics{
		RequiredCount: evidence.RequiredCount,
		UsableCount:   evidence.UsableCount,
		Attempts:      attempts,
	}
}

func (evidence looperFailureEvidence) addLogFields(fields map[string]interface{}) {
	fields["prompt_tokens"] = evidence.usage.PromptTokens
	fields["completion_tokens"] = evidence.usage.CompletionTokens
	fields["total_tokens"] = evidence.usage.TotalTokens
	if evidence.algorithm == config.DecisionAlgorithmConfidence {
		fields["models_used"] = evidence.modelsUsed
		fields["iterations"] = evidence.iterations
	}
	if evidence.fusionQuorum != nil {
		fields["fusion_quorum"] = evidence.fusionQuorum
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
		r.updateLooperReplayUsage(ctx, executionEvidence[0].usage)
	}
	r.updateRouterReplayStatus(ctx, statusCode, false)
	if immediate := response.GetImmediateResponse(); immediate != nil {
		r.attachRouterReplayResponse(ctx, immediate.Body, false)
	}
	r.finalizeRouterReplay(ctx, routerreplay.LifecycleFailed, reason)
	addRouterReplayHeaderToImmediateResponse(response, ctx.RouterReplayID)
	return response
}

func applyLooperFailureEvidence(ctx *RequestContext, evidence looperFailureEvidence) {
	switch evidence.algorithm {
	case config.DecisionAlgorithmConfidence:
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
		ctx.VSRFusionQuorum = evidence.fusionQuorum
	}
}
