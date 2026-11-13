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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

func (r *OpenAIRouter) selectDecisionDefaultRuntimeModel(
	decisionConfig *config.Decision,
	decisionName string,
	ctx *RequestContext,
) (string, entropy.ReasoningDecision, error) {
	if err := validateMinimumEligibleDecisionModels(
		decisionConfig,
		nil,
		ctx.VSRContextTokenCount,
	); err != nil {
		return "", entropy.ReasoningDecision{}, err
	}
	selectedModel := r.Config.DefaultModel
	if r.modelExceedsContextWindow(ctx, selectedModel, ctx.VSRContextTokenCount) {
		withoutReserves := contextTokensWithoutReserves(ctx, ctx.VSRContextTokenCount)
		if refusal := r.contextLengthRefusal(ctx, []string{selectedModel}, withoutReserves, fmt.Sprintf("the default model of decision %q", decisionName)); refusal != nil {
			return "", entropy.ReasoningDecision{}, refusal
		}
		if r.modelExceedsContextWindow(ctx, selectedModel, withoutReserves) {
			r.keepUnprovenContextModels(ctx, []config.ModelRef{{Model: selectedModel}}, nil, withoutReserves, 1, decisionName)
		} else {
			dropContextReserves(ctx, decisionName)
		}
	} else {
		r.dropReservesTheCountOverrules(ctx, []string{selectedModel}, decisionName)
	}
	ctx.VSRSelectedModel = selectedModel
	ctx.VSRSelectionMethod = "default"
	logging.ComponentDebugEvent("extproc", "decision_model_defaulted", map[string]interface{}{
		"request_id":     ctx.RequestID,
		"decision":       decisionName,
		"selected_model": selectedModel,
	})
	return selectedModel, entropy.ReasoningDecision{}, nil
}
