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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

// buildRoutingEvidence carries the DecisionEngine's confidence and matched
// rule identifiers (already computed earlier in the request pipeline, see
// req_filter_classification_runtime.go) into the Looper request. It never
// reads prompt content: reqCtx.VSRSelectedDecisionMatchedRules holds
// config-defined rule/category identifiers, not request text.
func buildRoutingEvidence(decision *config.Decision, reqCtx *RequestContext) *looper.RoutingEvidence {
	return &looper.RoutingEvidence{
		DecisionName:   decision.Name,
		Confidence:     reqCtx.VSRSelectedDecisionConfidence,
		MatchedSignals: reqCtx.VSRSelectedDecisionMatchedRules,
	}
}

// buildComputeBudget converts a decision's algorithm.budget config into the
// looper.ComputeBudget attached to Request. Cost is estimated from actual
// usage as calls complete (see looper.RecordBudgetUsage), not predicted here,
// so this is a direct field copy with no pricing lookup.
func buildComputeBudget(algorithm *config.AlgorithmConfig) *looper.ComputeBudget {
	if algorithm == nil || algorithm.Budget == nil {
		return nil
	}
	cfg := algorithm.Budget
	return &looper.ComputeBudget{
		MaxPromptTokens:     cfg.MaxPromptTokens,
		MaxCompletionTokens: cfg.MaxCompletionTokens,
		MaxTotalTokens:      cfg.MaxTotalTokens,
		MaxEstimatedCost:    cfg.MaxEstimatedCost,
		Currency:            "USD",
		MaxWallTimeMs:       cfg.MaxWallTimeMs,
	}
}
