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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildRoutingEvidenceCarriesConfidenceAndMatchedRules(t *testing.T) {
	decision := &config.Decision{Name: "coding-route"}
	reqCtx := &RequestContext{
		VSRSelectedDecisionConfidence:   0.87,
		VSRSelectedDecisionMatchedRules: []string{"domain:coding", "keyword:refactor"},
	}

	evidence := buildRoutingEvidence(decision, reqCtx)

	require.NotNil(t, evidence)
	assert.Equal(t, "coding-route", evidence.DecisionName)
	assert.Equal(t, 0.87, evidence.Confidence)
	assert.Equal(t, []string{"domain:coding", "keyword:refactor"}, evidence.MatchedSignals)
}

func TestBuildRoutingEvidenceZeroValueWhenNoDecisionEvaluationRan(t *testing.T) {
	decision := &config.Decision{Name: "explicit-model-route"}
	reqCtx := &RequestContext{}

	evidence := buildRoutingEvidence(decision, reqCtx)

	require.NotNil(t, evidence)
	assert.Equal(t, "explicit-model-route", evidence.DecisionName)
	assert.Equal(t, 0.0, evidence.Confidence)
	assert.Empty(t, evidence.MatchedSignals)
}

func TestBuildComputeBudgetNilWhenUnconfigured(t *testing.T) {
	assert.Nil(t, buildComputeBudget(nil))
	assert.Nil(t, buildComputeBudget(&config.AlgorithmConfig{Type: config.DecisionAlgorithmConfidence}))
}

func TestBuildComputeBudgetCopiesConfiguredLimits(t *testing.T) {
	algorithm := &config.AlgorithmConfig{
		Type: config.DecisionAlgorithmConfidence,
		Budget: &config.BudgetConfig{
			MaxPromptTokens:     1000,
			MaxCompletionTokens: 500,
			MaxTotalTokens:      1200,
			MaxEstimatedCost:    0.75,
			MaxWallTimeMs:       20000,
		},
	}

	budget := buildComputeBudget(algorithm)

	require.NotNil(t, budget)
	assert.Equal(t, int64(1000), budget.MaxPromptTokens)
	assert.Equal(t, int64(500), budget.MaxCompletionTokens)
	assert.Equal(t, int64(1200), budget.MaxTotalTokens)
	assert.Equal(t, 0.75, budget.MaxEstimatedCost)
	assert.Equal(t, int64(20000), budget.MaxWallTimeMs)
	assert.Equal(t, "USD", budget.Currency)
}
