package extproc

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestReasoningBudgetHintSurvivesOtherCandidateExclusions(t *testing.T) {
	router, budgetModel := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	contextModel := "context-limited-model"
	router.Config.ModelConfig[contextModel] = config.ModelParams{
		APIFormat: config.APIFormatOpenAI, ContextWindowSize: 1,
	}
	refs := []config.ModelRef{{Model: budgetModel}, {Model: contextModel}}
	request := testNeutralRequest("virtual", strings.Repeat("long context ", 100))
	request.ReasoningBudgetTokens = llmprotocol.Int64(1024)
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = &config.Decision{Name: "budget-selection", ModelRefs: refs}
	ctx.VSRContextTokenCount = 100

	selected, err := router.capabilityEligibleSelectionContext(&selection.SelectionContext{
		DecisionName: "budget-selection", CandidateModels: refs,
	}, nil, ctx)
	require.Error(t, err)
	assert.Nil(t, selected)
	assert.ErrorContains(t, err, "no candidate supporting the request capabilities")
	assert.ErrorContains(t, err, "reasoning_budget_tokens in global.router.allow_lossy_translation")
}
