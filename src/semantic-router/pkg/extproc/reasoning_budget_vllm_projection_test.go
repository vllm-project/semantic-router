package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A vLLM profile projects the neutral reasoning_budget_tokens to vLLM's
// thinking_token_budget, keeping reasoning_effort. An explicit native value
// wins; a provider that cannot honor the budget is refused.
func TestVLLMReasoningBudgetProjection(t *testing.T) {
	router := newReasoningRouter(
		config.ReasoningConfig{ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
			"deepseek": {
				Type: config.ReasoningFamilyTypeTopLevelReasoningEffort, Parameter: "reasoning_effort",
				Levels: []string{"low", "high", "max"}, Default: "max",
				Modes: []string{config.ReasoningModeEnabled}, DefaultMode: config.ReasoningModeEnabled,
			},
		}},
		[]config.Decision{reasoningDecision("route", "", 0, "deepseek-model", boolPtr(true), "max")},
		map[string]config.ModelParams{"deepseek-model": {ReasoningFamily: "deepseek"}},
	)
	decision := router.Config.GetDecisionByName("route")
	dispatch := func() *providerDispatch {
		return &providerDispatch{
			logicalModel: "deepseek-model", targetFormat: llmprotocol.OpenAIChatV1,
			decisionName: "route", useReasoning: true, profile: localVLLMProviderProfile(),
		}
	}
	ctx := func() *RequestContext {
		return &RequestContext{VSRSelectedDecision: decision, SourceFormat: llmprotocol.OpenAIChatV1}
	}

	t.Run("projects neutral budget", func(t *testing.T) {
		body := []byte(`{"model":"deepseek-model","messages":[{"role":"user","content":"synthetic"}],"reasoning_effort":"max","reasoning_budget_tokens":1024}`)
		adapted, err := router.adaptProviderRequest(body, dispatch(), ctx())
		require.NoError(t, err)
		request := unmarshalReasoningRequest(t, adapted)
		assert.Equal(t, float64(1024), request["thinking_token_budget"])
		assert.Equal(t, "max", request["reasoning_effort"])
	})

	t.Run("explicit native value wins", func(t *testing.T) {
		body := []byte(`{"model":"deepseek-model","messages":[],"reasoning_budget_tokens":1024,"thinking_token_budget":512}`)
		adapted, err := router.adaptProviderRequest(body, dispatch(), ctx())
		require.NoError(t, err)
		request := unmarshalReasoningRequest(t, adapted)
		assert.Equal(t, float64(512), request["thinking_token_budget"])
	})

	t.Run("provider without native budget is refused", func(t *testing.T) {
		body := []byte(`{"model":"deepseek-model","messages":[],"reasoning_budget_tokens":1024}`)
		d := dispatch()
		d.profile = &config.ProviderProfile{Type: "custom"}
		_, err := router.adaptProviderRequest(body, d, ctx())
		require.Error(t, err)
	})
}
