package extproc

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestAutomaticOutputSurvivesRuntimeSelectionAndLearning(t *testing.T) {
	for _, test := range []struct {
		name      string
		algorithm string
		explicit  bool
		legacy    bool
	}{
		{"static/automatic", config.DecisionAlgorithmStatic, false, false},
		{"static/explicit", config.DecisionAlgorithmStatic, true, false},
		{"multi_factor/automatic", config.DecisionAlgorithmMultiFactor, false, false},
		{"multi_factor/explicit", config.DecisionAlgorithmMultiFactor, true, false},
		{"static/automatic-without-strict-requirements", config.DecisionAlgorithmStatic, false, true},
		{"multi_factor/automatic-without-strict-requirements", config.DecisionAlgorithmMultiFactor, false, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			calls := 0
			text := "hello"
			if test.legacy {
				text = strings.Repeat("中文🙂", 9000)
			}
			r, ctx := automaticFixture(t, text, renderMock(t, &calls, 0, 0))
			if test.legacy {
				r.Config.CandidateRequirements = nil
				ctx.VSRContextTokenCount = len(text)
			}
			if test.explicit {
				ctx.SemanticRequest.Sampling.MaxOutputTokens = llmprotocol.Int64(12)
			}
			d := ctx.VSRSelectedDecision
			d.Algorithm.Type = test.algorithm
			if test.algorithm == config.DecisionAlgorithmMultiFactor {
				d.Algorithm.MultiFactor = &config.MultiFactorSelectionConfig{Weights: &config.MultiFactorWeightsConfig{Quality: 1}}
				r.Config.ModelConfig[d.ModelRefs[0].Model] = addTestQuality(r.Config.ModelConfig[d.ModelRefs[0].Model], 0.9)
			}
			_, _, _, model, err := r.finalizeDecisionEvaluation(&decision.DecisionResult{Decision: d, Confidence: 1}, "auto", text, ctx)
			require.NoError(t, err)
			require.Equal(t, d.ModelRefs[0].Model, model)
			require.Equal(t, d.ModelRefs, r.eligibleLearningModelRefs(d.ModelRefs, ctx))
			if test.explicit {
				require.Zero(t, calls)
				require.EqualValues(t, 12, *ctx.SemanticRequest.Sampling.MaxOutputTokens)
			} else {
				require.Equal(t, 1, calls, "capability checks must reuse the candidate render")
				require.Nil(t, ctx.SemanticRequest.Sampling.MaxOutputTokens, "selection must not mutate ingress")
			}
		})
	}
}

func TestAutomaticOutputCapabilityFilterStillRejectsInvalidCandidates(t *testing.T) {
	for _, scenario := range []string{"missing-render", "unresolved-render", "changed-capabilities", "invalid-policy", "excluded-by-policy"} {
		t.Run(scenario, func(t *testing.T) {
			calls := 0
			r, ctx := automaticFixture(t, "hello", renderMock(t, &calls, 0, 0))
			d := ctx.VSRSelectedDecision
			require.NoError(t, r.prepareDecisionContextOverflow(ctx, "auto"))
			refs, err := r.decisionEligibleModelRefs(d, ctx)
			require.NoError(t, err)
			switch scenario {
			case "missing-render":
				ctx.AutomaticCandidateDemands = nil
			case "unresolved-render":
				ctx.AutomaticCandidateDemands[refs[0].Model] = selection.CandidateDemand{Known: true, AutomaticOutput: true}
			case "changed-capabilities":
				ctx.SemanticRequest.ImageGeneration = &llmprotocol.ImageGenerationOptions{}
			case "invalid-policy":
				setAutomaticBlockedParams(t, ctx, []string{"messages"})
			case "excluded-by-policy":
				ctx.VSRPolicyEligibleModelRefs = []config.ModelRef{}
			}
			selected, _, err := r.selectModelFromCandidates(&selection.SelectionContext{
				DecisionName: d.Name, CandidateModels: refs,
			}, d.Algorithm, ctx)
			require.ErrorIs(t, err, selection.ErrNoEligibleCandidates)
			require.Nil(t, selected)
			require.Empty(t, r.eligibleLearningModelRefs(refs, ctx))
			require.Equal(t, 1, calls)
		})
	}
}
