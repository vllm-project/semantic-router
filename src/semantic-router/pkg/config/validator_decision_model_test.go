package config

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// defaultDecisionModelConfig is a decision signal that names no deployment,
// so the decision model answers it.
func defaultDecisionModelConfig() *RouterConfig {
	cfg := &RouterConfig{}
	cfg.ModelDeployments = map[string]ModelDeployment{
		DefaultDecisionDeployment: {
			Provider: ModelRuntimeProvider,
			Artifact: "vllm-sr/Decision-2.0-Kai-0.6B",
			Revision: strings.Repeat("a", 40),
		},
	}
	cfg.DecisionRules = []DecisionSignalRule{
		{Name: "hard", Question: DecisionQuestion{Type: DecisionQuestionNoul, Instructions: "Hard?"}},
	}
	return cfg
}

// A question that names no deployment is answered by the decision model, so
// the decision model's own input budget is checked: a budget a decision model
// cannot honour is refused at load time instead of ignored at request time.
func TestTheDefaultDecisionDeploymentRefusesAnInputItCannotHonour(t *testing.T) {
	cases := map[string]ModelInputBudget{
		"a token limit that rejects":   {MaxTokens: 32768, Overflow: "reject"},
		"a token limit that truncates": {MaxTokens: 32768, Overflow: "truncate"},
		"a scan budget without tokens": {Overflow: "window"},
	}
	for name, budget := range cases {
		t.Run(name, func(t *testing.T) {
			cfg := defaultDecisionModelConfig()
			deployment := cfg.ModelDeployments[DefaultDecisionDeployment]
			deployment.Input = budget
			cfg.ModelDeployments[DefaultDecisionDeployment] = deployment

			err := validateDecisionSignalContracts(cfg)
			require.Error(t, err)
			assert.Contains(t, err.Error(), "decision models never truncate")
			assert.Contains(t, err.Error(), DefaultDecisionDeployment)
		})
	}
}

// The decision model without an input budget, and with the scan budget of a
// model that reads a long part in windows, both answer the question.
func TestTheDefaultDecisionDeploymentAcceptsTheBudgetsItCanHonour(t *testing.T) {
	cases := map[string]ModelInputBudget{
		"no budget":      {},
		"no token limit": {Overflow: "reject"},
		"a scan budget":  {MaxTokens: 32768, Overflow: "window"},
	}
	for name, budget := range cases {
		t.Run(name, func(t *testing.T) {
			cfg := defaultDecisionModelConfig()
			deployment := cfg.ModelDeployments[DefaultDecisionDeployment]
			deployment.Input = budget
			cfg.ModelDeployments[DefaultDecisionDeployment] = deployment

			assert.NoError(t, validateDecisionSignalContracts(cfg))
		})
	}
}

// A selector that names no deployment is answered by the decision model too.
func TestTheDefaultDecisionDeploymentOfASelectorRefusesAnInputItCannotHonour(t *testing.T) {
	cfg := defaultDecisionModelConfig()
	deployment := cfg.ModelDeployments[DefaultDecisionDeployment]
	deployment.Input = ModelInputBudget{MaxTokens: 32768, Overflow: "reject"}
	cfg.ModelDeployments[DefaultDecisionDeployment] = deployment
	cfg.Decisions = []Decision{
		{
			Name:      "pick",
			Algorithm: &AlgorithmConfig{Type: DecisionAlgorithmDecision, Decision: &DecisionSelectionConfig{}},
		},
	}

	err := validateDecisionSignalContracts(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "decision models never truncate")
}

// A deployment the question names itself is checked the same way, and the
// two agree.
func TestANamedDeploymentAndTheDefaultDecisionDeploymentAgree(t *testing.T) {
	cfg := defaultDecisionModelConfig()
	deployment := cfg.ModelDeployments[DefaultDecisionDeployment]
	deployment.Input = ModelInputBudget{MaxTokens: 32768, Overflow: "reject"}
	cfg.ModelDeployments[DefaultDecisionDeployment] = deployment
	cfg.DecisionRules[0].Deployment = DefaultDecisionDeployment

	err := validateDecisionSignalContracts(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "decision models never truncate")
}
