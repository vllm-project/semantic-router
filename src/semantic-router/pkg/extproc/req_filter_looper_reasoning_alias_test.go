package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestLooperReasoningCapabilityUsesPhysicalModelNotRequestAlias(t *testing.T) {
	const requestAlias = "routing/remom"
	const physicalModel = "reasoning-model"
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Looper: config.LooperConfig{
				ReMoM: config.ReMoMRuntimeConfig{ModelNames: []string{requestAlias}},
			},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					physicalModel: {ReasoningFamily: "generic"},
				},
			},
		},
	}
	decision := &config.Decision{
		ModelRefs: []config.ModelRef{{Model: physicalModel}},
	}

	useReasoning, _ := router.getReasoningInfoFromDecision(decision, physicalModel)
	assert.True(t, useReasoning)

	useReasoning, effort := router.getReasoningInfoFromDecision(decision, requestAlias)
	assert.False(t, useReasoning)
	assert.Empty(t, effort)
}
