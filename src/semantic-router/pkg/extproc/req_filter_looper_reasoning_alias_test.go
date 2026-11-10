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
			Entrypoints: []config.EntrypointMapping{{ModelNames: []string{requestAlias}, Recipe: config.DefaultRecipeName}},
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

func TestLooperDispatchDoesNotInheritOuterLoRAOwner(t *testing.T) {
	selected := config.ModelRef{Model: "base-b", LoRAName: "shared"}
	ordinaryRequest := &RequestContext{VSRSelectedCandidate: &selected}
	hop := config.ModelRef{Model: "base-a", LoRAName: "shared"}
	hopDecision := &config.Decision{ModelRefs: []config.ModelRef{hop}}
	looperRequest := &RequestContext{
		LooperRequest:        true,
		VSRSelectedCandidate: &selected,
		VSREligibleModelRefs: []config.ModelRef{selected},
		VSRSelectedDecision:  hopDecision,
	}

	assert.Equal(t, "base-b", ordinaryRequest.backendModelForCandidate("shared"))
	assert.Equal(t, "shared", looperRequest.backendModelForCandidate("shared"))
	assert.Same(t, hopDecision, looperRequest.decisionForBackend("shared"))
}
