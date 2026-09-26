package looper

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestNonUserVisibleStagesNeverStream(t *testing.T) {
	req := &Request{
		IsStreaming: true,
		ModelRefs: []config.ModelRef{
			{Model: "gpt-4o"},
		},
	}

	nonUserVisibleRoles := []StageRole{
		StageRoleCandidate,
		StageRoleVerifier,
		StageRolePlanning,
		StageRoleAnalysis,
		StageRoleExecution,
	}

	for _, role := range nonUserVisibleRoles {
		t.Run(string(role), func(t *testing.T) {
			ownership := ClassifyStage(req, "test-stage", role, "gpt-4o")
			assert.False(t, ownership.IsFinalUserVisible, "role %s must not be user-visible", role)
			assert.Equal(t, StreamingIneligibleBufferingRequired, ownership.Eligibility)
			assert.Equal(t, BufferingReasonNonUserVisibleStage, ownership.BufferingReason)
		})
	}
}

func TestFusionFinalStageOwnership(t *testing.T) {
	t.Run("eligible native streaming for judge synthesis", func(t *testing.T) {
		req := &Request{
			IsStreaming: true,
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					Model:          "judge-model",
					AnalysisModels: []string{"worker-1", "worker-2"},
				},
			},
			ModelRefs: []config.ModelRef{
				{Model: "worker-1"},
				{Model: "worker-2"},
			},
		}

		ownership := ResolveFinalStageOwnership(req)
		assert.Equal(t, config.DecisionAlgorithmFusion, ownership.AlgorithmType)
		assert.Equal(t, "synthesis", ownership.StageName)
		assert.Equal(t, StageRoleSynthesis, ownership.StageRole)
		assert.Equal(t, "judge-model", ownership.TargetModel)
		assert.True(t, ownership.IsFinalUserVisible)
		assert.Equal(t, StreamingEligible, ownership.Eligibility)
		assert.Equal(t, BufferingReasonNone, ownership.BufferingReason)
	})

	t.Run("eligible when ModelRefs is empty but AnalysisModels and Model are configured", func(t *testing.T) {
		req := &Request{
			IsStreaming: true,
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					Model:          "judge-model",
					AnalysisModels: []string{"worker-1", "worker-2"},
				},
			},
			ModelRefs: nil,
		}

		ownership := ResolveFinalStageOwnership(req)
		assert.Equal(t, config.DecisionAlgorithmFusion, ownership.AlgorithmType)
		assert.Equal(t, "synthesis", ownership.StageName)
		assert.Equal(t, "judge-model", ownership.TargetModel)
		assert.True(t, ownership.IsFinalUserVisible)
		assert.Equal(t, StreamingEligible, ownership.Eligibility)
		assert.Equal(t, BufferingReasonNone, ownership.BufferingReason)
	})

	t.Run("fallback to AnalysisModels[0] when Model is unset and ModelRefs is empty", func(t *testing.T) {
		req := &Request{
			IsStreaming: true,
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					AnalysisModels: []string{"default-judge-worker", "worker-2"},
				},
			},
			ModelRefs: nil,
		}

		ownership := ResolveFinalStageOwnership(req)
		assert.Equal(t, "default-judge-worker", ownership.TargetModel)
		assert.Equal(t, StreamingEligible, ownership.Eligibility)
	})

	t.Run("fallback to buffering when non-streaming", func(t *testing.T) {
		req := &Request{
			IsStreaming: false,
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					Model: "judge-model",
				},
			},
			ModelRefs: []config.ModelRef{{Model: "worker-1"}},
		}

		ownership := ResolveFinalStageOwnership(req)
		assert.True(t, ownership.IsFinalUserVisible)
		assert.Equal(t, StreamingIneligibleBufferingRequired, ownership.Eligibility)
		assert.Equal(t, BufferingReasonNonStreamingRequest, ownership.BufferingReason)
	})

	t.Run("fallback to buffering when output contract requires JSON action transformation", func(t *testing.T) {
		req := &Request{
			IsStreaming: true,
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					Model: "judge-model",
				},
			},
			ModelRefs: []config.ModelRef{{Model: "worker-1"}},
			OutputContractSpec: &config.OutputContractSpec{
				Type: config.OutputContractTypeStructuredJSON,
				JSONSchema: &config.OutputContractJSONSchemaSpec{
					SchemaRef: config.OutputContractJSONTerminalActionV1,
				},
			},
		}

		ownership := ResolveFinalStageOwnership(req)
		assert.True(t, ownership.IsFinalUserVisible)
		assert.Equal(t, StreamingIneligibleBufferingRequired, ownership.Eligibility)
		assert.Equal(t, BufferingReasonOutputContractTransformation, ownership.BufferingReason)
	})

	t.Run("intermediate fusion stages are isolated", func(t *testing.T) {
		req := &Request{
			IsStreaming: true,
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
			},
		}

		panelOwnership := ClassifyStage(req, "panel", StageRoleAnalysis, "worker-1")
		assert.False(t, panelOwnership.IsFinalUserVisible)
		assert.Equal(t, StreamingIneligibleBufferingRequired, panelOwnership.Eligibility)
		assert.Equal(t, BufferingReasonNonUserVisibleStage, panelOwnership.BufferingReason)

		groundingOwnership := ClassifyStage(req, "grounding", StageRoleVerifier, "verifier-model")
		assert.False(t, groundingOwnership.IsFinalUserVisible)
		assert.Equal(t, StreamingIneligibleBufferingRequired, groundingOwnership.Eligibility)
		assert.Equal(t, BufferingReasonNonUserVisibleStage, groundingOwnership.BufferingReason)
	})
}

func TestReMoMFinalStageOwnership(t *testing.T) {
	t.Run("eligible native streaming for synthesis round", func(t *testing.T) {
		req := &Request{
			IsStreaming: true,
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmReMoM,
				ReMoM: &config.ReMoMAlgorithmConfig{
					BreadthSchedule:   []int{2},
					ModelDistribution: remomDistributionWeighted,
					ShuffleSeed:       42,
				},
			},
			ModelRefs: []config.ModelRef{
				{Model: "model-a", Weight: 1.0},
				{Model: "model-b", Weight: 0.0},
			},
		}

		ownership := ResolveFinalStageOwnership(req)
		assert.Equal(t, config.DecisionAlgorithmReMoM, ownership.AlgorithmType)
		assert.Equal(t, "synthesis", ownership.StageName)
		assert.Equal(t, StageRoleSynthesis, ownership.StageRole)
		assert.Equal(t, "model-a", ownership.TargetModel)
		assert.True(t, ownership.IsFinalUserVisible)
		assert.Equal(t, StreamingEligible, ownership.Eligibility)
		assert.Equal(t, BufferingReasonNone, ownership.BufferingReason)
	})

	t.Run("respects explicit synthesis_model override", func(t *testing.T) {
		req := &Request{
			IsStreaming: true,
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmReMoM,
				ReMoM: &config.ReMoMAlgorithmConfig{
					BreadthSchedule: []int{2},
					SynthesisModel:  "special-synthesis-model",
				},
			},
			ModelRefs: []config.ModelRef{
				{Model: "model-a", Weight: 1.0},
			},
		}

		ownership := ResolveFinalStageOwnership(req)
		assert.Equal(t, "special-synthesis-model", ownership.TargetModel)
		assert.Equal(t, StreamingEligible, ownership.Eligibility)
	})

	t.Run("eligible when ModelRefs is empty but SynthesisModel is configured", func(t *testing.T) {
		req := &Request{
			IsStreaming: true,
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmReMoM,
				ReMoM: &config.ReMoMAlgorithmConfig{
					SynthesisModel: "algorithm-owned-synthesis-model",
				},
			},
			ModelRefs: nil,
		}

		ownership := ResolveFinalStageOwnership(req)
		assert.Equal(t, "algorithm-owned-synthesis-model", ownership.TargetModel)
		assert.Equal(t, StageRoleSynthesis, ownership.StageRole)
		assert.True(t, ownership.StageRole.IsUserVisible())
		assert.True(t, ownership.IsFinalUserVisible)
		assert.Equal(t, StreamingEligible, ownership.Eligibility)
	})

	t.Run("fallback to buffering when output contract requires single choice", func(t *testing.T) {
		req := &Request{
			IsStreaming: true,
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmReMoM,
				ReMoM: &config.ReMoMAlgorithmConfig{
					BreadthSchedule: []int{2},
				},
			},
			ModelRefs: []config.ModelRef{{Model: "model-a"}},
			OutputContractSpec: &config.OutputContractSpec{
				Type: config.OutputContractTypeChoice,
			},
		}

		ownership := ResolveFinalStageOwnership(req)
		assert.True(t, ownership.IsFinalUserVisible)
		assert.Equal(t, StreamingIneligibleBufferingRequired, ownership.Eligibility)
		assert.Equal(t, BufferingReasonOutputContractTransformation, ownership.BufferingReason)
	})
}

func TestWorkflowsFinalStageOwnership(t *testing.T) {
	t.Run("eligible native streaming for workflow synthesis", func(t *testing.T) {
		req := &Request{
			IsStreaming: true,
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmWorkflows,
				Workflows: &config.WorkflowsAlgorithmConfig{
					Planner: config.WorkflowPlannerConfig{
						Model: "planner-model",
					},
					Final: config.WorkflowFinalConfig{
						Model: "final-synthesizer",
					},
				},
			},
			ModelRefs: []config.ModelRef{
				{Model: "worker-1"},
				{Model: "worker-2"},
			},
		}

		ownership := ResolveFinalStageOwnership(req)
		assert.Equal(t, config.DecisionAlgorithmWorkflows, ownership.AlgorithmType)
		assert.Equal(t, "synthesis", ownership.StageName)
		assert.Equal(t, StageRoleSynthesis, ownership.StageRole)
		assert.Equal(t, "final-synthesizer", ownership.TargetModel)
		assert.True(t, ownership.IsFinalUserVisible)
		assert.Equal(t, StreamingEligible, ownership.Eligibility)
		assert.Equal(t, BufferingReasonNone, ownership.BufferingReason)
	})

	t.Run("eligible when ModelRefs is empty but Final.Model is configured", func(t *testing.T) {
		req := &Request{
			IsStreaming: true,
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmWorkflows,
				Workflows: &config.WorkflowsAlgorithmConfig{
					Final: config.WorkflowFinalConfig{
						Model: "workflow-final-synth",
					},
				},
			},
			ModelRefs: nil,
		}

		ownership := ResolveFinalStageOwnership(req)
		assert.Equal(t, "workflow-final-synth", ownership.TargetModel)
		assert.Equal(t, StageRoleSynthesis, ownership.StageRole)
		assert.True(t, ownership.StageRole.IsUserVisible())
		assert.True(t, ownership.IsFinalUserVisible)
		assert.Equal(t, StreamingEligible, ownership.Eligibility)
	})

	t.Run("intermediate planning and execution stages are isolated", func(t *testing.T) {
		req := &Request{
			IsStreaming: true,
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmWorkflows,
			},
		}

		plannerOwnership := ClassifyStage(req, "planning", StageRolePlanning, "planner-model")
		assert.False(t, plannerOwnership.IsFinalUserVisible)
		assert.Equal(t, StreamingIneligibleBufferingRequired, plannerOwnership.Eligibility)
		assert.Equal(t, BufferingReasonNonUserVisibleStage, plannerOwnership.BufferingReason)

		workerOwnership := ClassifyStage(req, "step_1", StageRoleExecution, "worker-1")
		assert.False(t, workerOwnership.IsFinalUserVisible)
		assert.Equal(t, StreamingIneligibleBufferingRequired, workerOwnership.Eligibility)
		assert.Equal(t, BufferingReasonNonUserVisibleStage, workerOwnership.BufferingReason)
	})
}

func TestConfidenceSelectionPendingBuffering(t *testing.T) {
	req := &Request{
		IsStreaming: true,
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmConfidence,
			Confidence: &config.ConfidenceAlgorithmConfig{
				Threshold: 0.8,
			},
		},
		ModelRefs: []config.ModelRef{
			{Model: "small-model"},
			{Model: "large-model"},
		},
	}

	ownership := ResolveFinalStageOwnership(req)
	assert.Equal(t, config.DecisionAlgorithmConfidence, ownership.AlgorithmType)
	assert.Equal(t, StageRoleDirect, ownership.StageRole)
	assert.True(t, ownership.StageRole.IsUserVisible())
	assert.True(t, ownership.IsFinalUserVisible)
	assert.Equal(t, StreamingIneligibleBufferingRequired, ownership.Eligibility)
	assert.Equal(t, BufferingReasonSelectionPending, ownership.BufferingReason)
}

func TestRatingsMultiChoiceBuffering(t *testing.T) {
	req := &Request{
		IsStreaming: true,
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmRatings,
		},
		ModelRefs: []config.ModelRef{
			{Model: "model-a"},
			{Model: "model-b"},
		},
	}

	ownership := ResolveFinalStageOwnership(req)
	assert.Equal(t, config.DecisionAlgorithmRatings, ownership.AlgorithmType)
	assert.Equal(t, StageRoleSynthesis, ownership.StageRole)
	assert.True(t, ownership.StageRole.IsUserVisible())
	assert.True(t, ownership.IsFinalUserVisible)
	assert.Equal(t, StreamingIneligibleBufferingRequired, ownership.Eligibility)
	assert.Equal(t, BufferingReasonMultiChoiceOutput, ownership.BufferingReason)
}

func TestBaseLooperStageOwnership(t *testing.T) {
	t.Run("single model is direct user-visible and stream eligible", func(t *testing.T) {
		req := &Request{
			IsStreaming: true,
			Algorithm: &config.AlgorithmConfig{
				Type: AlgorithmTypeBase,
			},
			ModelRefs: []config.ModelRef{
				{Model: "single-model"},
			},
		}

		ownership := ResolveFinalStageOwnership(req)
		assert.Equal(t, AlgorithmTypeBase, ownership.AlgorithmType)
		assert.Equal(t, "direct", ownership.StageName)
		assert.Equal(t, StageRoleDirect, ownership.StageRole)
		assert.Equal(t, "single-model", ownership.TargetModel)
		assert.True(t, ownership.IsFinalUserVisible)
		assert.Equal(t, StreamingEligible, ownership.Eligibility)
		assert.Equal(t, BufferingReasonNone, ownership.BufferingReason)
	})

	t.Run("multiple models sequential aggregation requires buffering", func(t *testing.T) {
		req := &Request{
			IsStreaming: true,
			Algorithm: &config.AlgorithmConfig{
				Type: AlgorithmTypeBase,
			},
			ModelRefs: []config.ModelRef{
				{Model: "model-1"},
				{Model: "model-2"},
			},
		}

		ownership := ResolveFinalStageOwnership(req)
		assert.True(t, ownership.IsFinalUserVisible)
		assert.Equal(t, StreamingIneligibleBufferingRequired, ownership.Eligibility)
		assert.Equal(t, BufferingReasonMultiModelAggregation, ownership.BufferingReason)
	})
}

func TestUnsupportedAndEmptyCases(t *testing.T) {
	t.Run("nil request or empty models", func(t *testing.T) {
		nilOwnership := ResolveFinalStageOwnership(nil)
		assert.Equal(t, StreamingIneligibleBufferingRequired, nilOwnership.Eligibility)
		assert.Equal(t, BufferingReasonNoModels, nilOwnership.BufferingReason)

		emptyOwnership := ResolveFinalStageOwnership(&Request{ModelRefs: nil})
		assert.Equal(t, StreamingIneligibleBufferingRequired, emptyOwnership.Eligibility)
		assert.Equal(t, BufferingReasonNoModels, emptyOwnership.BufferingReason)
	})

	t.Run("unsupported algorithm type", func(t *testing.T) {
		req := &Request{
			IsStreaming: true,
			Algorithm: &config.AlgorithmConfig{
				Type: "non_existent_algorithm",
			},
			ModelRefs: []config.ModelRef{{Model: "model-1"}},
		}

		ownership := ResolveFinalStageOwnership(req)
		assert.Equal(t, StreamingIneligibleBufferingRequired, ownership.Eligibility)
		assert.Equal(t, BufferingReasonUnsupportedAlgorithm, ownership.BufferingReason)
	})
}

func TestRequiresOutputContractTransformation(t *testing.T) {
	require.False(t, RequiresOutputContractTransformation(nil))
	require.False(t, RequiresOutputContractTransformation(&config.OutputContractSpec{}))

	// Structured JSON with terminal action
	require.True(t, RequiresOutputContractTransformation(&config.OutputContractSpec{
		Type: config.OutputContractTypeStructuredJSON,
		JSONSchema: &config.OutputContractJSONSchemaSpec{
			SchemaRef: config.OutputContractJSONTerminalActionV1,
		},
	}))

	// Single choice
	require.True(t, RequiresOutputContractTransformation(&config.OutputContractSpec{
		Type: config.OutputContractTypeChoice,
	}))

	// Reference select
	require.True(t, RequiresOutputContractTransformation(&config.OutputContractSpec{
		Type: config.OutputContractTypeReferenceSelect,
	}))

	// General Structured JSON
	require.True(t, RequiresOutputContractTransformation(&config.OutputContractSpec{
		Type: config.OutputContractTypeStructuredJSON,
	}))

	// Render spec
	require.True(t, RequiresOutputContractTransformation(&config.OutputContractSpec{
		Render: &config.OutputContractRenderSpec{Mode: "template"},
	}))

	// Normalize spec
	require.True(t, RequiresOutputContractTransformation(&config.OutputContractSpec{
		Normalize: &config.OutputContractNormalizeSpec{},
	}))

	// Postprocess chain
	require.True(t, RequiresOutputContractTransformation(&config.OutputContractSpec{
		Postprocess: []config.OutputContractPostprocess{{Type: "dereference"}},
	}))
}
