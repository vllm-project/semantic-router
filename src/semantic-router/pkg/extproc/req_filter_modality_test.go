package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// =============================================================================
// resolveModalityModelsFromDecision Tests
// =============================================================================

func TestResolveModalityModelsFromDecision_Both(t *testing.T) {
	decision := &config.Decision{
		Name: "text_and_image",
		ModelRefs: []config.ModelRef{
			{Model: "Qwen/Qwen2.5-14B-Instruct", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPtr(false)}},
			{Model: "Qwen/Qwen-Image", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPtr(false)}},
		},
	}
	modelConfig := map[string]config.ModelParams{
		"Qwen/Qwen2.5-14B-Instruct": {Modality: "ar"},
		"Qwen/Qwen-Image":           {Modality: "diffusion"},
	}

	ar, diffusion, err := resolveModalityModelsFromDecision(decision, modelConfig)
	require.NoError(t, err)
	assert.Equal(t, "Qwen/Qwen2.5-14B-Instruct", ar)
	assert.Equal(t, "Qwen/Qwen-Image", diffusion)
}

func TestResolveModalityModelsFromDecision_DiffusionOnly(t *testing.T) {
	decision := &config.Decision{
		Name: "image_generation",
		ModelRefs: []config.ModelRef{
			{Model: "Qwen/Qwen-Image", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPtr(false)}},
		},
	}
	modelConfig := map[string]config.ModelParams{
		"Qwen/Qwen-Image": {Modality: "diffusion"},
	}

	ar, diffusion, err := resolveModalityModelsFromDecision(decision, modelConfig)
	require.NoError(t, err)
	assert.Equal(t, "", ar, "AR model should be empty for diffusion-only decision")
	assert.Equal(t, "Qwen/Qwen-Image", diffusion)
}

func TestResolveModalityModelsFromDecision_AROnly(t *testing.T) {
	decision := &config.Decision{
		Name: "text_generation",
		ModelRefs: []config.ModelRef{
			{Model: "Qwen/Qwen2.5-14B-Instruct", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPtr(false)}},
		},
	}
	modelConfig := map[string]config.ModelParams{
		"Qwen/Qwen2.5-14B-Instruct": {Modality: "ar"},
	}

	ar, diffusion, err := resolveModalityModelsFromDecision(decision, modelConfig)
	require.NoError(t, err)
	assert.Equal(t, "Qwen/Qwen2.5-14B-Instruct", ar)
	assert.Equal(t, "", diffusion, "Diffusion model should be empty for AR-only decision")
}

func TestResolveModalityModelsFromDecision_NilDecision(t *testing.T) {
	_, _, err := resolveModalityModelsFromDecision(nil, map[string]config.ModelParams{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "decision is nil")
}

func TestResolveModalityModelsFromDecision_EmptyModelRefs(t *testing.T) {
	decision := &config.Decision{
		Name:      "empty_decision",
		ModelRefs: []config.ModelRef{},
	}
	_, _, err := resolveModalityModelsFromDecision(decision, map[string]config.ModelParams{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "has no modelRefs")
}

func TestResolveModalityModelsFromDecision_ModelNotInConfig(t *testing.T) {
	decision := &config.Decision{
		Name: "unknown_models",
		ModelRefs: []config.ModelRef{
			{Model: "model-that-does-not-exist", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPtr(false)}},
		},
	}
	modelConfig := map[string]config.ModelParams{
		"Qwen/Qwen2.5-14B-Instruct": {Modality: "ar"},
	}

	ar, diffusion, err := resolveModalityModelsFromDecision(decision, modelConfig)
	require.NoError(t, err, "should not error, just return empty strings for unresolvable models")
	assert.Equal(t, "", ar)
	assert.Equal(t, "", diffusion)
}

func TestResolveModalityModelsFromDecision_MultipleARModels_PicksFirst(t *testing.T) {
	// When multiple AR models are in ModelRefs, the first one should be selected
	decision := &config.Decision{
		Name: "multi_ar",
		ModelRefs: []config.ModelRef{
			{Model: "model-ar-1", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPtr(false)}},
			{Model: "model-ar-2", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPtr(false)}},
			{Model: "model-diffusion-1", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPtr(false)}},
		},
	}
	modelConfig := map[string]config.ModelParams{
		"model-ar-1":        {Modality: "ar"},
		"model-ar-2":        {Modality: "ar"},
		"model-diffusion-1": {Modality: "diffusion"},
	}

	ar, diffusion, err := resolveModalityModelsFromDecision(decision, modelConfig)
	require.NoError(t, err)
	assert.Equal(t, "model-ar-1", ar, "should pick first AR model from ModelRefs order")
	assert.Equal(t, "model-diffusion-1", diffusion)
}

func TestResolveModalityModelsFromDecision_MultipleDiffusionModels_PicksFirst(t *testing.T) {
	decision := &config.Decision{
		Name: "multi_diffusion",
		ModelRefs: []config.ModelRef{
			{Model: "model-ar-1", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPtr(false)}},
			{Model: "model-diffusion-1", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPtr(false)}},
			{Model: "model-diffusion-2", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPtr(false)}},
		},
	}
	modelConfig := map[string]config.ModelParams{
		"model-ar-1":        {Modality: "ar"},
		"model-diffusion-1": {Modality: "diffusion"},
		"model-diffusion-2": {Modality: "diffusion"},
	}

	ar, diffusion, err := resolveModalityModelsFromDecision(decision, modelConfig)
	require.NoError(t, err)
	assert.Equal(t, "model-ar-1", ar)
	assert.Equal(t, "model-diffusion-1", diffusion, "should pick first diffusion model from ModelRefs order")
}

func TestResolveModalityModelsFromDecision_ModelWithNoModality(t *testing.T) {
	// Models in model_config without a modality field should be skipped
	decision := &config.Decision{
		Name: "mixed_modality",
		ModelRefs: []config.ModelRef{
			{Model: "model-no-modality", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPtr(false)}},
			{Model: "model-ar", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPtr(false)}},
		},
	}
	modelConfig := map[string]config.ModelParams{
		"model-no-modality": {Modality: ""}, // no modality set
		"model-ar":          {Modality: "ar"},
	}

	ar, diffusion, err := resolveModalityModelsFromDecision(decision, modelConfig)
	require.NoError(t, err)
	assert.Equal(t, "model-ar", ar)
	assert.Equal(t, "", diffusion)
}

// =============================================================================
// resolveARModelEndpoint Tests
// =============================================================================

func newRouterConfigWithBackend(modelConfig map[string]config.ModelParams, endpoints []config.VLLMEndpoint, imageGenBackends map[string]config.ImageGenBackendEntry) *config.RouterConfig {
	cfg := &config.RouterConfig{}
	cfg.BackendModels.ModelConfig = modelConfig
	cfg.BackendModels.VLLMEndpoints = endpoints
	cfg.BackendModels.ImageGenBackends = imageGenBackends
	return cfg
}

func TestResolveARModelEndpoint_Success(t *testing.T) {
	cfg := newRouterConfigWithBackend(
		map[string]config.ModelParams{
			"Qwen/Qwen2.5-14B-Instruct": {
				PreferredEndpoints: []string{"vllm-ar"},
				Modality:           "ar",
			},
		},
		[]config.VLLMEndpoint{
			{Name: "vllm-ar", Address: "localhost", Port: 8000},
		},
		nil,
	)

	endpoint, err := resolveARModelEndpoint(cfg, "Qwen/Qwen2.5-14B-Instruct")
	require.NoError(t, err)
	assert.Equal(t, "http://localhost:8000/v1", endpoint)
}

func TestResolveARModelEndpoint_EmptyModelName(t *testing.T) {
	cfg := newRouterConfigWithBackend(nil, nil, nil)
	_, err := resolveARModelEndpoint(cfg, "")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "AR model name is required")
}

func TestResolveARModelEndpoint_ModelNotInConfig(t *testing.T) {
	cfg := newRouterConfigWithBackend(
		map[string]config.ModelParams{},
		nil, nil,
	)
	_, err := resolveARModelEndpoint(cfg, "nonexistent-model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found in model_config")
}

func TestResolveARModelEndpoint_NoPreferredEndpoints(t *testing.T) {
	cfg := newRouterConfigWithBackend(
		map[string]config.ModelParams{
			"model-a": {
				PreferredEndpoints: []string{},
				Modality:           "ar",
			},
		},
		nil, nil,
	)
	_, err := resolveARModelEndpoint(cfg, "model-a")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "has no preferred_endpoints")
}

func TestResolveARModelEndpoint_EndpointNotFound(t *testing.T) {
	cfg := newRouterConfigWithBackend(
		map[string]config.ModelParams{
			"model-a": {
				PreferredEndpoints: []string{"nonexistent-endpoint"},
				Modality:           "ar",
			},
		},
		[]config.VLLMEndpoint{
			{Name: "vllm-other", Address: "localhost", Port: 8000},
		},
		nil,
	)
	_, err := resolveARModelEndpoint(cfg, "model-a")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found in vllm_endpoints")
}

func TestResolveARModelEndpoint_SelectsFirstPreferredEndpoint(t *testing.T) {
	cfg := newRouterConfigWithBackend(
		map[string]config.ModelParams{
			"model-a": {
				PreferredEndpoints: []string{"ep-primary", "ep-secondary"},
				Modality:           "ar",
			},
		},
		[]config.VLLMEndpoint{
			{Name: "ep-primary", Address: "primary.local", Port: 8000},
			{Name: "ep-secondary", Address: "secondary.local", Port: 8001},
		},
		nil,
	)

	endpoint, err := resolveARModelEndpoint(cfg, "model-a")
	require.NoError(t, err)
	assert.Equal(t, "http://primary.local:8000/v1", endpoint, "should select first preferred endpoint")
}

// =============================================================================
// resolveDiffusionBackend Tests
// =============================================================================

func TestResolveDiffusionBackend_Success(t *testing.T) {
	cfg := newRouterConfigWithBackend(
		map[string]config.ModelParams{
			"Qwen/Qwen-Image": {
				Modality:        "diffusion",
				ImageGenBackend: "vllm_omni_local",
			},
		},
		nil,
		map[string]config.ImageGenBackendEntry{
			"vllm_omni_local": {
				Type:    "vllm_omni",
				BaseURL: "http://localhost:8001",
				Model:   "Qwen/Qwen-Image",
			},
		},
	)

	entry, err := resolveDiffusionBackend(cfg, "Qwen/Qwen-Image")
	require.NoError(t, err)
	assert.Equal(t, "vllm_omni", entry.Type)
	assert.Equal(t, "http://localhost:8001", entry.BaseURL)
	assert.Equal(t, "Qwen/Qwen-Image", entry.Model)
}

func TestResolveDiffusionBackend_EmptyModelName(t *testing.T) {
	cfg := newRouterConfigWithBackend(nil, nil, nil)
	_, err := resolveDiffusionBackend(cfg, "")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "diffusion model name is required")
}

func TestResolveDiffusionBackend_ModelNotInConfig(t *testing.T) {
	cfg := newRouterConfigWithBackend(
		map[string]config.ModelParams{},
		nil, nil,
	)
	_, err := resolveDiffusionBackend(cfg, "nonexistent-model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found in model_config")
}

func TestResolveDiffusionBackend_NoImageGenBackend(t *testing.T) {
	cfg := newRouterConfigWithBackend(
		map[string]config.ModelParams{
			"model-a": {
				Modality:        "diffusion",
				ImageGenBackend: "", // not configured
			},
		},
		nil, nil,
	)
	_, err := resolveDiffusionBackend(cfg, "model-a")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "has no image_gen_backend configured")
}

func TestResolveDiffusionBackend_BackendNotFound(t *testing.T) {
	cfg := newRouterConfigWithBackend(
		map[string]config.ModelParams{
			"model-a": {
				Modality:        "diffusion",
				ImageGenBackend: "nonexistent_backend",
			},
		},
		nil,
		map[string]config.ImageGenBackendEntry{
			"other_backend": {Type: "openai"},
		},
	)
	_, err := resolveDiffusionBackend(cfg, "model-a")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found in image_gen_backends")
}

func TestResolveDiffusionBackend_NoFallbackScan(t *testing.T) {
	// Even if other diffusion models exist in model_config,
	// passing a model name that doesn't exist should NOT fall back to scanning.
	cfg := newRouterConfigWithBackend(
		map[string]config.ModelParams{
			"model-diffusion": {
				Modality:        "diffusion",
				ImageGenBackend: "backend-1",
			},
		},
		nil,
		map[string]config.ImageGenBackendEntry{
			"backend-1": {Type: "vllm_omni"},
		},
	)

	// Requesting a different model name should fail, not fall back to model-diffusion
	_, err := resolveDiffusionBackend(cfg, "wrong-model-name")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found in model_config")
}
