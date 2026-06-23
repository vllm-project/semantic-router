package looper

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestFusionWeightPolicy_LiveOllama exercises the full Fusion path with the default
// weight grounding policy against locally-running Ollama models via the
// OpenAI-compatible endpoint. The cross-model NLI scorer is stubbed (the candle NLI
// model is not needed for this check), so this validates the orchestration + the
// new soft-weight policy against REAL panel/judge/synthesis LLM calls: the panel is
// scored, nothing is dropped, the groundedness notes reach synthesis, and a real
// answer comes back.
//
// Skipped unless OLLAMA_LIVE=1. Models default to llama3.1:8b + gemma3:12b (no
// "thinking" latency); override with OLLAMA_ENDPOINT / OLLAMA_PANEL / OLLAMA_JUDGE.
//
//	OLLAMA_LIVE=1 go test ./pkg/looper/ -run TestFusionWeightPolicy_LiveOllama -v
func TestFusionWeightPolicy_LiveOllama(t *testing.T) {
	if os.Getenv("OLLAMA_LIVE") != "1" {
		t.Skip("set OLLAMA_LIVE=1 to run against a local Ollama instance")
	}
	endpoint := envOr("OLLAMA_ENDPOINT", "http://localhost:11434/v1/chat/completions")
	judge := envOr("OLLAMA_JUDGE", "llama3.1:8b")
	panelEnv := envOr("OLLAMA_PANEL", "llama3.1:8b,gemma3:12b")
	panelModels := strings.Split(panelEnv, ",")

	// Deterministic NLI stub: flag any answer mentioning "teleport" as contradicted
	// by its peers, everything else entailed. Produces a real score spread so the
	// weight policy has something to surface to the judge — without the candle model.
	withGroundingBackends(t, func(_, hypothesis string) (float32, float32, error) {
		if strings.Contains(strings.ToLower(hypothesis), "teleport") {
			return 0.05, 0.9, nil
		}
		return 0.9, 0.05, nil
	}, nil)

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          judge,
			AnalysisModels: panelModels,
			MaxConcurrent:  1, // serialize local model loads to avoid Ollama thrash
			Grounding: &config.FusionGroundingConfig{
				Enabled:   true,
				Reference: config.FusionGroundingReferencePanel,
				// policy unset -> defaults to weight (no hard-drop)
			},
		},
	}

	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: endpoint}).
		Execute(context.Background(), req)
	require.NoError(t, err)

	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(resp.Body, &body))

	// A real synthesized answer came back from the judge model.
	choices := body["choices"].([]interface{})
	require.NotEmpty(t, choices)
	msg := choices[0].(map[string]interface{})["message"].(map[string]interface{})
	content, _ := msg["content"].(string)
	assert.NotEmpty(t, content, "expected a synthesized final answer from Ollama")
	t.Logf("final answer (truncated): %.300s", content)

	// Grounding ran under the weight policy and kept the WHOLE panel (no drops).
	grounding := body["fusion"].(map[string]interface{})["grounding"].(map[string]interface{})
	assert.Equal(t, config.FusionGroundingPolicyWeight, grounding["policy"])
	scores := grounding["scores"].([]interface{})
	assert.Len(t, scores, len(panelModels), "weight policy must keep every panel response")
	for _, s := range scores {
		dropped, _ := s.(map[string]interface{})["dropped"].(bool)
		assert.False(t, dropped, "weight policy must not drop responses")
	}
	assert.Positive(t, resp.Usage.TotalTokens)
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
