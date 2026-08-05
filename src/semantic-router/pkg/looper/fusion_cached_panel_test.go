package looper

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"hash/fnv"
	"math/rand"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// These tests cover the CachedPanel seam used by the paired multi-arm fusion
// evaluator (bench/grounded_fusion): generate the panel once, then synthesize
// every arm from the byte-identical panel so deltas isolate the intervention.

// cachedTestPanel builds panel responses with explicit model names + usage, as
// the fusioneval driver loads them from the panel cache on disk.
func cachedTestPanel() []*ModelResponse {
	return []*ModelResponse{
		{Model: "panel-a", Content: "good grounded answer", Usage: TokenUsage{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12}},
		{Model: "panel-b", Content: "bad contradicted answer", Usage: TokenUsage{PromptTokens: 20, CompletionTokens: 3, TotalTokens: 23}},
	}
}

// placeboNLI is a deterministic seeded-random NLI: scores are reproducible for a
// given (seed, premise, hypothesis) but carry no real signal. It mirrors the
// random-weight placebo arm in the fusioneval driver, isolating "does the score
// help" from "does any weighting help".
func placeboNLI(seed uint64) NLIClassifyFunc {
	return func(premise, hypothesis string) (float32, float32, error) {
		h := fnv.New64a()
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], seed)
		_, _ = h.Write(b[:])
		_, _ = h.Write([]byte(premise))
		_, _ = h.Write([]byte{0})
		_, _ = h.Write([]byte(hypothesis))
		r := rand.New(rand.NewSource(int64(h.Sum64()))) //nolint:gosec // deterministic test seed, overflow harmless
		entail := float32(r.Float64())
		contradict := float32(r.Float64() * (1 - float64(entail)))
		return entail, contradict, nil
	}
}

// runCachedPanelArm runs Execute against a fixed cached panel with the given
// grounding config and NLI backend, capturing every prompt the judge saw. The
// stub fails the test if any panel model is called live, proving the short-circuit.
func runCachedPanelArm(
	t *testing.T,
	p []*ModelResponse,
	grounding *config.FusionGroundingConfig,
	nli NLIClassifyFunc,
) (body map[string]interface{}, judgePrompts []string) {
	t.Helper()
	if nli != nil {
		withGroundingBackends(t, nli, nil)
	}
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		if model != "judge" {
			t.Errorf("cached panel must not call panel model %q live", model)
			return "unexpected", http.StatusInternalServerError
		}
		judgePrompts = append(judgePrompts, prompt)
		if strings.Contains(prompt, "return only valid JSON") {
			return `{"consensus":[],"contradictions":[],"partial_coverage":[],"unique_insights":[],"blind_spots":[]}`, http.StatusOK
		}
		return "final answer", http.StatusOK
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.CachedPanel = p
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a", "panel-b"},
			Grounding:      grounding,
		},
	}
	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(resp.Body, &body))
	return body, judgePrompts
}

func groundingBlock(t *testing.T, body map[string]interface{}) (map[string]interface{}, bool) {
	t.Helper()
	fusion, ok := body["fusion"].(map[string]interface{})
	require.True(t, ok, "response should carry a fusion trace")
	g, ok := fusion["grounding"].(map[string]interface{})
	return g, ok
}

func groundingScoresFromBody(t *testing.T, body map[string]interface{}) []float64 {
	t.Helper()
	g, ok := groundingBlock(t, body)
	require.True(t, ok, "expected a grounding block")
	raw := g["scores"].([]interface{})
	out := make([]float64, 0, len(raw))
	for _, s := range raw {
		out = append(out, s.(map[string]interface{})["score"].(float64))
	}
	return out
}

func panelContentsFromBody(t *testing.T, body map[string]interface{}) []string {
	t.Helper()
	fusion := body["fusion"].(map[string]interface{})
	raw, _ := fusion["responses"].([]interface{})
	out := make([]string, 0, len(raw))
	for _, r := range raw {
		out = append(out, r.(map[string]interface{})["content"].(string))
	}
	return out
}

// TestFusionExecute_CachedPanelSkipsLiveCalls verifies a non-nil CachedPanel
// short-circuits the live panel calls (the stub errors if any panel model is
// called) and that usage still reflects the cached panel cost plus the judge.
func TestFusionExecute_CachedPanelSkipsLiveCalls(t *testing.T) {
	p := cachedTestPanel()
	body, judgePrompts := runCachedPanelArm(t, p, nil, nil) // grounding off

	// Exactly two judge calls: analysis + final synthesis.
	require.Len(t, judgePrompts, 2)

	choices := body["choices"].([]interface{})
	msg := choices[0].(map[string]interface{})["message"].(map[string]interface{})
	assert.Equal(t, "final answer", msg["content"])

	// Usage = cached panel (12+23) + two judge calls (34 each via fusionTestUsage).
	usage := body["usage"].(map[string]interface{})
	assert.Equal(t, float64(12+23+34+34), usage["total_tokens"])
}

// TestFusionExecute_CachedPanel_ArmIsolation_BvsC proves arms B (plain fusion)
// and C (weight) synthesize from the identical cached panel and differ ONLY in
// the grounding stage: B has no grounding block and no weighting directive; C
// scores every response, reports policy=weight, and annotates synthesis.
func TestFusionExecute_CachedPanel_ArmIsolation_BvsC(t *testing.T) {
	p := cachedTestPanel()

	// Arm B: plain fusion (grounding disabled).
	bodyB, judgeB := runCachedPanelArm(t, p, nil, nil)

	// Arm C: weight (grounding on, panel mode, policy defaults to weight).
	highEntail := func(_, _ string) (float32, float32, error) { return 0.9, 0.05, nil }
	bodyC, judgeC := runCachedPanelArm(t, p, &config.FusionGroundingConfig{
		Enabled:   true,
		Reference: config.FusionGroundingReferencePanel,
	}, highEntail)

	// Both arms feed the IDENTICAL panel content to the judge analysis stage.
	for _, r := range p {
		assert.Contains(t, judgeB[0], r.Content)
		assert.Contains(t, judgeC[0], r.Content)
	}

	// Arm B: no grounding block, no weighting directive at synthesis.
	_, hasGroundingB := groundingBlock(t, bodyB)
	assert.False(t, hasGroundingB, "plain fusion must not emit a grounding block")
	assert.NotContains(t, judgeB[1], "Weight each panel answer")

	// Arm C: grounding block with a score per response, policy=weight, and the
	// weighting directive on the final synthesis prompt.
	gC, ok := groundingBlock(t, bodyC)
	require.True(t, ok)
	assert.Equal(t, config.FusionGroundingPolicyWeight, gC["policy"])
	assert.Len(t, gC["scores"].([]interface{}), len(p))
	assert.Contains(t, judgeC[1], "Weight each panel answer")
}

// TestFusionExecute_CachedPanel_PlaceboMechanism proves arm C (real NLI) and arm
// D (seeded-random placebo NLI) synthesize from the identical panel but produce
// different groundedness scores — so the A/B isolates the score's signal from the
// mere act of weighting.
func TestFusionExecute_CachedPanel_PlaceboMechanism(t *testing.T) {
	p := cachedTestPanel()
	grounding := &config.FusionGroundingConfig{
		Enabled:   true,
		Reference: config.FusionGroundingReferencePanel,
	}

	realNLI := func(_, hypothesis string) (float32, float32, error) {
		if strings.Contains(hypothesis, "bad") {
			return 0.1, 0.8, nil
		}
		return 0.9, 0.05, nil
	}
	bodyC, _ := runCachedPanelArm(t, p, grounding, realNLI)
	bodyD, _ := runCachedPanelArm(t, p, grounding, placeboNLI(7))

	scoresC := groundingScoresFromBody(t, bodyC)
	scoresD := groundingScoresFromBody(t, bodyD)
	require.Len(t, scoresC, len(p))
	require.Len(t, scoresD, len(p))

	// Same panel content reaches the judge in both arms.
	assert.Equal(t, panelContentsFromBody(t, bodyC), panelContentsFromBody(t, bodyD))
	// But the scores differ: the placebo weights on noise, not the real signal.
	assert.NotEqual(t, scoresC, scoresD)
}

// TestPlaceboNLI_DeterministicAndSpread guards the placebo's two required
// properties: reproducible for a fixed seed, and non-constant across inputs.
func TestPlaceboNLI_DeterministicAndSpread(t *testing.T) {
	nli := placeboNLI(42)
	e1, c1, err := nli("premise one", "hypothesis one")
	require.NoError(t, err)
	e2, c2, err := nli("premise one", "hypothesis one")
	require.NoError(t, err)
	assert.Equal(t, e1, e2, "same inputs must yield identical scores")
	assert.Equal(t, c1, c2)

	e3, _, err := nli("premise one", "a different hypothesis")
	require.NoError(t, err)
	assert.NotEqual(t, e1, e3, "different inputs must yield different scores")
}
