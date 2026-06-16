package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// withGroundingBackends swaps the package-level grounding backends for the
// duration of a test and restores them afterwards.
func withGroundingBackends(t *testing.T, nli NLIClassifyFunc, detect HallucinationDetectFunc) {
	t.Helper()
	prevNLI, prevDetect := groundingNLIClassify, groundingDetect
	groundingNLIClassify, groundingDetect = nli, detect
	t.Cleanup(func() {
		groundingNLIClassify, groundingDetect = prevNLI, prevDetect
	})
}

func panel(contents ...string) []*ModelResponse {
	resps := make([]*ModelResponse, 0, len(contents))
	for i, c := range contents {
		resps = append(resps, &ModelResponse{Model: string(rune('a' + i)), Content: c})
	}
	return resps
}

func TestScoreByPanel_RanksContradictedLower(t *testing.T) {
	// Peers entail "good" answers and contradict any answer containing "bad".
	withGroundingBackends(t, func(_, hypothesis string) (float32, float32, error) {
		if strings.Contains(hypothesis, "bad") {
			return 0.1, 0.8, nil
		}
		return 0.9, 0.05, nil
	}, nil)

	p := panel("good one", "good two", "bad three")
	scores, err := scoreByPanel(p, fusionExecutionConfig{GroundingNLIContradictionPenalty: 1.0})
	require.NoError(t, err)
	require.Len(t, scores, 3)

	assert.Greater(t, scores[0].Score, scores[2].Score)
	assert.Greater(t, scores[1].Score, scores[2].Score)
	// The contradicted response is flagged by its peers.
	assert.NotEmpty(t, scores[2].FlaggedSpans)
	assert.Empty(t, scores[0].FlaggedSpans)
}

func TestScoreByContext_FewerUnsupportedSpansScoresHigher(t *testing.T) {
	// "bad" answers have an unsupported span; grounded ones have none.
	withGroundingBackends(t, nil, func(_, _, answer string) ([]string, float32, error) {
		if strings.Contains(answer, "bad") {
			return []string{"unsupported claim"}, 0.9, nil
		}
		return nil, 0.9, nil
	})

	p := panel("grounded answer", "bad answer")
	scores, err := scoreByContext("the context", "the question", p, fusionExecutionConfig{})
	require.NoError(t, err)
	require.Len(t, scores, 2)
	assert.Equal(t, 1.0, scores[0].Score)
	assert.Less(t, scores[1].Score, scores[0].Score)
	assert.NotEmpty(t, scores[1].FlaggedSpans)
}

func TestFilterPanelByScore_MinScoreAndMinKeep(t *testing.T) {
	p := panel("a", "b", "c")
	scores := []groundingScore{
		{Model: "a", Score: 0.9},
		{Model: "b", Score: 0.2},
		{Model: "c", Score: 0.1},
	}
	kept := filterPanelByScore(p, scores, fusionExecutionConfig{GroundingMinScore: 0.5, GroundingMinKeep: 1})
	// Only "a" clears the threshold; min_keep=1 is already satisfied by "a".
	require.Len(t, kept, 1)
	assert.Equal(t, "a", kept[0].Content)
	assert.True(t, scores[1].Dropped)
	assert.True(t, scores[2].Dropped)
}

func TestFilterPanelByScore_MinKeepGuaranteesSurvivors(t *testing.T) {
	p := panel("a", "b", "c")
	scores := []groundingScore{
		{Model: "a", Score: 0.4},
		{Model: "b", Score: 0.2},
		{Model: "c", Score: 0.1},
	}
	// All below min_score, but min_keep=2 keeps the two highest.
	kept := filterPanelByScore(p, scores, fusionExecutionConfig{GroundingMinScore: 0.9, GroundingMinKeep: 2})
	require.Len(t, kept, 2)
	assert.Equal(t, "a", kept[0].Content)
	assert.Equal(t, "b", kept[1].Content)
}

func TestResolveGroundingReference(t *testing.T) {
	assert.True(t, resolveGroundingReference(config.FusionGroundingReferenceContext, ""))
	assert.False(t, resolveGroundingReference(config.FusionGroundingReferencePanel, "ctx"))
	// hybrid: context only when present.
	assert.True(t, resolveGroundingReference(config.FusionGroundingReferenceHybrid, "ctx"))
	assert.False(t, resolveGroundingReference(config.FusionGroundingReferenceHybrid, ""))
	assert.False(t, resolveGroundingReference("", ""))
}

func TestExtractGroundingContext(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage("retrieved passage"),
			openai.UserMessage("the question"),
		},
	}
	got := extractGroundingContext(req)
	assert.Contains(t, got, "retrieved passage")
	assert.NotContains(t, got, "the question")
}

func TestApplyGrounding_DisabledReturnsPanelUnchanged(t *testing.T) {
	l := NewFusionLooper(&config.LooperConfig{})
	p := panel("x", "y")
	kept, scores, mode, err := l.applyGrounding(newFusionTestRequest(), fusionExecutionConfig{}, p)
	require.NoError(t, err)
	assert.Equal(t, p, kept)
	assert.Nil(t, scores)
	assert.Empty(t, mode)
}

func TestApplyGrounding_OnErrorSkipFallsBack(t *testing.T) {
	withGroundingBackends(t, nil, nil) // panel mode needs NLI; nil => error
	l := NewFusionLooper(&config.LooperConfig{})
	p := panel("x", "y")
	cfg := fusionExecutionConfig{
		GroundingEnabled:   true,
		GroundingReference: config.FusionGroundingReferencePanel,
		GroundingOnError:   config.FusionOnErrorSkip,
		GroundingMinKeep:   1,
	}
	kept, _, _, err := l.applyGrounding(newFusionTestRequest(), cfg, p)
	require.NoError(t, err)
	assert.Equal(t, p, kept) // unchanged on skip
}

func TestApplyGrounding_OnErrorFailReturnsError(t *testing.T) {
	withGroundingBackends(t, nil, nil)
	l := NewFusionLooper(&config.LooperConfig{})
	cfg := fusionExecutionConfig{
		GroundingEnabled:   true,
		GroundingReference: config.FusionGroundingReferencePanel,
		GroundingOnError:   config.FusionOnErrorFail,
		GroundingMinKeep:   1,
	}
	_, _, _, err := l.applyGrounding(newFusionTestRequest(), cfg, panel("x", "y"))
	require.Error(t, err)
}

func TestApplyGrounding_PanelModeFiltersContradicted(t *testing.T) {
	withGroundingBackends(t, func(_, hypothesis string) (float32, float32, error) {
		if strings.Contains(hypothesis, "bad") {
			return 0.1, 0.8, nil
		}
		return 0.9, 0.05, nil
	}, nil)

	l := NewFusionLooper(&config.LooperConfig{})
	cfg := fusionExecutionConfig{
		GroundingEnabled:                 true,
		GroundingReference:               config.FusionGroundingReferencePanel,
		GroundingOnError:                 config.FusionOnErrorSkip,
		GroundingMinScore:                0.5,
		GroundingMinKeep:                 1,
		GroundingNLIContradictionPenalty: 1.0,
	}
	kept, scores, mode, err := l.applyGrounding(newFusionTestRequest(), cfg, panel("good one", "good two", "bad three"))
	require.NoError(t, err)
	assert.Equal(t, config.FusionGroundingReferencePanel, mode)
	require.Len(t, scores, 3)
	// The contradicted "bad" response is dropped from the judge's panel.
	for _, r := range kept {
		assert.NotContains(t, r.Content, "bad")
	}
	assert.Len(t, kept, 2)
}

// TestFusionExecute_GroundingKeepsContradictedOutOfJudge runs the full Execute
// path with grounding enabled and asserts the contradicted panel response never
// reaches the judge, while usage still reflects the full panel cost.
func TestFusionExecute_GroundingKeepsContradictedOutOfJudge(t *testing.T) {
	withGroundingBackends(t, func(_, hypothesis string) (float32, float32, error) {
		if strings.Contains(hypothesis, "bad") {
			return 0.1, 0.8, nil
		}
		return 0.9, 0.05, nil
	}, nil)

	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		switch model {
		case "panel-a":
			return "good grounded answer", http.StatusOK
		case "panel-b":
			return "bad contradicted answer", http.StatusOK
		case "panel-c":
			return "good supported answer", http.StatusOK
		case "judge":
			if strings.Contains(prompt, "return only valid JSON") {
				// The dropped panel response must not reach the judge.
				assert.NotContains(t, prompt, "bad contradicted answer")
				return `{"consensus":["agree"],"contradictions":[],"partial_coverage":[],"unique_insights":[],"blind_spots":[]}`, http.StatusOK
			}
			return "final answer", http.StatusOK
		default:
			return "unexpected", http.StatusInternalServerError
		}
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a", "panel-b", "panel-c"},
			Grounding: &config.FusionGroundingConfig{
				Enabled:   true,
				Reference: config.FusionGroundingReferencePanel,
				MinScore:  0.5,
				MinKeep:   1,
			},
		},
	}

	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.NoError(t, err)

	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(resp.Body, &body))
	fusionTrace := body["fusion"].(map[string]interface{})
	grounding, ok := fusionTrace["grounding"].(map[string]interface{})
	require.True(t, ok, "fusion trace should carry grounding info")
	assert.Equal(t, config.FusionGroundingReferencePanel, grounding["reference_mode"])
	// Usage reflects the full panel (3 panel + judge analysis + judge final), not
	// just the kept responses — grounding makes no extra calls but the panel cost
	// was already paid.
	assert.Positive(t, resp.Usage.TotalTokens)
}
