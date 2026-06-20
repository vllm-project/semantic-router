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

// TestScoreByPanel_LongAnswerSentenceChunking verifies that on long answers (which
// would otherwise truncate the hypothesis away in a single 512-token NLI call) the
// scorer chunks the hypothesis into sentences and still discriminates: a long answer
// whose peers contradict one of its sentences scores below a fully-corroborated one.
func TestScoreByPanel_LongAnswerSentenceChunking(t *testing.T) {
	// Stub NLI keyed on the hypothesis sentence: any sentence containing the
	// "FALSE" marker is contradicted by its peers; everything else is entailed.
	var calls int
	withGroundingBackends(t, func(_, hypothesis string) (float32, float32, error) {
		calls++
		if strings.Contains(hypothesis, "FALSE") {
			return 0.05, 0.9, nil
		}
		return 0.9, 0.05, nil
	}, nil)

	// Both answers exceed nliSingleCallBudget runes so the chunking path engages.
	long := func(dirty bool) string {
		var b strings.Builder
		for i := 0; i < 24; i++ {
			if dirty {
				b.WriteString("This FALSE clinical claim is rejected by the other panel models, item ")
			} else {
				b.WriteString("This is a benign and well supported clinical statement, item number ")
			}
			b.WriteString(string(rune('A' + (i % 26))))
			b.WriteString(". ")
		}
		return b.String()
	}

	clean := long(false)
	dirty := long(true)

	p := panel(clean, dirty)
	scores, err := scoreByPanel(p, fusionExecutionConfig{GroundingNLIContradictionPenalty: 1.0})
	require.NoError(t, err)
	require.Len(t, scores, 2)

	// The chunking path must actually run (many NLI calls, not one per pair).
	assert.Greater(t, calls, 4, "expected sentence-level chunking to issue many NLI calls")
	// The answer with a contradicted sentence scores lower and is flagged.
	assert.Greater(t, scores[0].Score, scores[1].Score)
	assert.NotEmpty(t, scores[1].FlaggedSpans)
}

func TestSplitSentencesCapped(t *testing.T) {
	got := splitSentencesCapped("First real sentence here. Short. Second real sentence here!\nThird real one?", 1000, 10)
	// "Short." is < 12 runes and dropped; the three real sentences survive.
	require.Len(t, got, 3)
	assert.Contains(t, got[0], "First real sentence")
	assert.Contains(t, got[2], "Third real one")
}

func TestChunkTextCapped(t *testing.T) {
	windows := chunkTextCapped("aaaaabbbbbccccc", 5, 10)
	require.Len(t, windows, 3)
	assert.Equal(t, "aaaaa", windows[0])
	assert.Equal(t, "ccccc", windows[2])
	// maxWindows caps the count even when more content remains.
	assert.Len(t, chunkTextCapped("aaaaabbbbbccccc", 5, 2), 2)
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
		GroundingPolicy:                  config.FusionGroundingPolicyFilter,
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

// TestApplyGrounding_WeightPolicyKeepsAll verifies the default soft-weight policy
// scores the panel but drops nothing, even a peer-contradicted response.
func TestApplyGrounding_WeightPolicyKeepsAll(t *testing.T) {
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
		GroundingPolicy:                  config.FusionGroundingPolicyWeight,
		GroundingOnError:                 config.FusionOnErrorSkip,
		GroundingMinScore:                0.5, // high threshold is ignored under weight
		GroundingMinKeep:                 1,
		GroundingNLIContradictionPenalty: 1.0,
	}
	in := panel("good one", "good two", "bad three")
	kept, scores, _, err := l.applyGrounding(newFusionTestRequest(), cfg, in)
	require.NoError(t, err)
	// Nothing dropped: the contradicted response is still present.
	assert.Len(t, kept, 3)
	require.Len(t, scores, 3)
	for _, s := range scores {
		assert.False(t, s.Dropped, "weight policy must not drop responses")
	}
}

// TestApplyGrounding_AnnotatePolicyKeepsAll mirrors the weight test for annotate.
func TestApplyGrounding_AnnotatePolicyKeepsAll(t *testing.T) {
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
		GroundingPolicy:                  config.FusionGroundingPolicyAnnotate,
		GroundingOnError:                 config.FusionOnErrorSkip,
		GroundingMinScore:                0.9,
		GroundingMinKeep:                 1,
		GroundingNLIContradictionPenalty: 1.0,
	}
	kept, scores, _, err := l.applyGrounding(newFusionTestRequest(), cfg, panel("good one", "bad two"))
	require.NoError(t, err)
	assert.Len(t, kept, 2)
	require.Len(t, scores, 2)
	for _, s := range scores {
		assert.False(t, s.Dropped)
	}
}

func TestGroundingSynthesisNotes(t *testing.T) {
	scores := []groundingScore{
		{Model: "a", Score: 0.8},
		{Model: "b", Score: 0.2, FlaggedSpans: []string{"a"}},
	}
	// weight: includes the weighting directive plus the per-model notes.
	weight := groundingSynthesisNotes(scores, config.FusionGroundingPolicyWeight)
	assert.Contains(t, weight, "Weight each panel answer")
	assert.Contains(t, weight, "consistency is not the same as correctness")
	assert.Contains(t, weight, "score 0.80")
	// annotate: notes without the weighting directive.
	annotate := groundingSynthesisNotes(scores, config.FusionGroundingPolicyAnnotate)
	assert.NotContains(t, annotate, "Weight each panel answer")
	assert.Contains(t, annotate, "Groundedness notes")
	// filter: nothing (the panel was already pruned).
	assert.Empty(t, groundingSynthesisNotes(scores, config.FusionGroundingPolicyFilter))
	// no scores: empty regardless of policy.
	assert.Empty(t, groundingSynthesisNotes(nil, config.FusionGroundingPolicyWeight))
}

// TestApplyGroundingDefaults_PolicyDefaultsToWeight verifies the resolved config
// defaults the policy to weight when grounding is enabled and policy is unset.
func TestApplyGroundingDefaults_PolicyDefaultsToWeight(t *testing.T) {
	cfg := fusionExecutionConfig{GroundingEnabled: true}
	applyGroundingDefaults(&cfg)
	assert.Equal(t, config.FusionGroundingPolicyWeight, cfg.GroundingPolicy)

	// An explicit policy is preserved.
	cfg = fusionExecutionConfig{GroundingEnabled: true, GroundingPolicy: config.FusionGroundingPolicyFilter}
	applyGroundingDefaults(&cfg)
	assert.Equal(t, config.FusionGroundingPolicyFilter, cfg.GroundingPolicy)

	// Disabled grounding leaves the policy untouched.
	cfg = fusionExecutionConfig{}
	applyGroundingDefaults(&cfg)
	assert.Empty(t, cfg.GroundingPolicy)
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
				Policy:    config.FusionGroundingPolicyFilter,
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

// TestFusionExecute_WeightPolicyKeepsPanelAndAnnotatesSynthesis runs the full
// Execute path under the default weight policy: the contradicted response is NOT
// dropped (it still reaches the judge) and the final synthesis prompt carries the
// groundedness weighting notes.
func TestFusionExecute_WeightPolicyKeepsPanelAndAnnotatesSynthesis(t *testing.T) {
	withGroundingBackends(t, func(_, hypothesis string) (float32, float32, error) {
		if strings.Contains(hypothesis, "bad") {
			return 0.1, 0.8, nil
		}
		return 0.9, 0.05, nil
	}, nil)

	var sawDissenterAtJudge, sawNotesAtSynthesis bool
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		switch model {
		case "panel-a":
			return "good grounded answer", http.StatusOK
		case "panel-b":
			return "bad contradicted answer", http.StatusOK
		case "judge":
			if strings.Contains(prompt, "return only valid JSON") {
				// Weight policy keeps the dissenter in the judge's panel.
				if strings.Contains(prompt, "bad contradicted answer") {
					sawDissenterAtJudge = true
				}
				return `{"consensus":[],"contradictions":["x"],"partial_coverage":[],"unique_insights":[],"blind_spots":[]}`, http.StatusOK
			}
			// Final synthesis prompt carries the weighting notes.
			if strings.Contains(prompt, "Weight each panel answer") {
				sawNotesAtSynthesis = true
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
			AnalysisModels: []string{"panel-a", "panel-b"},
			Grounding: &config.FusionGroundingConfig{
				Enabled:   true,
				Reference: config.FusionGroundingReferencePanel,
				// Policy unset -> defaults to weight.
			},
		},
	}

	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.NoError(t, err)
	assert.True(t, sawDissenterAtJudge, "weight policy should keep the contradicted response in the judge panel")
	assert.True(t, sawNotesAtSynthesis, "weight policy should annotate the final synthesis prompt")

	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(resp.Body, &body))
	grounding := body["fusion"].(map[string]interface{})["grounding"].(map[string]interface{})
	assert.Equal(t, config.FusionGroundingPolicyWeight, grounding["policy"])
}
