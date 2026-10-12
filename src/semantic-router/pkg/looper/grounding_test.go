package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Test construction keeps the fake detector local to each constructed looper.
var groundingDetect HallucinationDetectFunc

func withGroundingDetector(t *testing.T, detect HallucinationDetectFunc) {
	t.Helper()
	prev := groundingDetect
	groundingDetect = detect
	t.Cleanup(func() { groundingDetect = prev })
}

// unsupportedWhen is a detector that flags any answer containing marker as
// unsupported by its context with the given hallucinated-token score.
func unsupportedWhen(marker string, score float32) HallucinationDetectFunc {
	return func(_ context.Context, _, _, answer string) (GroundingEvidence, error) {
		if strings.Contains(answer, marker) {
			return spanEvidence([]string{"unsupported claim"}, score), nil
		}
		return GroundingEvidence{}, nil
	}
}

func panel(contents ...string) []*ModelResponse {
	resps := make([]*ModelResponse, 0, len(contents))
	for i, c := range contents {
		resps = append(resps, &ModelResponse{Model: string(rune('a' + i)), Content: c})
	}
	return resps
}

func TestScoreByPanel_RanksContradictedLower(t *testing.T) {
	// Read against any peer, an answer containing "bad" is unsupported.
	withGroundingDetector(t, unsupportedWhen("bad", 0.8))

	p := panel("good one", "good two", "bad three")
	scores, err := scoreByPanel(context.Background(), "the question", p, fusionExecutionConfig{GroundingContradictionPenalty: 1.0}, groundingDetect)
	require.NoError(t, err)
	require.Len(t, scores, 3)
	// Fully supported: (1 + 1) / 2; contradicted at 0.8: (0.2 - 0.8 + 1) / 2.
	assert.InDelta(t, 1.0, scores[0].Score, 1e-6)
	assert.InDelta(t, 0.2, scores[2].Score, 1e-6)

	assert.Greater(t, scores[0].Score, scores[2].Score)
	assert.Greater(t, scores[1].Score, scores[2].Score)
	// The contradicted response is flagged by its peers.
	assert.NotEmpty(t, scores[2].FlaggedSpans)
	assert.Empty(t, scores[0].FlaggedSpans)
	// Regression (#2857): flags carry caller-visible peer model labels, never
	// internal panel candidate IDs, so traces and the synthesis notes stay
	// associated with real responses.
	assert.ElementsMatch(t, []string{"a", "b"}, scores[2].FlaggedSpans)
	assert.NotContains(t, strings.Join(scores[2].FlaggedSpans, ","), "panel-")
	// And the synthesis notes render the model labels, not synthetic IDs.
	notes := formatGroundingNotes(scores)
	assert.Contains(t, notes, "c: score")
	assert.NotContains(t, notes, "panel-")
}

// TestScoreByPanel_ReadsEachResponseAgainstItsPeers pins the panel reference:
// every peer answer is the detector's context for the response, with the
// request's question, and a response is never read against itself.
func TestScoreByPanel_ReadsEachResponseAgainstItsPeers(t *testing.T) {
	type read struct{ context, question, answer string }
	var (
		mu    sync.Mutex
		reads []read
	)
	withGroundingDetector(t, func(_ context.Context, contextText, question, answer string) (GroundingEvidence, error) {
		mu.Lock()
		defer mu.Unlock()
		reads = append(reads, read{contextText, question, answer})
		return GroundingEvidence{}, nil
	})

	_, err := scoreByPanel(context.Background(), "q", panel("one", "two", "three"), fusionExecutionConfig{}, groundingDetect)
	require.NoError(t, err)
	assert.ElementsMatch(t, []read{
		{"two", "q", "one"},
		{"three", "q", "one"},
		{"one", "q", "two"},
		{"three", "q", "two"},
		{"one", "q", "three"},
		{"two", "q", "three"},
	}, reads)
}

func TestScoreByPanel_UnscoredSpanCountsAsContradiction(t *testing.T) {
	// A detector without token scores (score 0) still flags the response.
	withGroundingDetector(t, unsupportedWhen("bad", 0))

	scores, err := scoreByPanel(context.Background(), "q", panel("good", "bad"), fusionExecutionConfig{GroundingContradictionPenalty: 1.0}, groundingDetect)
	require.NoError(t, err)
	assert.InDelta(t, 0.0, scores[1].Score, 1e-6)
	assert.Equal(t, []string{"a"}, scores[1].FlaggedSpans)
}

func TestScoreByContext_FewerUnsupportedSpansScoresHigher(t *testing.T) {
	// "bad" answers have an unsupported span; grounded ones have none.
	withGroundingDetector(t, unsupportedWhen("bad", 0.9))

	p := panel("grounded answer", "bad answer")
	scores, err := scoreByContext(context.Background(), "the context", "the question", p, groundingDetect)
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
	l := newGroundedTestFusionLooper(&config.LooperConfig{})
	p := panel("x", "y")
	kept, scores, mode, err := l.applyGrounding(context.Background(), newFusionTestRequest(), fusionExecutionConfig{}, p)
	require.NoError(t, err)
	assert.Equal(t, p, kept)
	assert.Nil(t, scores)
	assert.Empty(t, mode)
}

func TestApplyGrounding_OnErrorSkipFallsBack(t *testing.T) {
	withGroundingDetector(t, nil) // no detector => grounding error
	l := newGroundedTestFusionLooper(&config.LooperConfig{})
	p := panel("x", "y")
	cfg := fusionExecutionConfig{
		GroundingEnabled:   true,
		GroundingReference: config.FusionGroundingReferencePanel,
		GroundingOnError:   config.FusionOnErrorSkip,
		GroundingMinKeep:   1,
	}
	kept, _, _, err := l.applyGrounding(context.Background(), newFusionTestRequest(), cfg, p)
	require.NoError(t, err)
	assert.Equal(t, p, kept) // unchanged on skip
}

func TestApplyGrounding_OnErrorFailReturnsError(t *testing.T) {
	withGroundingDetector(t, nil)
	l := newGroundedTestFusionLooper(&config.LooperConfig{})
	cfg := fusionExecutionConfig{
		GroundingEnabled:   true,
		GroundingReference: config.FusionGroundingReferencePanel,
		GroundingOnError:   config.FusionOnErrorFail,
		GroundingMinKeep:   1,
	}
	_, _, _, err := l.applyGrounding(context.Background(), newFusionTestRequest(), cfg, panel("x", "y"))
	require.Error(t, err)
}

func TestApplyGrounding_PanelModeFiltersContradicted(t *testing.T) {
	withGroundingDetector(t, unsupportedWhen("bad", 0.8))

	l := newGroundedTestFusionLooper(&config.LooperConfig{})
	cfg := fusionExecutionConfig{
		GroundingEnabled:              true,
		GroundingReference:            config.FusionGroundingReferencePanel,
		GroundingPolicy:               config.FusionGroundingPolicyFilter,
		GroundingOnError:              config.FusionOnErrorSkip,
		GroundingMinScore:             0.5,
		GroundingMinKeep:              1,
		GroundingContradictionPenalty: 1.0,
	}
	kept, scores, mode, err := l.applyGrounding(context.Background(), newFusionTestRequest(), cfg, panel("good one", "good two", "bad three"))
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
	withGroundingDetector(t, unsupportedWhen("bad", 0.8))

	l := newGroundedTestFusionLooper(&config.LooperConfig{})
	cfg := fusionExecutionConfig{
		GroundingEnabled:              true,
		GroundingReference:            config.FusionGroundingReferencePanel,
		GroundingPolicy:               config.FusionGroundingPolicyWeight,
		GroundingOnError:              config.FusionOnErrorSkip,
		GroundingMinScore:             0.5, // high threshold is ignored under weight
		GroundingMinKeep:              1,
		GroundingContradictionPenalty: 1.0,
	}
	in := panel("good one", "good two", "bad three")
	kept, scores, _, err := l.applyGrounding(context.Background(), newFusionTestRequest(), cfg, in)
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
	withGroundingDetector(t, unsupportedWhen("bad", 0.8))

	l := newGroundedTestFusionLooper(&config.LooperConfig{})
	cfg := fusionExecutionConfig{
		GroundingEnabled:              true,
		GroundingReference:            config.FusionGroundingReferencePanel,
		GroundingPolicy:               config.FusionGroundingPolicyAnnotate,
		GroundingOnError:              config.FusionOnErrorSkip,
		GroundingMinScore:             0.9,
		GroundingMinKeep:              1,
		GroundingContradictionPenalty: 1.0,
	}
	kept, scores, _, err := l.applyGrounding(context.Background(), newFusionTestRequest(), cfg, panel("good one", "bad two"))
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
	withGroundingDetector(t, unsupportedWhen("bad", 0.8))

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

	resp, err := newGroundedTestFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
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
	withGroundingDetector(t, unsupportedWhen("bad", 0.8))

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

	resp, err := newGroundedTestFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.NoError(t, err)
	assert.True(t, sawDissenterAtJudge, "weight policy should keep the contradicted response in the judge panel")
	assert.True(t, sawNotesAtSynthesis, "weight policy should annotate the final synthesis prompt")

	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(resp.Body, &body))
	grounding := body["fusion"].(map[string]interface{})["grounding"].(map[string]interface{})
	assert.Equal(t, config.FusionGroundingPolicyWeight, grounding["policy"])
}

func newGroundedTestFusionLooper(cfg *config.LooperConfig) *FusionLooper {
	looper := NewFusionLooper(cfg)
	looper.grounding = &GroundingBackends{Detect: groundingDetect}
	return looper
}

func spanEvidence(spans []string, score float32) GroundingEvidence {
	evidence := GroundingEvidence{Unsupported: len(spans) > 0, Spans: spans}
	if score > 0 {
		evidence.Probability = &score
	}
	return evidence
}
