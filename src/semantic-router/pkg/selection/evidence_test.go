package selection

import (
	"testing"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const testIntelligenceIndex = "vllm-sr/test-intelligence@1.0.0"

func modelParamsWithTestQuality(score float64) config.ModelParams {
	return addTestQuality(config.ModelParams{}, score)
}

func addTestQuality(params config.ModelParams, score float64) config.ModelParams {
	return addTestEvidence(params, score, 1, "")
}

func addTestEvidence(params config.ModelParams, score, coverage float64, effort string) config.ModelParams {
	scaled := score * 100
	result := modelcatalog.IndexResult{
		Model: "test-model", ReasoningEffort: effort, Index: testIntelligenceIndex,
		Status: "available", Score: &scaled, Coverage: coverage,
	}
	params.QualityIndex = testIntelligenceIndex
	if effort == "" {
		params.IndexResults = map[string]modelcatalog.IndexResult{testIntelligenceIndex: result}
	} else {
		if params.IndexResultsByEffort == nil {
			params.IndexResultsByEffort = map[string]map[string]modelcatalog.IndexResult{}
		}
		params.IndexResultsByEffort[effort] = map[string]modelcatalog.IndexResult{testIntelligenceIndex: result}
	}
	return params
}

func TestCandidateScoreKeyDistinguishesEfforts(t *testing.T) {
	candidates := []config.ModelRef{
		{Model: "model", ModelReasoningControl: config.ModelReasoningControl{ReasoningEffort: "low"}},
		{Model: "model", ModelReasoningControl: config.ModelReasoningControl{ReasoningEffort: "high"}},
	}
	if low, high := candidateScoreKey(candidates, 0), candidateScoreKey(candidates, 1); low == high {
		t.Fatalf("duplicate candidate score keys = %q", low)
	}
}

func TestCandidateRankCompare(t *testing.T) {
	score := 75.0
	evidence := func(coverage float64) *modelcatalog.IndexResult {
		return &modelcatalog.IndexResult{Index: testIntelligenceIndex, Status: "available", Score: &score, Coverage: coverage}
	}
	if got := (candidateRank{score: 0.5, evidence: evidence(1)}).Compare(candidateRank{score: 0.6, evidence: evidence(0.6)}); got >= 0 {
		t.Fatalf("algorithm score must rank first, got %d", got)
	}
	if got := (candidateRank{score: 0.5, evidence: evidence(1)}).Compare(candidateRank{score: 0.5, evidence: evidence(0.6)}); got <= 0 {
		t.Fatalf("higher coverage must break equal evidence ties, got %d", got)
	}
	otherScore := 74.0
	if got := (candidateRank{score: 0.5, evidence: evidence(1)}).Compare(candidateRank{score: 0.5, evidence: &modelcatalog.IndexResult{Index: testIntelligenceIndex, Status: "available", Score: &otherScore, Coverage: 0.6}}); got != 0 {
		t.Fatalf("different evidence scores must not reorder equal algorithm scores, got %d", got)
	}
	if got := (candidateRank{score: 0.5, evidence: evidence(1)}).Compare(candidateRank{score: 0.5}); got != 0 {
		t.Fatalf("missing evidence must be incomparable, got %d", got)
	}
}

func TestEvidenceForCandidateUsesExactEffort(t *testing.T) {
	params := addTestEvidence(config.ModelParams{}, 0.7, 0.6, "low")
	params = addTestEvidence(params, 0.8, 1, "high")
	models := map[string]config.ModelParams{"model": params}

	got := evidenceForCandidate(models, config.ModelRef{Model: "model", ModelReasoningControl: config.ModelReasoningControl{ReasoningEffort: "high"}})
	if got == nil || got.Score == nil || *got.Score != 80 || got.Coverage != 1 {
		t.Fatalf("high-effort evidence = %+v", got)
	}
	if got := evidenceForCandidate(models, config.ModelRef{Model: "model", ModelReasoningControl: config.ModelReasoningControl{ReasoningEffort: "medium"}}); got != nil {
		t.Fatalf("missing exact effort fell back: %+v", got)
	}
}
