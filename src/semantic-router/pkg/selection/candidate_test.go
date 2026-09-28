package selection

import (
	"errors"
	"math"
	"testing"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestCandidateIdentityUsesControlValues(t *testing.T) {
	on, anotherOn, off := true, true, false
	ref := config.ModelRef{Model: "model", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: &on, ReasoningEffort: "high"}}
	copy := ref
	copy.UseReasoning = &anotherOn
	if CandidateIdentity(ref) != CandidateIdentity(copy) {
		t.Fatal("equal control values acquired different identities")
	}
	for _, mutate := range []func(*config.ModelRef){
		func(r *config.ModelRef) { r.UseReasoning = nil },
		func(r *config.ModelRef) { r.UseReasoning = &off },
		func(r *config.ModelRef) { r.ReasoningEffort = "low" },
		func(r *config.ModelRef) { r.ReasoningMode = "adaptive" },
		func(r *config.ModelRef) { r.LoRAName = "adapter" },
		func(r *config.ModelRef) { r.Weight = 2 },
	} {
		copy = ref
		mutate(&copy)
		if CandidateIdentity(ref) == CandidateIdentity(copy) {
			t.Fatalf("different candidate collapsed: %+v", copy)
		}
	}
}

func TestCandidateScoresSurviveReorderingAndDiagnosticMutation(t *testing.T) {
	low := config.ModelRef{Model: "model", ModelReasoningControl: config.ModelReasoningControl{ReasoningEffort: "low"}}
	high := low
	high.ReasoningEffort = "high"
	scores := CandidateScores{{Candidate: low, Score: .6}, {Candidate: high, Score: .9}}
	result := (&SelectionResult{Score: .9}).WithCandidate(high).WithScores(scores)
	result.AllScores = map[string]float64{"model": 100}
	rows := result.ScoresFor([]config.ModelRef{high, low})
	for _, row := range scores {
		if value, ok := rows.Get(row.Candidate); !ok || value != row.Score {
			t.Fatalf("candidate %+v lost its score: %v %v", row.Candidate, value, ok)
		}
	}
	missing := high
	missing.ReasoningEffort = "medium"
	if _, ok := rows.Get(missing); ok {
		t.Fatal("missing candidate borrowed another effort's score")
	}
	if len(rows.Diagnostics()) != 2 {
		t.Fatal("diagnostics collapsed candidate variants")
	}
}

func TestCandidateScoresBestPreservesObjectiveAndEvidenceBoundaries(t *testing.T) {
	quality := 75.0
	evidence := func(coverage float64) *modelcatalog.IndexResult {
		return &modelcatalog.IndexResult{Index: "test/intelligence@1.0.0", Status: "available", Score: &quality, Coverage: coverage}
	}
	for _, test := range []struct {
		name      string
		scores    CandidateScores
		direction ScoreDirection
		coverage  bool
		want      int
	}{
		{"higher", CandidateScores{{Score: .6}, {Score: .9}}, HigherIsBetter, false, 1},
		{"lower", CandidateScores{{Score: .6}, {Score: .9}}, LowerIsBetter, false, 0},
		{"coverage_tie", CandidateScores{{Score: .9, Evidence: evidence(.6)}, {Score: .9, Evidence: evidence(1)}}, HigherIsBetter, true, 1},
		{"objective_first", CandidateScores{{Score: .9, Evidence: evidence(.6)}, {Score: .8, Evidence: evidence(1)}}, HigherIsBetter, true, 0},
		{"quality_disabled", CandidateScores{{Score: .9, Evidence: evidence(.6)}, {Score: .9, Evidence: evidence(1)}}, HigherIsBetter, false, 0},
		{"missing_not_zero", CandidateScores{{Score: 0}, {Score: 0, Evidence: evidence(1)}}, HigherIsBetter, true, 0},
		{"nonfinite", CandidateScores{{Score: math.NaN()}, {Score: math.Inf(1)}}, HigherIsBetter, true, -1},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := test.scores.Best(test.direction, test.coverage); got != test.want {
				t.Fatalf("winner=%d, want %d", got, test.want)
			}
		})
	}
}

func TestBlendCandidateScoresUsesIdentityAndDirection(t *testing.T) {
	low := config.ModelRef{Model: "model", ModelReasoningControl: config.ModelReasoningControl{ReasoningEffort: "low"}}
	high := low
	high.ReasoningEffort = "high"
	quality := CandidateScores{{Candidate: high, Score: .9}, {Candidate: low, Score: .6}}
	latency := CandidateScores{{Candidate: low, Score: 10}, {Candidate: high, Score: 20}}
	scores := BlendCandidateScores([]config.ModelRef{low, high}, []ScoreComponent{
		{Weight: .75, Scores: NormalizeCandidateScores(quality, HigherIsBetter)},
		{Weight: .25, Scores: NormalizeCandidateScores(latency, LowerIsBetter)},
		{Weight: 100}, // unavailable component must not dilute the available weights
	})
	if scores[0].Score != .25 || scores[1].Score != .75 || scores.Best(HigherIsBetter, false) != 1 {
		t.Fatalf("incorrect candidate composition: %+v", scores)
	}
	if quality[0].Score != .9 || latency[0].Score != 10 {
		t.Fatal("normalization mutated component inputs")
	}
}

func TestResolveSelectionCandidateRejectsAmbiguousModelOnlyResult(t *testing.T) {
	refs := []config.ModelRef{
		{Model: "model", ModelReasoningControl: config.ModelReasoningControl{ReasoningEffort: "low"}},
		{Model: "model", ModelReasoningControl: config.ModelReasoningControl{ReasoningEffort: "high"}},
	}
	ctx := &SelectionContext{CandidateModels: refs}
	if _, err := ResolveSelectionCandidate(ctx, &SelectionResult{SelectedModel: "model"}); !errors.Is(err, ErrNoEligibleCandidates) {
		t.Fatalf("ambiguous selection must fail closed, got %v", err)
	}
	result := (&SelectionResult{}).WithCandidate(refs[1])
	candidate, err := ResolveSelectionCandidate(ctx, result)
	if err != nil || candidate.ReasoningEffort != "high" {
		t.Fatalf("exact candidate = %+v, %v", candidate, err)
	}
}
