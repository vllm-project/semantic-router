package selection

import (
	"context"
	"math"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selectiontrace"
)

// estimateRecorder answers every candidate from answers, by model and
// effort, and records what it was asked.
type estimateRecorder struct {
	answers  map[string]CandidateAnswer
	requests []string
	asked    [][]config.ModelRef
}

func (r *estimateRecorder) estimates() *MultiFactorEstimates {
	return &MultiFactorEstimates{
		Mode:       config.MultiFactorEstimatesModeObserve,
		Deployment: "primary",
		Estimate: func(_ context.Context, request string, candidates []config.ModelRef) []CandidateAnswer {
			r.requests = append(r.requests, request)
			r.asked = append(r.asked, candidates)
			answers := make([]CandidateAnswer, len(candidates))
			for i, candidate := range candidates {
				answers[i] = r.answers[candidate.Model+"/"+candidate.ReasoningEffort]
			}
			return answers
		},
	}
}

func lengths(under256, to1k, to4k, over4k float64) map[string]float64 {
	return map[string]float64{"under_256": under256, "256_to_1k": to1k, "1k_to_4k": to4k, "over_4k": over4k}
}

func effortRef(model, effort string) config.ModelRef {
	enabled := effort != ""
	return config.ModelRef{Model: model, ModelReasoningControl: config.ModelReasoningControl{UseReasoning: &enabled, ReasoningEffort: effort}}
}

// Observe mode records an estimate per exact candidate (model and effort)
// and changes nothing else: the selected model, scores and objective are
// those of the same selector without estimates.
func TestMultiFactorObserveEstimatesLeaveTheSelectionUnchanged(t *testing.T) {
	for _, strategy := range []string{config.MultiFactorObjectiveWeighted, config.MultiFactorObjectiveLexicographic} {
		t.Run(strategy, func(t *testing.T) {
			build := func(estimates *MultiFactorEstimates) *MultiFactorSelector {
				cfg := DefaultMultiFactorConfig()
				cfg.Objective = MultiFactorObjective{Strategy: strategy}
				cfg.Estimates = estimates
				return buildMFSelector(cfg, map[string]config.ModelParams{
					"qwen": {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 1, CompletionPer1M: 2}},
					"tiny": {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: .1, CompletionPer1M: .2}},
				}, func(string) int { return 0 }, nil, func(string, int) (float64, bool) { return 0, false })
			}
			recorder := &estimateRecorder{answers: map[string]CandidateAnswer{
				"qwen/":     {Status: selectiontrace.EstimateAvailable, SuccessProbability: .4, OutputLength: lengths(1, 0, 0, 0)},
				"qwen/high": {Status: selectiontrace.EstimateAvailable, SuccessProbability: .86, OutputLength: lengths(0, .5, .5, 0)},
				"tiny/":     {Status: selectiontrace.EstimateDeadlineExceeded, Reason: "timeout"},
			}}
			refs := []config.ModelRef{effortRef("qwen", ""), effortRef("qwen", "high"), effortRef("tiny", "")}
			selCtx := func() *SelectionContext {
				return &SelectionContext{Query: "If 3x + 7 = 22, what is x?", CandidateModels: refs, InputTokens: 1000}
			}
			base, err := build(nil).Select(context.Background(), selCtx())
			if err != nil {
				t.Fatal(err)
			}
			observed, err := build(recorder.estimates()).Select(context.Background(), selCtx())
			if err != nil {
				t.Fatal(err)
			}
			if observed.SelectedModel != base.SelectedModel || !reflect.DeepEqual(observed.SelectedCandidate, base.SelectedCandidate) ||
				!reflect.DeepEqual(observed.CandidateScores, base.CandidateScores) || !reflect.DeepEqual(observed.EligibleModels, base.EligibleModels) {
				t.Fatalf("observe mode changed the selection:\nbase %+v\nobserved %+v", base, observed)
			}
			trace := observed.MultiFactor
			if base.MultiFactor != nil {
				if !reflect.DeepEqual(trace.Stages, base.MultiFactor.Stages) || !reflect.DeepEqual(trace.FinalSurvivors, base.MultiFactor.FinalSurvivors) {
					t.Fatal("observe mode changed the objective trace")
				}
			} else if trace.Stages != nil || trace.FinalSurvivors != nil {
				t.Fatalf("a weighted objective recorded stages: %+v", trace)
			}
			if len(recorder.asked) != 1 || recorder.requests[0] != "If 3x + 7 = 22, what is x?" || !reflect.DeepEqual(recorder.asked[0], refs) {
				t.Fatalf("asked %v about %v", recorder.requests, recorder.asked)
			}
			estimates := trace.Estimates
			if estimates == nil || estimates.Mode != config.MultiFactorEstimatesModeObserve || len(estimates.Candidates) != 3 {
				t.Fatalf("estimates = %+v", estimates)
			}
			direct, high, late := estimates.Candidates[0], estimates.Candidates[1], estimates.Candidates[2]
			if !reflect.DeepEqual(direct.Candidate, refs[0]) || !reflect.DeepEqual(high.Candidate, refs[1]) {
				t.Fatal("estimates are not keyed by the exact candidate")
			}
			// 1000 input tokens and 128 expected output tokens at $1 and $2 per 1M.
			if direct.Status != selectiontrace.EstimateAvailable || *direct.SuccessProbability != .4 || *direct.ExpectedOutputTokens != 128 ||
				math.Abs(*direct.EstimatedCost-(1000*1+128*2)/1e6) > 1e-12 || direct.Currency != "USD" || direct.Calibrated ||
				direct.Source != selectiontrace.EstimateSourceSystemOne || direct.Deployment != "primary" {
				t.Fatalf("direct estimate = %+v", direct)
			}
			if *high.SuccessProbability != .86 || *high.ExpectedOutputTokens != 1600 {
				t.Fatalf("high-effort estimate = %+v", high)
			}
			if late.Status != selectiontrace.EstimateDeadlineExceeded || late.Reason != "timeout" ||
				late.SuccessProbability != nil || late.ExpectedOutputTokens != nil || late.EstimatedCost != nil {
				t.Fatalf("late estimate = %+v", late)
			}
		})
	}
}

// Expected output tokens stay within the candidate's output capacity: the
// request's budget, the provider-rendered budget and the model's maximum.
func TestCandidateEstimateIsBoundedByOutputCapacity(t *testing.T) {
	cfg := DefaultMultiFactorConfig()
	// Requests that leave the output budget to the model need a cost
	// forecast; it does not bound the estimate.
	forecast := 512
	cfg.ExpectedOutputTokens = &forecast
	cfg.Estimates = (&estimateRecorder{answers: map[string]CandidateAnswer{
		"bounded/": {Status: selectiontrace.EstimateAvailable, SuccessProbability: .5, OutputLength: lengths(0, 0, 0, 1)},
		"capped/":  {Status: selectiontrace.EstimateAvailable, SuccessProbability: .5, OutputLength: lengths(0, 0, 0, 1)},
		"free/":    {Status: selectiontrace.EstimateAvailable, SuccessProbability: .5, OutputLength: lengths(0, 0, 0, 1)},
	}}).estimates()
	selector := buildMFSelector(cfg, map[string]config.ModelParams{
		"capped": {MaxOutputTokens: 2000},
		"free":   {Pricing: config.ModelPricing{Currency: "USD"}},
	}, nil, nil, nil)
	limit := int64(3000)
	result, err := selector.Select(context.Background(), &SelectionContext{
		CandidateModels:  []config.ModelRef{effortRef("bounded", ""), effortRef("capped", ""), effortRef("free", "")},
		CandidateDemands: map[string]CandidateDemand{"bounded": {MaxOutputTokens: &limit}},
	})
	if err != nil {
		t.Fatal(err)
	}
	estimates := result.MultiFactor.Estimates.Candidates
	if *estimates[0].ExpectedOutputTokens != 3000 || *estimates[1].ExpectedOutputTokens != 2000 || *estimates[2].ExpectedOutputTokens != 8192 {
		t.Fatalf("capacity bounds: %d %d %d", *estimates[0].ExpectedOutputTokens, *estimates[1].ExpectedOutputTokens, *estimates[2].ExpectedOutputTokens)
	}
	// Without rates there is no cost; explicit zero rates price a model free.
	if estimates[0].EstimatedCost != nil || estimates[2].EstimatedCost == nil || *estimates[2].EstimatedCost != 0 {
		t.Fatalf("costs: %+v %+v", estimates[0], estimates[2])
	}
}

// An answer the selector cannot read is recorded as unavailable, and the
// candidates over the SLO ceilings are not asked about.
func TestCandidateEstimatesOfEligibleCandidates(t *testing.T) {
	cfg := DefaultMultiFactorConfig()
	cfg.SLO = MultiFactorSLO{MaxInflight: 1}
	recorder := &estimateRecorder{answers: map[string]CandidateAnswer{
		"strange/": {Status: selectiontrace.EstimateAvailable, SuccessProbability: math.NaN(), OutputLength: lengths(1, 0, 0, 0)},
		"partial/": {Status: selectiontrace.EstimateAvailable, SuccessProbability: .5, OutputLength: map[string]float64{"under_256": 1}},
	}}
	cfg.Estimates = recorder.estimates()
	selector := buildMFSelector(cfg, nil, func(model string) int {
		if model == "busy" {
			return 5
		}
		return 0
	}, nil, nil)
	result, err := selector.Select(context.Background(), &SelectionContext{CandidateModels: candidates("strange", "busy", "partial", "silent")})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(recorder.asked[0], candidates("strange", "partial", "silent")) {
		t.Fatalf("asked about %v", recorder.asked[0])
	}
	reasons := []string{}
	for _, estimate := range result.MultiFactor.Estimates.Candidates {
		if estimate.Status != selectiontrace.EstimateUnavailable || estimate.SuccessProbability != nil {
			t.Fatalf("unreadable answer recorded as %+v", estimate)
		}
		reasons = append(reasons, estimate.Reason)
	}
	if !reflect.DeepEqual(reasons, []string{"invalid_success_probability", "invalid_output_length", ""}) {
		t.Fatalf("reasons %v", reasons)
	}

	// Every candidate over the SLO leaves nothing to estimate.
	cfg = DefaultMultiFactorConfig()
	cfg.SLO = MultiFactorSLO{MaxInflight: 1}
	recorder = &estimateRecorder{}
	cfg.Estimates = recorder.estimates()
	selector = buildMFSelector(cfg, nil, func(string) int { return 5 }, nil, nil)
	if _, err := selector.Select(context.Background(), &SelectionContext{CandidateModels: candidates("a", "b")}); err != nil || len(recorder.asked) != 0 {
		t.Fatalf("asked %v, error %v", recorder.asked, err)
	}
}

// When the quality floor leaves no candidate, the no-candidate policy still
// chooses, and the estimates of the eligible candidates are recorded with it.
func TestCandidateEstimatesSurviveTheNoCandidatePolicy(t *testing.T) {
	floor := 90.0
	cfg := DefaultMultiFactorConfig()
	cfg.QualityMinScore = &floor
	cfg.QualityOnMissing = config.QualityEvidenceOnMissingExclude
	cfg.Estimates = (&estimateRecorder{answers: map[string]CandidateAnswer{
		"a/": {Status: selectiontrace.EstimateAvailable, SuccessProbability: .7, OutputLength: lengths(0, 1, 0, 0)},
	}}).estimates()
	selector := buildMFSelector(cfg, map[string]config.ModelParams{"a": modelParamsWithTestQuality(.5)}, nil, nil, nil)
	result, err := selector.Select(context.Background(), &SelectionContext{CandidateModels: candidates("a")})
	if err != nil {
		t.Fatal(err)
	}
	if result.SelectedModel != "a" || result.MultiFactor == nil || len(result.MultiFactor.Estimates.Candidates) != 1 ||
		*result.MultiFactor.Estimates.Candidates[0].ExpectedOutputTokens != 640 {
		t.Fatalf("result = %+v", result)
	}
}
