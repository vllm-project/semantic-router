package selection

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selectiontrace"
)

func TestMultiFactorLexicographicLatencyCoverage(t *testing.T) {
	for _, tc := range []struct {
		name      string
		latency   map[string]float64
		tolerance float64
		want      string
		action    selectiontrace.StageAction
		reason    selectiontrace.StageReason
		stages    int
	}{
		{"none", nil, 0, "cheap", selectiontrace.StageSkipped, selectiontrace.IncompleteLatencyCoverage, 2},
		{"only_expensive_measured_slow", map[string]float64{"expensive": 100}, 0, "cheap", selectiontrace.StageSkipped, selectiontrace.IncompleteLatencyCoverage, 2},
		{"only_expensive_measured_fast", map[string]float64{"expensive": .001}, 0, "cheap", selectiontrace.StageSkipped, selectiontrace.IncompleteLatencyCoverage, 2},
		{"only_cheap_measured", map[string]float64{"cheap": .1}, 0, "cheap", selectiontrace.StageSkipped, selectiontrace.IncompleteLatencyCoverage, 2},
		{"all_known", map[string]float64{"expensive": .1, "cheap": 1}, 0, "expensive", selectiontrace.StageApplied, selectiontrace.ToleranceBand, 1},
		{"known_zero", map[string]float64{"expensive": 0, "cheap": 1}, 0, "expensive", selectiontrace.StageApplied, selectiontrace.ToleranceBand, 1},
		{"all_known_within_band", map[string]float64{"expensive": 1, "cheap": 1.05}, .1, "cheap", selectiontrace.StageApplied, selectiontrace.ToleranceBand, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultMultiFactorConfig()
			cfg.LatencyMetric = "ttft"
			cfg.Objective = MultiFactorObjective{Strategy: config.MultiFactorObjectiveLexicographic, Priorities: []MultiFactorPriority{
				{Factor: config.MultiFactorFactorLatency, Tolerance: tc.tolerance}, {Factor: config.MultiFactorFactorCost},
			}}
			params := map[string]config.ModelParams{
				"expensive": {Pricing: config.ModelPricing{PromptPer1M: 10, CompletionPer1M: 10}},
				"cheap":     {Pricing: config.ModelPricing{PromptPer1M: 1, CompletionPer1M: 1}},
			}
			selector := buildMFSelector(cfg, params, nil, nil, func(model string, _ int) (float64, bool) {
				value, available := tc.latency[model]
				return value, available
			})
			refs := candidates("expensive", "cheap")
			ctx := &SelectionContext{CandidateModels: refs}
			result, err := selector.Select(context.Background(), ctx)
			if err != nil {
				t.Fatal(err)
			}
			if result.SelectedModel != tc.want || !reflect.DeepEqual(result.EligibleModels, candidates(tc.want)) {
				t.Fatalf("selection = %+v; want %s only", result, tc.want)
			}
			trace := result.MultiFactor
			if trace == nil || len(trace.Stages) != tc.stages || !reflect.DeepEqual(trace.FinalSurvivors, result.EligibleModels) {
				t.Fatalf("objective trace = %+v", trace)
			}
			stage := trace.Stages[0]
			if stage.Factor != "latency" || stage.Metric != "ttft" || stage.Percentile != 95 ||
				stage.Action != tc.action || stage.Reason != tc.reason || stage.Available != len(tc.latency) || stage.Total != 2 || stage.Tolerance != tc.tolerance {
				t.Fatalf("latency stage = %+v", stage)
			}
			for i, row := range stage.Candidates {
				if !reflect.DeepEqual(row.Candidate, refs[i]) {
					t.Fatal("stage lost input candidate ordering or identity")
				}
				value, available := tc.latency[row.Candidate.Model]
				if available != (row.Value != nil) || (available && (*row.Value != value || row.Metric != "ttft")) {
					t.Fatalf("candidate value = %+v; expected value=%v known=%v", row, value, available)
				}
				wantElimination := selectiontrace.EliminationReason("")
				if tc.action == selectiontrace.StageApplied && tc.stages == 1 && row.Candidate.Model != tc.want {
					wantElimination = selectiontrace.OutsideTolerance
				}
				if row.EliminationReason != wantElimination {
					t.Fatalf("candidate elimination = %q; want %q", row.EliminationReason, wantElimination)
				}
			}
			encoded, err := json.Marshal(trace)
			if err != nil {
				t.Fatal(err)
			}
			var decoded selectiontrace.MultiFactorObjective
			if decodeErr := json.Unmarshal(encoded, &decoded); decodeErr != nil || !reflect.DeepEqual(trace, &decoded) {
				t.Fatalf("typed JSON lost known-zero/unknown evidence: %s, %v", encoded, decodeErr)
			}
			repeated, err := selector.Select(context.Background(), ctx)
			if err != nil || !reflect.DeepEqual(trace, repeated.MultiFactor) {
				t.Fatalf("same evidence produced a different trace: %+v, %v", repeated, err)
			}
		})
	}
}

func TestMultiFactorLatencyCoverageUsesCurrentQualitySurvivors(t *testing.T) {
	cfg := DefaultMultiFactorConfig()
	cfg.LatencyMetric = "ttft"
	cfg.Objective = MultiFactorObjective{Strategy: config.MultiFactorObjectiveLexicographic, Priorities: []MultiFactorPriority{
		{Factor: config.MultiFactorFactorQuality}, {Factor: config.MultiFactorFactorLatency}, {Factor: config.MultiFactorFactorCost},
	}}
	params := map[string]config.ModelParams{
		"low-quality": modelParamsWithTestQuality(.1),
		"fast":        modelParamsWithTestQuality(.9),
		"slow":        modelParamsWithTestQuality(.9),
	}
	latencies := map[string]float64{"fast": .1, "slow": 1}
	selector := buildMFSelector(cfg, params, nil, nil, func(model string, _ int) (float64, bool) {
		value, available := latencies[model]
		return value, available
	})
	refs := candidates("low-quality", "fast", "slow")
	result, err := selector.Select(context.Background(), &SelectionContext{CandidateModels: refs})
	if err != nil {
		t.Fatal(err)
	}
	if result.SelectedModel != "fast" || !reflect.DeepEqual(result.EligibleModels, refs[1:2]) {
		t.Fatalf("quality exclusion/fully covered surviving latency pool changed: %+v", result)
	}
	stages := result.MultiFactor.Stages
	if len(stages) != 2 || stages[0].Candidates[0].EliminationReason != selectiontrace.OutsideTolerance ||
		stages[1].Action != selectiontrace.StageApplied || stages[1].Available != 2 || stages[1].Total != 2 ||
		len(stages[1].Candidates) != 2 || stages[1].Candidates[0].Candidate.Model != "fast" {
		t.Fatalf("stage coverage included an earlier elimination: %+v", stages)
	}
}

func TestMultiFactorLatencySkipPreservesMissingCostExclusion(t *testing.T) {
	cfg := DefaultMultiFactorConfig()
	cfg.LatencyMetric = "ttft"
	cfg.Objective = MultiFactorObjective{Strategy: config.MultiFactorObjectiveLexicographic, Priorities: []MultiFactorPriority{
		{Factor: config.MultiFactorFactorLatency}, {Factor: config.MultiFactorFactorCost},
	}}
	selector := buildMFSelector(cfg, map[string]config.ModelParams{
		"priced": {Pricing: config.ModelPricing{PromptPer1M: 10, CompletionPer1M: 10}},
	}, nil, nil, func(model string, _ int) (float64, bool) { return .1, model == "unpriced" })
	result, err := selector.Select(context.Background(), &SelectionContext{CandidateModels: candidates("unpriced", "priced")})
	if err != nil {
		t.Fatal(err)
	}
	if result.SelectedModel != "priced" || !reflect.DeepEqual(result.EligibleModels, candidates("priced")) {
		t.Fatalf("latency skip weakened missing cost policy: %+v", result)
	}
	stage := result.MultiFactor.Stages[1]
	if stage.Factor != "cost" || stage.Action != selectiontrace.StageApplied || stage.Available != 1 || stage.Total != 2 ||
		stage.Candidates[0].Value != nil || stage.Candidates[0].EliminationReason != selectiontrace.MissingMeasurement {
		t.Fatalf("missing cost was hidden or fabricated: %+v", stage)
	}
}

func TestMultiFactorTraceRetainsExactCandidateReferences(t *testing.T) {
	cfg := DefaultMultiFactorConfig()
	cfg.LatencyMetric = "ttft"
	cfg.Objective = MultiFactorObjective{Strategy: config.MultiFactorObjectiveLexicographic, Priorities: []MultiFactorPriority{{Factor: config.MultiFactorFactorLatency}}}
	selector := buildMFSelector(cfg, nil, nil, nil, func(string, int) (float64, bool) { return 0, false })
	enabled, disabled := true, false
	refs := []config.ModelRef{
		{Model: "shared", LoRAName: "first", Weight: 2, ModelReasoningControl: config.ModelReasoningControl{UseReasoning: &enabled, ReasoningEffort: "high"}},
		{Model: "shared", LoRAName: "second", Weight: 1, ModelReasoningControl: config.ModelReasoningControl{UseReasoning: &disabled, ReasoningEffort: "low"}},
	}
	result, err := selector.Select(context.Background(), &SelectionContext{CandidateModels: refs})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.MultiFactor.FinalSurvivors, refs) || len(result.MultiFactor.Stages[0].Candidates) != 2 {
		t.Fatalf("trace collapsed exact candidates: %+v", result.MultiFactor)
	}
	*result.MultiFactor.FinalSurvivors[0].UseReasoning = false
	result.MultiFactor.FinalSurvivors[0].Model = "changed"
	*result.MultiFactor.Stages[0].Candidates[1].Candidate.UseReasoning = true
	if !enabled || disabled || refs[0].Model != "shared" || !*result.MultiFactor.Stages[0].Candidates[0].Candidate.UseReasoning {
		t.Fatal("trace aliases caller candidates or other trace entries")
	}
}

func TestMultiFactorLatencySkipPreservesQualityEvidencePolicy(t *testing.T) {
	for _, policy := range []string{config.QualityEvidenceOnMissingExclude, config.QualityEvidenceOnMissingDisable} {
		t.Run(policy, func(t *testing.T) {
			cfg := DefaultMultiFactorConfig()
			cfg.QualityOnMissing = policy
			cfg.LatencyMetric = "ttft"
			cfg.Objective = MultiFactorObjective{Strategy: config.MultiFactorObjectiveLexicographic, Priorities: []MultiFactorPriority{
				{Factor: config.MultiFactorFactorQuality}, {Factor: config.MultiFactorFactorLatency}, {Factor: config.MultiFactorFactorCost},
			}}
			scored := modelParamsWithTestQuality(.9)
			scored.Pricing = config.ModelPricing{PromptPer1M: 10, CompletionPer1M: 10}
			selector := buildMFSelector(cfg, map[string]config.ModelParams{
				"scored": scored, "unscored": {Pricing: config.ModelPricing{PromptPer1M: 1, CompletionPer1M: 1}},
			}, nil, nil, func(model string, _ int) (float64, bool) { return .1, model == "scored" })
			result, err := selector.Select(context.Background(), &SelectionContext{CandidateModels: candidates("unscored", "scored")})
			if err != nil {
				t.Fatal(err)
			}
			stages := result.MultiFactor.Stages
			if policy == config.QualityEvidenceOnMissingExclude {
				if result.SelectedModel != "scored" || !reflect.DeepEqual(result.EligibleModels, candidates("scored")) ||
					len(stages) != 1 || stages[0].Total != 1 || stages[0].Available != 1 {
					t.Fatalf("latency policy restored an excluded quality candidate: %+v", result)
				}
				return
			}
			if result.SelectedModel != "unscored" || len(stages) != 3 ||
				stages[0].Action != selectiontrace.StageSkipped || stages[0].Reason != selectiontrace.NoAvailableValues ||
				stages[0].Available != 0 || stages[0].Total != 2 || stages[1].Reason != selectiontrace.IncompleteLatencyCoverage {
				t.Fatalf("disabled quality or partial latency did not defer to cost: %+v", result)
			}
		})
	}
}

func TestMultiFactorWeightedOmitsObjectiveTrace(t *testing.T) {
	selector := buildMFSelector(DefaultMultiFactorConfig(), nil, nil, nil, nil)
	result, err := selector.Select(context.Background(), &SelectionContext{CandidateModels: candidates("first", "second")})
	if err != nil || result.MultiFactor != nil {
		t.Fatalf("weighted trace = %+v, error = %v", result, err)
	}
}
