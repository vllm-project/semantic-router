package selection

import (
	"context"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMultiFactorLexicographicPublishesOnlyFinalSurvivors(t *testing.T) {
	cfg := DefaultMultiFactorConfig()
	cfg.Objective = MultiFactorObjective{
		Strategy:   config.MultiFactorObjectiveLexicographic,
		Priorities: []MultiFactorPriority{{Factor: config.MultiFactorFactorQuality, Tolerance: .05}, {Factor: config.MultiFactorFactorCost, Tolerance: .1}},
	}
	params := map[string]config.ModelParams{}
	for name, values := range map[string][2]float64{
		"outside-quality": {.2, .01}, "expensive": {.99, 8}, "first": {.97, 1}, "second": {.95, 1.05},
	} {
		p := modelParamsWithTestQuality(values[0])
		p.Pricing = config.ModelPricing{PromptPer1M: values[1], CompletionPer1M: values[1]}
		params[name] = p
	}
	selector := buildMFSelector(cfg, params, nil, nil, nil)
	refs := candidates("outside-quality", "expensive", "first", "second")
	refs[2].LoRAName = "first-adapter"
	result, err := selector.Select(context.Background(), &SelectionContext{CandidateModels: refs})
	if err != nil {
		t.Fatal(err)
	}
	if result.SelectedModel != "first" || !reflect.DeepEqual(result.EligibleModels, refs[2:]) {
		t.Fatalf("selection lost final objective survivors: %+v", result)
	}
	if len(result.AllScores) != 4 {
		t.Fatal("eliminated candidate diagnostics must remain available")
	}
	result.EligibleModels[0].Model = "modified"
	if refs[2].Model != "first" {
		t.Fatal("published survivor slice aliases the caller inventory")
	}
}

func TestMultiFactorLexicographicMissingFactorKeepsDeclaredSurvivors(t *testing.T) {
	cfg := DefaultMultiFactorConfig()
	cfg.Objective = MultiFactorObjective{
		Strategy:   config.MultiFactorObjectiveLexicographic,
		Priorities: []MultiFactorPriority{{Factor: config.MultiFactorFactorQuality}},
	}
	selector := buildMFSelector(cfg, nil, nil, nil, nil)
	refs := candidates("first", "second")
	result, err := selector.Select(context.Background(), &SelectionContext{CandidateModels: refs})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.EligibleModels, refs) {
		t.Fatalf("a disabled factor must keep the evaluated inventory, not authorize outside models: %+v", result.EligibleModels)
	}
}
