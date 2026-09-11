package selection

import (
	"testing"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestSwitchGateDowngradeRequiresComparableEvidence(t *testing.T) {
	sel := NewSessionAwareSelector(nil)
	a := modelParamsWithTestQuality(0.9)
	b := modelParamsWithTestQuality(0.6)
	sel.InitializeFromConfig(map[string]config.ModelParams{"a": a, "b": b})
	ctx := &SelectionContext{CandidateModels: []config.ModelRef{{Model: "a"}, {Model: "b"}}}
	if !sel.IsDowngrade(ctx, "a", "b") || sel.IsDowngrade(ctx, "b", "a") {
		t.Fatal("same-index quality direction was not used")
	}
	ctx.CandidateModels[1].ReasoningEffort = "high"
	if sel.IsDowngrade(ctx, "a", "b") {
		t.Fatal("missing exact-effort evidence must not borrow the preferred score")
	}
	low := 40.0
	b.IndexResultsByEffort = map[string]map[string]modelcatalog.IndexResult{
		"high": {testIntelligenceIndex: {Index: testIntelligenceIndex, Status: "available", Score: &low}},
	}
	sel.InitializeFromConfig(map[string]config.ModelParams{"a": a, "b": b})
	if !sel.IsDowngrade(ctx, "a", "b") {
		t.Fatal("exact effort evidence was ignored")
	}
	b.QualityIndex = "different/index@1.0.0"
	sel.InitializeFromConfig(map[string]config.ModelParams{"a": a, "b": b})
	if sel.IsDowngrade(ctx, "a", "b") {
		t.Fatal("unrelated indexes are not comparable")
	}
}
