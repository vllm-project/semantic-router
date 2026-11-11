package dsl

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A projection input written with match: 0 keeps it through decompile and
// compile, and an omitted match stays omitted.
func TestDecompileRoutingKeepsExplicitZeroProjectionMatch(t *testing.T) {
	inputs := []config.ProjectionScoreInput{
		{Type: config.SignalTypeKeyword, Name: "greeting", Weight: 1, Match: floatPtr(0), Miss: 1},
		{Type: config.SignalTypeKeyword, Name: "greeting", Weight: 0.5},
	}
	cfg := &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{
		Signals:     config.Signals{KeywordRules: []config.KeywordRule{{Name: "greeting", Operator: "OR", Keywords: []string{"hello"}}}},
		Projections: config.Projections{Scores: []config.ProjectionScore{{Name: "substance", Method: "weighted_sum", Inputs: inputs}}},
	}}
	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting error: %v", err)
	}
	roundTripped, errs := Compile(dslText)
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}
	if got := roundTripped.Projections.Scores[0].Inputs; !reflect.DeepEqual(got, inputs) {
		t.Fatalf("round-trip inputs = %+v, want %+v\n%s", got, inputs, dslText)
	}
}
