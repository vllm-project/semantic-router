package dsl

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Mirrors an unedited Builder or CLI deploy of the shipped config: decompile, compile, merge, validate.
func TestReferenceConfigSurvivesDSLRoundTrip(t *testing.T) {
	base, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "config", "config.yaml"))
	if err != nil {
		t.Fatalf("read config/config.yaml: %v", err)
	}
	original, err := config.ParseYAMLBytesDeferringEnv(base)
	if err != nil {
		t.Fatalf("ParseYAMLBytesDeferringEnv error: %v", err)
	}
	dslText, err := Decompile(original)
	if err != nil {
		t.Fatalf("Decompile error: %v", err)
	}
	compiled, errs := Compile(dslText)
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}
	merged, err := MergeRoutingIntoBase(compiled, base)
	if err != nil {
		t.Fatalf("MergeRoutingIntoBase error: %v", err)
	}
	deployed, err := config.ParseYAMLBytes(merged)
	if err != nil {
		t.Fatalf("round-tripped config/config.yaml fails validation: %v", err)
	}
	if !reflect.DeepEqual(original.Entrypoints, deployed.Entrypoints) {
		t.Fatalf("entrypoints changed in the round trip:\n%+v\n%+v", original.Entrypoints, deployed.Entrypoints)
	}
	if diff := cmp.Diff(decisionModelCandidates(original), decisionModelCandidates(deployed)); diff != "" {
		t.Fatalf("decision model candidates changed in the round trip (-original +round-tripped):\n%s", diff)
	}
}

func decisionModelCandidates(cfg *config.RouterConfig) map[string][]string {
	candidates := make(map[string][]string)
	for _, ref := range cfg.RoutingDecisionRefs() {
		key := config.RoutingDecisionKey(ref.Recipe, ref.Decision.Name)
		for _, model := range ref.Decision.ModelRefs {
			candidates[key] = append(candidates[key], model.Model+"/"+model.LoRAName)
		}
	}
	return candidates
}
