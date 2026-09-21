package dsl

import (
	"reflect"
	"strings"
	"testing"
)

func TestEmbeddingCandidateBanksRoundTrip(t *testing.T) {
	input := `SIGNAL embedding media { threshold: 0.1 image_candidates: ["./positive.png"] negative_candidates: ["different subject"] negative_image_candidates: ["./negative.png"] query_modality: "image" aggregation_method: "max" }
 ROUTE choose { PRIORITY 1 WHEN embedding("media") MODEL "model" }`
	first, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	rendered, err := Decompile(first)
	if err != nil {
		t.Fatal(err)
	}
	second, errs := Compile(rendered)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	if !reflect.DeepEqual(first.EmbeddingRules, second.EmbeddingRules) {
		t.Fatalf("bank loss: %s", rendered)
	}
	ast := DecompileToAST(first)
	for _, field := range []string{"image_candidates", "negative_candidates", "negative_image_candidates"} {
		if !strings.Contains(rendered, field) || ast.Signals[0].Fields[field] == nil {
			t.Fatalf("missing %s", field)
		}
	}
}
