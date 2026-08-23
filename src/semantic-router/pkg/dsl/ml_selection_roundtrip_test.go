package dsl

import (
	"strings"
	"testing"
)

func TestMLSelectionAlgorithmRoundTripsInsideRecipe(t *testing.T) {
	input := `
RECIPE learned {
  ROUTE choose {
    PRIORITY 100
    WHEN domain("math")
    ALGORITHM knn {
      ml: {
        models_path: "/models/selection"
        embedding_dim: 1024
        knn: { k: 7, pretrained_path: "/models/selection/knn.json" }
      }
    }
  }
  SIGNAL domain math { description: "Mathematics" }
}
`
	cfg, errs := Compile(input)
	if len(errs) != 0 {
		t.Fatalf("Compile() errors = %v", errs)
	}
	ml := cfg.Recipes[0].Profile.Decisions[0].Algorithm.ML
	if ml == nil || ml.KNN == nil || ml.KNN.K != 7 || ml.EmbeddingDim != 1024 {
		t.Fatalf("compiled algorithm.ml = %#v", ml)
	}

	output, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("Decompile() error = %v", err)
	}
	for _, fragment := range []string{"ALGORITHM knn", "ml:", "models_path:", "embedding_dim: 1024", "knn:", "k: 7"} {
		if !strings.Contains(output, fragment) {
			t.Fatalf("decompiled DSL missing %q:\n%s", fragment, output)
		}
	}
	roundTrip, roundTripErrs := Compile(output)
	if len(roundTripErrs) != 0 {
		t.Fatalf("round-trip Compile() errors = %v\n%s", roundTripErrs, output)
	}
	roundTripML := roundTrip.Recipes[0].Profile.Decisions[0].Algorithm.ML
	if roundTripML == nil || roundTripML.KNN == nil || roundTripML.KNN.PretrainedPath != "/models/selection/knn.json" {
		t.Fatalf("round-trip algorithm.ml = %#v", roundTripML)
	}
}

func TestMLSelectionDSLRequiresMatchingFamily(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{name: "missing ml", body: `ALGORITHM knn`, want: "requires algorithm.ml"},
		{name: "wrong family", body: `ALGORITHM knn { ml: { svm: { kernel: "rbf" } } }`, want: "requires only algorithm.ml.knn"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, errs := Compile("ROUTE choose { PRIORITY 1 " + test.body + " }")
			if len(errs) == 0 || !strings.Contains(errs[0].Error(), test.want) {
				t.Fatalf("Compile() errors = %v, want %q", errs, test.want)
			}
		})
	}
}
