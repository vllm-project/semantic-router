package routingmanagement

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func TestCompileRecipeDocumentDerivesIdentityAndDispatchCardinality(t *testing.T) {
	document := json.RawMessage(`{
  "signals": {},
  "projections": {},
  "decisions": [
    {"name":"Static","rules":{},"algorithm":{"type":"static"}},
    {"name":"Fusion","rules":{},"algorithm":{"type":"fusion","fusion":{}}}
  ]
}`)
	_, decisions, err := CompileRecipeDocument("recipe_test", document)
	if err != nil {
		t.Fatal(err)
	}
	cardinality := make(map[string]routingsnapshot.DispatchCardinality, len(decisions))
	for _, decision := range decisions {
		if decision.ID == "" {
			t.Fatal("compiled Decision identity is empty")
		}
		cardinality[decision.Name] = decision.DispatchCardinality
	}
	if cardinality["Static"] != routingsnapshot.DispatchCardinalitySingle ||
		cardinality["Fusion"] != routingsnapshot.DispatchCardinalityMulti {
		t.Fatalf("dispatch cardinality = %#v", cardinality)
	}
}

func TestCompileRecipeDocumentRejectsAuthoredDecisionIdentity(t *testing.T) {
	document := json.RawMessage(`{"decisions":[{"id":"decision_static","name":"Static","rules":{}}]}`)
	if _, _, err := CompileRecipeDocument("recipe_test", document); err == nil {
		t.Fatal("CompileRecipeDocument accepted compiler-owned Decision identity")
	}
}

func TestCompileRecipeAllowsEmptyDraftButRuntimeSnapshotRejectsIt(t *testing.T) {
	document := json.RawMessage(`{"signals":{},"projections":{},"decisions":[]}`)
	recipe, err := compileRecipe(RecipeInput{
		ID: "recipe_empty", Name: "Empty draft", Document: document,
	}, 1)
	if err != nil {
		t.Fatalf("compileRecipe() draft error = %v", err)
	}
	if len(recipe.Decisions) != 0 {
		t.Fatalf("draft decisions = %+v, want none", recipe.Decisions)
	}
	if _, err := routingsnapshot.Compile(routingsnapshot.Bundle{
		NamespaceID: "namespace", Revision: 1, Recipes: []routingsnapshot.Recipe{recipe},
	}); err == nil {
		t.Fatal("runtime snapshot accepted an empty Recipe")
	}
}
