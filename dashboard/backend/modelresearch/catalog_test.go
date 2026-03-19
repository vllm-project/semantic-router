package modelresearch

import "testing"

func TestRecipesResponseIncludesIntentAndPII(t *testing.T) {
	t.Parallel()

	resp := recipesResponse("http://router.internal:8080", "MoM", "amd", nil)
	got := make([]string, 0, len(resp.Recipes))
	for _, recipe := range resp.Recipes {
		got = append(got, recipe.Key)
	}

	want := []string{"feedback", "fact-check", "jailbreak", "intent", "pii", "domain"}
	if len(got) != len(want) {
		t.Fatalf("recipes length = %d, want %d (%v)", len(got), len(want), got)
	}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("recipes[%d] = %q, want %q", index, got[index], want[index])
		}
	}
}

func TestResolveRecipeSupportsExpandedClassifierTargets(t *testing.T) {
	t.Parallel()

	for _, target := range []string{"intent", "pii"} {
		def, err := resolveRecipe(target, GoalImproveAccuracy)
		if err != nil {
			t.Fatalf("resolveRecipe(%q) error = %v", target, err)
		}
		if def.Key != target {
			t.Fatalf("resolveRecipe(%q) key = %q, want %q", target, def.Key, target)
		}
	}
}
