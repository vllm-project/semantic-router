package modelresearch

import "testing"

func TestExpandedClassifierMappings(t *testing.T) {
	t.Parallel()

	if got := runtimeDatasetForRecipe("intent", ""); got != "mmlu-prox-en" {
		t.Fatalf("runtimeDatasetForRecipe(intent) = %q, want mmlu-prox-en", got)
	}
	if got := runtimeDatasetForRecipe("pii", ""); got != "" {
		t.Fatalf("runtimeDatasetForRecipe(pii) = %q, want empty", got)
	}
	if got := offlineEvalModelKey("intent"); got != "intent" {
		t.Fatalf("offlineEvalModelKey(intent) = %q, want intent", got)
	}
	if got := offlineEvalModelKey("pii"); got != "pii" {
		t.Fatalf("offlineEvalModelKey(pii) = %q, want pii", got)
	}
	if got := offlineDatasetOverride("intent", "custom-intent.json"); got != "custom-intent.json" {
		t.Fatalf("offlineDatasetOverride(intent) = %q, want custom-intent.json", got)
	}
	if got := offlineDatasetOverride("pii", "presidio"); got != "" {
		t.Fatalf("offlineDatasetOverride(pii) = %q, want empty", got)
	}
}
