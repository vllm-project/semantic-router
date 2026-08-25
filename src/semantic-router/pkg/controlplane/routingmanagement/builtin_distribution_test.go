package routingmanagement

import (
	"bytes"
	"context"
	"errors"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"
)

func TestCanonicalBuiltInRecipeDistributionParsesAsRoutingRecipes(t *testing.T) {
	_, source, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve test source")
	}
	directory := source
	for range 6 {
		directory = filepath.Dir(directory)
	}
	distribution, err := LoadBuiltInRecipeDistribution(
		directory + "/config/recipes/built-in/latest/mom-v1",
	)
	if err != nil {
		t.Fatal(err)
	}
	if distribution.ID != "mom-v1" || distribution.Version == "" || len(distribution.Recipes) != 5 {
		t.Fatalf("distribution identity = %s@%s with %d Recipes",
			distribution.ID, distribution.Version, len(distribution.Recipes))
	}
	first, err := distribution.RecipesForNamespace("11111111-1111-4111-8111-111111111111")
	if err != nil {
		t.Fatal(err)
	}
	second, err := distribution.RecipesForNamespace("22222222-2222-4222-8222-222222222222")
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != len(distribution.Recipes) || first[0].ID == second[0].ID || first[0].Name != second[0].Name {
		t.Fatalf("namespace projection is not stable and isolated: first=%#v second=%#v", first[0], second[0])
	}
	for _, recipe := range first {
		if recipe.Revision != 1 || len(recipe.Decisions) == 0 {
			t.Fatalf("invalid installed Recipe projection: %#v", recipe)
		}
		if string(recipe.Document) == "" || containsRecipeModelBinding(recipe.Document) {
			t.Fatalf("built-in Recipe retained a physical Model binding: %s", recipe.Document)
		}
	}
}

func TestBuiltInRecipeVersionCreatesSiblingIdentity(t *testing.T) {
	first := testBuiltInDistribution(t, "1.0.0", "Simple")
	second := testBuiltInDistribution(t, "1.1.0", "Simple v2")
	namespaceID := "11111111-1111-4111-8111-111111111111"
	firstRecipes, err := first.RecipesForNamespace(namespaceID)
	if err != nil {
		t.Fatal(err)
	}
	secondRecipes, err := second.RecipesForNamespace(namespaceID)
	if err != nil {
		t.Fatal(err)
	}
	if firstRecipes[0].ID == secondRecipes[0].ID {
		t.Fatal("a new distribution version reused an immutable Recipe identity")
	}
	if got, want := distributionRecipeID(namespaceID, first.ID, first.Version, first.Recipes[0].SourceID), firstRecipes[0].ID; got != want {
		t.Fatalf("deterministic Recipe id = %q, want %q", got, want)
	}
}

func TestBuiltInRecipeDistributionRejectsNonRecipeAuthority(t *testing.T) {
	metadata := []byte("schema_version: vllm-sr/recipe-metadata/v1\nid: test-recipes\nname: Test Recipes\nversion: 1.0.0\n")
	recipe := "recipes:\n  - name: Simple\n    routing:\n      decisions:\n        - name: direct\n          rules: {}\n"
	tests := map[string]string{
		"listener":   "listeners:\n  - name: public\n    address: 0.0.0.0\n    port: 8899\n",
		"model":      "providers:\n  models:\n    - name: physical\n      backend_refs: []\n",
		"entrypoint": "entrypoints:\n  - model_names: [public]\n    recipe: Simple\n    assignments: {}\n",
		"global":     "global: {}\n",
		"billing":    "billing_currency: USD\n",
	}
	for name, authority := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := ParseBuiltInRecipeDistribution(
				metadata,
				[]byte("version: v0.3\n"+authority+recipe),
			)
			if !errors.Is(err, ErrInvalid) {
				t.Fatalf("error = %v, want ErrInvalid", err)
			}
		})
	}
}

func TestBuiltInRecipeInstallerReconcilesRestartAndFutureNamespace(t *testing.T) {
	distribution := testBuiltInDistribution(t, "1.0.0", "Simple")
	store := &fakeBuiltInRecipeStore{pending: map[string]bool{
		"11111111-1111-4111-8111-111111111111": true,
		"22222222-2222-4222-8222-222222222222": true,
	}}
	clock := time.Date(2026, time.August, 23, 10, 0, 0, 0, time.UTC)
	installer, newBuiltInRecipeInstallerErr := NewBuiltInRecipeInstaller(BuiltInRecipeInstallerOptions{
		Store: store, Distribution: distribution, Now: func() time.Time { return clock },
	})
	if newBuiltInRecipeInstallerErr != nil {
		t.Fatal(newBuiltInRecipeInstallerErr)
	}
	if err := installer.Ready(context.Background()); err == nil {
		t.Fatal("installer was ready before its initial durable reconciliation")
	}
	if err := installer.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := installer.Ready(context.Background()); err != nil {
		t.Fatal(err)
	}
	if store.installCalls != 2 || store.verifyCalls != 1 {
		t.Fatalf("initial reconciliation calls = install %d verify %d", store.installCalls, store.verifyCalls)
	}

	restarted, newBuiltInRecipeInstallerErr := NewBuiltInRecipeInstaller(BuiltInRecipeInstallerOptions{
		Store: store, Distribution: distribution,
	})
	if newBuiltInRecipeInstallerErr != nil {
		t.Fatal(newBuiltInRecipeInstallerErr)
	}
	if err := restarted.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	if store.installCalls != 2 || store.verifyCalls != 2 {
		t.Fatalf("restart was not idempotent: install %d verify %d", store.installCalls, store.verifyCalls)
	}

	store.mu.Lock()
	store.pending["33333333-3333-4333-8333-333333333333"] = true
	store.mu.Unlock()
	if err := restarted.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	if store.installCalls != 3 || store.verifyCalls != 3 {
		t.Fatalf("future Namespace was not reconciled: install %d verify %d", store.installCalls, store.verifyCalls)
	}
}

func testBuiltInDistribution(t *testing.T, version, decisionName string) BuiltInRecipeDistribution {
	t.Helper()
	metadata := []byte("schema_version: vllm-sr/recipe-metadata/v1\nid: test-recipes\nname: Test Recipes\nversion: " + version + "\n")
	config := []byte("version: v0.3\nrecipes:\n" +
		"  - name: Simple\n" +
		"    description: A reusable test Recipe.\n" +
		"    routing:\n" +
		"      decisions:\n" +
		"        - name: " + decisionName + "\n" +
		"          rules: {}\n")
	distribution, err := ParseBuiltInRecipeDistribution(metadata, config)
	if err != nil {
		t.Fatal(err)
	}
	return distribution
}

func containsRecipeModelBinding(document []byte) bool {
	for _, marker := range [][]byte{[]byte("model_private"), []byte("modelRefs"), []byte("analysisModels")} {
		if bytes.Contains(document, marker) {
			return true
		}
	}
	return false
}

type fakeBuiltInRecipeStore struct {
	mu           sync.Mutex
	pending      map[string]bool
	installCalls int
	verifyCalls  int
}

func (store *fakeBuiltInRecipeStore) PendingBuiltInRecipeNamespaces(
	_ context.Context, _ BuiltInRecipeDistribution, limit int,
) ([]string, error) {
	store.mu.Lock()
	defer store.mu.Unlock()
	result := make([]string, 0, limit)
	for namespaceID, pending := range store.pending {
		if pending {
			result = append(result, namespaceID)
			if len(result) == limit {
				break
			}
		}
	}
	return result, nil
}

func (store *fakeBuiltInRecipeStore) InstallBuiltInRecipes(
	_ context.Context, namespaceID string, distribution BuiltInRecipeDistribution, _ MutationContext,
) ([]Recipe, error) {
	if _, err := distribution.RecipesForNamespace(namespaceID); err != nil {
		return nil, err
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	if !store.pending[namespaceID] {
		return nil, errors.New("Namespace is not pending")
	}
	store.pending[namespaceID] = false
	store.installCalls++
	return []Recipe{}, nil
}

func (store *fakeBuiltInRecipeStore) VerifyBuiltInRecipes(
	_ context.Context, _ BuiltInRecipeDistribution,
) error {
	store.mu.Lock()
	defer store.mu.Unlock()
	for _, pending := range store.pending {
		if pending {
			return errors.New("pending Namespace remains")
		}
	}
	store.verifyCalls++
	return nil
}

var _ BuiltInRecipeStore = (*fakeBuiltInRecipeStore)(nil)
