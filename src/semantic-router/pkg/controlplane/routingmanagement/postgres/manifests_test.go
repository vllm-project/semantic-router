package postgres

import (
	"reflect"
	"testing"

	"github.com/google/uuid"
)

func TestManifestPlanReportsStableNamesAndKeepsGeneratedIDsInternal(t *testing.T) {
	snapshot := bootstrapRoutingSnapshot(t)
	namespaceID := uuid.NewString()
	first, err := buildManifestPlan(namespaceID, snapshot, manifestState{currency: "USD"})
	if err != nil {
		t.Fatal(err)
	}
	second, err := buildManifestPlan(namespaceID, snapshot, manifestState{currency: "USD"})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first.diff, second.diff) {
		t.Fatalf("equivalent dry runs produced different diffs: %#v != %#v", first.diff, second.diff)
	}
	if !reflect.DeepEqual(first.diff.Models.Create, []string{"Model One"}) ||
		!reflect.DeepEqual(first.diff.Recipes.Create, []string{"Recipe One"}) ||
		!reflect.DeepEqual(first.diff.Entrypoints.Create, []string{"Entrypoint One"}) {
		t.Fatalf("manifest diff did not use human names: %#v", first.diff)
	}
	if len(first.targetIDs) != 3 || reflect.DeepEqual(first.targetIDs, second.targetIDs) {
		t.Fatalf("compiler-owned target IDs were absent or leaked into the stable diff")
	}
}
