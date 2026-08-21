package accesscontrol

import (
	"reflect"
	"testing"
)

func budgetIDs(items []Budget) []string {
	ids := make([]string, 0, len(items))
	for _, item := range items {
		ids = append(ids, item.ID)
	}
	return ids
}

func TestEffectiveBudgetsKeepsGlobalAndUsesUserBeforeTeam(t *testing.T) {
	got := effectiveBudgets("", []Budget{
		{ID: "team", ScopeType: "team"},
		{ID: "global", ScopeType: "global"},
		{ID: "user", ScopeType: "user"},
	})
	want := []string{"global", "user"}
	if !reflect.DeepEqual(budgetIDs(got), want) {
		t.Fatalf("effectiveBudgets() = %#v, want %#v", budgetIDs(got), want)
	}
}

func TestEffectiveBudgetsFallsBackToTeam(t *testing.T) {
	got := effectiveBudgets("", []Budget{{ID: "team", ScopeType: "team"}})
	want := []string{"team"}
	if !reflect.DeepEqual(budgetIDs(got), want) {
		t.Fatalf("effectiveBudgets() = %#v, want %#v", budgetIDs(got), want)
	}
}

func TestEffectiveBudgetsUsesKeyTierBeforeUserAndTeam(t *testing.T) {
	got := effectiveBudgets("linked", []Budget{
		{ID: "team", ScopeType: "team"},
		{ID: "user", ScopeType: "user"},
		{ID: "inline", ScopeType: "key"},
		{ID: "linked", ScopeType: "team"},
	})
	want := []string{"inline", "linked"}
	if !reflect.DeepEqual(budgetIDs(got), want) {
		t.Fatalf("effectiveBudgets() = %#v, want %#v", budgetIDs(got), want)
	}
}

func TestEffectiveBudgetsDeduplicatesLinkedDirectBudget(t *testing.T) {
	got := effectiveBudgets("key", []Budget{{ID: "key", ScopeType: "key"}})
	want := []string{"key"}
	if !reflect.DeepEqual(budgetIDs(got), want) {
		t.Fatalf("effectiveBudgets() = %#v, want %#v", budgetIDs(got), want)
	}
}
