package config

import "testing"

func TestEntrypointRuleIsCatchAll(t *testing.T) {
	cases := []struct {
		name string
		rule EntrypointRule
		want bool
	}{
		{"no matches at all", EntrypointRule{Matches: nil}, true},
		{"one unconstrained match", EntrypointRule{Matches: []EntrypointMatch{{}}}, true},
		{"unconstrained among constrained", EntrypointRule{Matches: []EntrypointMatch{
			{Headers: []HeaderMatcher{{Name: "x-a", Value: "1"}}},
			{},
		}}, true},
		{"only constrained", EntrypointRule{Matches: []EntrypointMatch{
			{Headers: []HeaderMatcher{{Name: "x-a", Value: "1"}}},
		}}, false},
		{"path-only constrained", EntrypointRule{Matches: []EntrypointMatch{
			{Path: &PathMatcher{Type: PathMatchExact, Value: "/v1/chat/completions"}},
		}}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.rule.IsCatchAll(); got != c.want {
				t.Errorf("IsCatchAll() = %v, want %v", got, c.want)
			}
		})
	}
}

func TestEntrypointRecipeNames(t *testing.T) {
	t.Run("legacy entrypoint returns its single recipe", func(t *testing.T) {
		got := entrypointRecipeNames(EntrypointMapping{Recipe: "default"})
		if len(got) != 1 || got[0] != "default" {
			t.Fatalf("entrypointRecipeNames() = %v, want [default]", got)
		}
	})

	t.Run("legacy entrypoint with no recipe returns nothing", func(t *testing.T) {
		got := entrypointRecipeNames(EntrypointMapping{})
		if len(got) != 0 {
			t.Fatalf("entrypointRecipeNames() = %v, want empty", got)
		}
	})

	t.Run("rules-based entrypoint returns every distinct rule recipe", func(t *testing.T) {
		got := entrypointRecipeNames(EntrypointMapping{Rules: []EntrypointRule{
			{Name: "a", Recipe: "recipe-a"},
			{Name: "b", Recipe: "recipe-b"},
			{Name: "c", Recipe: "recipe-a"}, // duplicate, must be deduped
		}})
		if len(got) != 2 {
			t.Fatalf("entrypointRecipeNames() = %v, want 2 distinct names", got)
		}
		seen := map[RecipeName]bool{}
		for _, n := range got {
			seen[n] = true
		}
		if !seen["recipe-a"] || !seen["recipe-b"] {
			t.Fatalf("entrypointRecipeNames() = %v, want recipe-a and recipe-b", got)
		}
	})
}
