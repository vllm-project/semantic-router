package accesscontrol

import "testing"

func TestEffectiveBudgetIDUsesKeyBeforeUserAndTeam(t *testing.T) {
	if got := effectiveBudgetID("key", "user", "team"); got != "key" {
		t.Fatalf("effectiveBudgetID() = %q, want key", got)
	}
}

func TestEffectiveBudgetIDUsesUserBeforeTeam(t *testing.T) {
	if got := effectiveBudgetID("", "user", "team"); got != "user" {
		t.Fatalf("effectiveBudgetID() = %q, want user", got)
	}
}

func TestEffectiveBudgetIDFallsBackToTeam(t *testing.T) {
	if got := effectiveBudgetID("", "", "team"); got != "team" {
		t.Fatalf("effectiveBudgetID() = %q, want team", got)
	}
}
