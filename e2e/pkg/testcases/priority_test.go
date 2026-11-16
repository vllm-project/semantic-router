package testcases

import "testing"

func TestInScopeRunsEachPriorityInItsScopes(t *testing.T) {
	want := map[Priority]map[string]bool{
		P0: {"p0": true, "standard": true, "full": true},
		P1: {"p0": false, "standard": true, "full": true},
		P2: {"p0": false, "standard": false, "full": true},
		"": {"p0": false, "standard": true, "full": true},
	}
	for priority, scopes := range want {
		for scope, runs := range scopes {
			got, err := InScope(TestCase{Priority: priority}, scope)
			if err != nil || got != runs {
				t.Fatalf("priority %q in scope %q: got %t, %v; want %t", priority, scope, got, err, runs)
			}
		}
	}
	if _, err := InScope(TestCase{}, "stnadard"); err == nil {
		t.Fatal("unknown scope accepted")
	}
}
