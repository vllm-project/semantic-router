package websearch

import (
	"strings"
	"testing"
)

func TestRandomUserAgentRotatesThroughBrowserAgents(t *testing.T) {
	seen := map[string]bool{}
	for i := 0; i < 50; i++ {
		agent := RandomUserAgent()
		if !strings.HasPrefix(agent, "Mozilla/5.0") {
			t.Fatalf("RandomUserAgent() = %q, want a browser user agent", agent)
		}
		seen[agent] = true
	}
	if len(seen) < 2 {
		t.Fatalf("RandomUserAgent() returned %d distinct agents in 50 draws, want rotation", len(seen))
	}
}

func TestDefaultUserAgentsReturnsACopy(t *testing.T) {
	agents := DefaultUserAgents()
	if len(agents) == 0 {
		t.Fatal("DefaultUserAgents() is empty")
	}
	agents[0] = "mutated"

	if again := DefaultUserAgents(); again[0] == "mutated" {
		t.Fatal("DefaultUserAgents() shares its backing array with the caller")
	}
}
