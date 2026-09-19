package sessiontelemetry

import (
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRoutingSessionKeyPreservesOrdinaryIdentities(t *testing.T) {
	for _, tc := range []struct {
		recipe     config.RecipeName
		components []string
		want       string
	}{
		{components: []string{"session-a"}, want: "session-a"},
		{recipe: config.DefaultRecipeName, components: []string{"session-a", "conversation-a"}, want: "session-a/conversation-a"},
		{recipe: "speed", components: []string{"session-a"}, want: "speed::session-a"},
		{recipe: "speed", components: []string{"session-a", "conversation-a"}, want: "speed::session-a%2Fconversation-a"},
	} {
		if got := RoutingSessionKey(tc.recipe, tc.components...); got != tc.want {
			t.Errorf("recipe=%q components=%q: got %q, want %q", tc.recipe, tc.components, got, tc.want)
		}
	}
}

func TestRoutingSessionKeyRejectsBlankComponents(t *testing.T) {
	for _, components := range [][]string{nil, {}, {""}, {" \t "}, {"", "conversation"}, {"session", ""}, {"session", " \t "}} {
		for _, recipe := range []config.RecipeName{config.DefaultRecipeName, "speed"} {
			if got := RoutingSessionKey(recipe, components...); got != "" {
				t.Errorf("recipe=%q components=%q created identity %q", recipe, components, got)
			}
		}
	}
}

func TestRoutingSessionKeyDoesNotMergeDistinctTuples(t *testing.T) {
	// Include real delimiters and their literal encoded spellings. Neither a
	// client component nor the default recipe can impersonate another tuple.
	components := []string{"client", "team/run", "team%2Frun", "team%2frun", "team::client", "speed::client", "a+b", "a b", "%", "团队/运行", "任务", "/", ":"}
	recipes := []config.RecipeName{config.DefaultRecipeName, "speed", "speed::team", "speed%3A%3Ateam", "团队"}
	seen := make(map[string]string)
	check := func(recipe config.RecipeName, tuple []string) {
		t.Helper()
		identity := fmt.Sprintf("%q:%q", recipe, tuple)
		key := RoutingSessionKey(recipe, tuple...)
		if key == "" {
			t.Fatalf("valid tuple %s was rejected", identity)
		}
		if repeated := RoutingSessionKey(recipe, tuple...); repeated != key {
			t.Fatalf("same tuple %s changed key from %q to %q", identity, key, repeated)
		}
		if prior, exists := seen[key]; exists && prior != identity {
			t.Fatalf("distinct tuples %s and %s share key %q", prior, identity, key)
		}
		seen[key] = identity
	}
	for _, recipe := range recipes {
		for _, sid := range components {
			check(recipe, []string{sid})
			for _, cid := range components {
				check(recipe, []string{sid, cid})
			}
		}
	}
	if implicit := RoutingSessionKey("", "client"); implicit != RoutingSessionKey(config.DefaultRecipeName, "client") {
		t.Fatalf("implicit default recipe differs from explicit default: %q", implicit)
	}
}
