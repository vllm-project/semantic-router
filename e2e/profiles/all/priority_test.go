package all

import (
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"
)

// The baseline's standard scope leaves out only its P2 load stages, and
// its p0 scope keeps the routed request and the refusal contracts.
func TestBaselineScopesFollowCasePriority(t *testing.T) {
	scoped := func(scope string) []string {
		var names []string
		for _, name := range testmatrix.BaselineCases() {
			tc, ok := pkgtestcases.Get(name)
			if !ok {
				t.Fatalf("baseline case %s is not registered", name)
			}
			in, err := pkgtestcases.InScope(tc, scope)
			if err != nil {
				t.Fatal(err)
			}
			if in {
				names = append(names, name)
			}
		}
		return names
	}
	full, standard, p0 := scoped("full"), scoped("standard"), scoped("p0")
	if len(full)-len(standard) != 1 || slices.Contains(standard, "chat-completions-progressive-stress") {
		t.Fatalf("standard scope must leave out only the progressive stress case: %v", standard)
	}
	for _, name := range []string{"chat-completions-request", "plugin-short-circuit-no-dispatch", "pii-detection", "jailbreak-detection"} {
		if !slices.Contains(p0, name) {
			t.Fatalf("p0 scope misses %s: %v", name, p0)
		}
	}
}

// A P0 case no profile selects would be a priority nothing runs.
func TestEveryP0CaseIsSelectedByAProfile(t *testing.T) {
	selected := map[string]bool{}
	for _, name := range testmatrix.BaselineCases() {
		selected[name] = true
	}
	for _, profileName := range framework.RegisteredProfileNames() {
		profile, err := framework.NewProfileByName(profileName)
		if err != nil {
			t.Fatal(err)
		}
		for _, name := range profile.GetTestCases() {
			selected[name] = true
		}
	}
	for _, tc := range pkgtestcases.List() {
		if tc.EffectivePriority() == pkgtestcases.P0 && !selected[tc.Name] {
			t.Errorf("P0 case %s is selected by no profile", tc.Name)
		}
	}
}
