package testcases

import "testing"

// Guarding retired testcases stops a revert from silently re-introducing it.
var retiredTestCaseNames = []string{
	"rl-driven-basic-selection",
	"rl-driven-personalization",
	"rl-driven-feedback-loop",
	"rl-driven-multi-turn",
	"rl-driven-exploration",
}

func TestRetiredTestCasesAreNotRegistered(t *testing.T) {
	for _, name := range retiredTestCaseNames {
		if _, ok := Get(name); ok {
			t.Errorf("retired test case %q is still registered in the E2E catalog", name)
		}
	}
}
