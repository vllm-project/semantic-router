package looper

import (
	"strings"
	"testing"
)

func TestWorkflowPlannerPromptMatchesRequiredWorkerCount(t *testing.T) {
	for _, tc := range []struct {
		minimum int
		want    string
	}{
		{0, "models per step: 1 to 3"},
		{2, "models per step: 2 to 3"},
	} {
		prompt := buildWorkflowPlannerPrompt("Summarize the meeting agenda.", []string{"one", "two", "three"}, workflowsExecutionConfig{MaxSteps: 2, MaxParallel: 3, MinSuccessfulResponses: tc.minimum}, nil)
		if !strings.Contains(prompt, tc.want) {
			t.Fatalf("planner prompt omitted worker requirement %q", tc.want)
		}
	}
}
