package classification

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestStateCapsOversizedParts(t *testing.T) {
	reading := systemOneReading{
		request:     "run the tests",
		turns:       []ConversationTurn{{Role: "user", Text: strings.Repeat("u", 6000)}},
		toolResults: []string{strings.Repeat("x", 20000), "pytest: all passed"},
	}
	rule := config.SystemOneRule{Input: config.SystemOneInput{PreviousTurns: 2, ToolResults: true}}
	req := reading.state(rule)
	if got := req.Parts["tool_results"]; len(got) != len("[state truncated] ")+toolResultsCapBytes {
		t.Fatalf("tool_results len = %d, want marker + %d", len(got), toolResultsCapBytes)
	}
	if !strings.HasSuffix(req.Parts["tool_results"], "pytest: all passed") {
		t.Fatalf("tail of tool results lost")
	}
	if got := req.Parts["previous_turns"]; len(got) != len("[state truncated] ")+previousTurnsCapBytes {
		t.Fatalf("previous_turns len = %d, want marker + %d", len(got), previousTurnsCapBytes)
	}
}

func TestStateKeepsSmallPartsVerbatim(t *testing.T) {
	reading := systemOneReading{
		request:     "run the tests",
		turns:       []ConversationTurn{{Role: "user", Text: "make it blue"}},
		toolResults: []string{"rows: 3"},
	}
	rule := config.SystemOneRule{Input: config.SystemOneInput{PreviousTurns: 2, ToolResults: true}}
	req := reading.state(rule)
	if got := req.Parts["tool_results"]; got != "rows: 3" {
		t.Fatalf("small tool result altered: %q", got)
	}
	if got := req.Parts["previous_turns"]; got != "user: make it blue" {
		t.Fatalf("small turn altered: %q", got)
	}
}
