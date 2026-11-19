package classification

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestStateCapsOversizedParts(t *testing.T) {
	reading := systemOneReading{
		request:     strings.Repeat("r", 6000),
		turns:       []ConversationTurn{{Role: "user", Text: strings.Repeat("u", 6000)}},
		toolResults: []string{strings.Repeat("x", 20000), "pytest: all passed"},
	}
	rule := config.SystemOneRule{Input: config.SystemOneInput{PreviousTurns: 2, ToolResults: true, MaxStateBytes: 3072}}
	req := reading.state(rule)
	for _, part := range []string{"request", "previous_turns", "tool_results"} {
		got := req.Parts[part]
		if len(got) != len("[state truncated] ")+3072 {
			t.Fatalf("%s len = %d, want marker + %d", part, len(got), 3072)
		}
	}
	if !strings.HasSuffix(req.Parts["tool_results"], "pytest: all passed") {
		t.Fatalf("tail of tool results lost")
	}
}

func TestStateKeepsSmallPartsVerbatim(t *testing.T) {
	reading := systemOneReading{
		request:     "run the tests",
		turns:       []ConversationTurn{{Role: "user", Text: "make it blue"}},
		toolResults: []string{"rows: 3"},
	}
	rule := config.SystemOneRule{Input: config.SystemOneInput{PreviousTurns: 2, ToolResults: true, MaxStateBytes: 3072}}
	req := reading.state(rule)
	if got := req.Parts["tool_results"]; got != "rows: 3" {
		t.Fatalf("small tool result altered: %q", got)
	}
	if got := req.Parts["previous_turns"]; got != "user: make it blue" {
		t.Fatalf("small turn altered: %q", got)
	}
}

func TestStateDefaultKeepsPartsWhole(t *testing.T) {
	reading := systemOneReading{
		request:     strings.Repeat("r", 20000),
		toolResults: []string{strings.Repeat("x", 20000)},
	}
	rule := config.SystemOneRule{Input: config.SystemOneInput{ToolResults: true}}
	req := reading.state(rule)
	if got := req.Parts["tool_results"]; len(got) != 20000 {
		t.Fatalf("default cap altered tool_results: %d", len(got))
	}
	if got := req.Parts["request"]; len(got) != 20000 {
		t.Fatalf("default cap altered request: %d", len(got))
	}
}
