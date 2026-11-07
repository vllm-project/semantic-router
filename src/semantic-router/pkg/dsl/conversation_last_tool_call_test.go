package dsl

import (
	"reflect"
	"strings"
	"testing"
)

// The last_tool_call source keeps its tools, calls and match through a
// compile, decompile and recompile.
func TestConversationLastToolCallSourceRoundTrips(t *testing.T) {
	input := `
SIGNAL conversation read_turn {
  feature: { type: "count", source: { type: "last_tool_call", calls: [{ tools: ["Read", "view_image"] }, { tools: ["Bash"], argument: "command", pattern: "^\\s*(ls|cat)\\b[^;&|<>\\n]*$" }, { tools: ["exec_command"], argument: "cmd", pattern: "^\\s*git (log|diff)\\b" }] } }
  predicate: { gte: 1 }
}
SIGNAL conversation other_turn {
  feature: { type: "count", source: { type: "last_tool_call", calls: [{ tools: ["Read"] }], match: "no_match" } }
  predicate: { gte: 1 }
}
SIGNAL conversation named_read {
  feature: { type: "count", source: { type: "last_tool_call", tools: ["Read", "ToolSearch"] } }
  predicate: { gte: 1 }
}
ROUTE r { PRIORITY 1 WHEN conversation("read_turn") MODEL "m:1b" }
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if len(cfg.ConversationRules) != 3 {
		t.Fatalf("got %d conversation rules, want 3", len(cfg.ConversationRules))
	}
	read := cfg.ConversationRules[0].Feature.Source
	if len(read.Calls) != 3 || read.Calls[1].Argument != "command" || read.Calls[1].Pattern != `^\s*(ls|cat)\b[^;&|<>\n]*$` {
		t.Fatalf("calls not compiled: %+v", read.Calls)
	}
	if cfg.ConversationRules[1].Feature.Source.Match != "no_match" {
		t.Fatalf("match not compiled: %+v", cfg.ConversationRules[1].Feature.Source)
	}
	text, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile: %v", err)
	}
	again, errs := Compile(text)
	if len(errs) > 0 {
		t.Fatalf("recompile errors: %v\n%s", errs, text)
	}
	if !reflect.DeepEqual(cfg.ConversationRules, again.ConversationRules) {
		t.Fatalf("round trip changed the rules:\nfirst  %+v\nsecond %+v\n%s", cfg.ConversationRules, again.ConversationRules, text)
	}
}

// A wrong matcher field name fails the compile instead of leaving a looser,
// name-only matcher behind.
func TestConversationLastToolCallRejectsUnknownMatcherFields(t *testing.T) {
	input := `
SIGNAL conversation read_turn {
  feature: { type: "count", source: { type: "last_tool_call", calls: [{ tools: ["Bash"], regex: "^ls$" }] } }
  predicate: { gte: 1 }
}
ROUTE r { PRIORITY 1 WHEN conversation("read_turn") MODEL "m:1b" }
`
	_, errs := Compile(input)
	if len(errs) == 0 || !strings.Contains(errs[0].Error(), "regex") {
		t.Fatalf("regex in place of pattern compiled: %v", errs)
	}
}
