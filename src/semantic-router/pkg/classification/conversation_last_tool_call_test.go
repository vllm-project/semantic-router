package classification

import (
	"fmt"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestLastToolCallCountsListedNames(t *testing.T) {
	facts := ConversationFacts{LastAssistantToolNames: []string{"Read", "Grep"}}
	feature := func(tools ...string) config.ConversationFeature {
		return config.ConversationFeature{Type: "count", Source: config.ConversationSource{Type: "last_tool_call", Tools: tools}}
	}
	for _, tc := range []struct {
		tools []string
		want  float64
	}{
		{[]string{"Read", "Grep", "Glob"}, 2},
		{[]string{"Edit", "Bash"}, 0},
		{nil, 2},
	} {
		if got := resolveConversationValue(feature(tc.tools...), facts); got != tc.want {
			t.Fatalf("tools %v: got %v, want %v", tc.tools, got, tc.want)
		}
	}
	if got := resolveConversationValue(feature(), ConversationFacts{}); got != 0 {
		t.Fatalf("no assistant tool calls: got %v, want 0", got)
	}
}

func TestLastToolCallToolsNeedTheirSource(t *testing.T) {
	rule := config.ConversationRule{Name: "x", Feature: config.ConversationFeature{
		Type: "count", Source: config.ConversationSource{Type: "assistant_tool_call", Tools: []string{"Read"}},
	}, Predicate: &config.NumericPredicate{GTE: ptrFloat(1)}}
	if err := config.ValidateConversationRuleContract(rule); err == nil {
		t.Fatal("tools on assistant_tool_call should be rejected")
	}
	rule.Feature.Source.Type = "last_tool_call"
	if err := config.ValidateConversationRuleContract(rule); err != nil {
		t.Fatalf("last_tool_call with tools: %v", err)
	}
}

// A read-only turn is a Read call or a shell call whose command only reads;
// match no_match counts everything else, so "only reads" is reads >= 1 and others == 0.
func TestLastToolCallMatchesCommandArguments(t *testing.T) {
	const readOnly = "^[ \\t]*(ls|cat|head|tail|wc|file|stat|rg|grep|tree|pwd|git (status|log|show))([ \\t][^;&|<>\\n`$]*)?$"
	reads := []config.ToolCallMatch{
		{Tools: []string{"Read", "view_image"}},
		{Tools: []string{"Bash"}, Argument: "command", Pattern: readOnly},
		{Tools: []string{"exec_command"}, Argument: "cmd", Pattern: readOnly},
	}
	count := func(match string, names []string, args []string) float64 {
		feature := config.ConversationFeature{Type: "count", Source: config.ConversationSource{Type: "last_tool_call", Calls: reads, Match: match}}
		return resolveConversationValue(feature, ConversationFacts{LastAssistantToolNames: names, LastAssistantToolArgs: args})
	}
	for _, tc := range []struct {
		name         string
		names, args  []string
		reads, other float64
	}{
		{"claude read and ls", []string{"Read", "Bash"}, []string{`{"file_path":"/a"}`, `{"description":"x","command":"ls -la /app"}`}, 2, 0},
		{"claude ls then rm", []string{"Bash"}, []string{`{"command":"ls && rm -rf /tmp/x"}`}, 0, 1},
		{"claude multi-line command", []string{"Bash"}, []string{`{"command":"cat a\nrm b"}`}, 0, 1},
		{"codex git log", []string{"exec_command"}, []string{`{"cmd":"git log -3"}`}, 1, 0},
		{"newline after ls", []string{"Bash"}, []string{`{"command":"ls\nrm x"}`}, 0, 1},
		{"command name prefix", []string{"Bash"}, []string{`{"command":"cat-helper a"}`}, 0, 1},
		{"substitution", []string{"Bash"}, []string{`{"command":"cat $(ls)"}`}, 0, 1},
		{"redirection", []string{"Bash"}, []string{`{"command":"cat a > b"}`}, 0, 1},
		{"git diff can write", []string{"Bash"}, []string{`{"command":"git diff --output=x"}`}, 0, 1},
		{"codex writes", []string{"exec_command"}, []string{`{"cmd":"python3 fix.py > out.txt"}`}, 0, 1},
		{"command under another key", []string{"exec_command"}, []string{`{"command":"ls"}`}, 0, 1},
		{"arguments not JSON", []string{"Bash"}, []string{`ls`}, 0, 1},
		{"edit", []string{"Edit"}, []string{`{"file_path":"/a"}`}, 0, 1},
	} {
		if got := count("", tc.names, tc.args); got != tc.reads {
			t.Errorf("%s: reads = %v, want %v", tc.name, got, tc.reads)
		}
		if got := count("no_match", tc.names, tc.args); got != tc.other {
			t.Errorf("%s: others = %v, want %v", tc.name, got, tc.other)
		}
	}
}

func TestLastToolCallMatchersAreValidated(t *testing.T) {
	rule := func(source config.ConversationSource) config.ConversationRule {
		source.Type = "last_tool_call"
		return config.ConversationRule{Name: "x", Feature: config.ConversationFeature{Type: "count", Source: source}, Predicate: &config.NumericPredicate{GTE: ptrFloat(1)}}
	}
	for name, source := range map[string]config.ConversationSource{
		"argument without pattern": {Calls: []config.ToolCallMatch{{Tools: []string{"Bash"}, Argument: "command"}}},
		"bad pattern":              {Calls: []config.ToolCallMatch{{Pattern: "("}}},
		"unknown match":            {Calls: []config.ToolCallMatch{{Tools: []string{"Read"}}}, Match: "some"},
		"tools and calls":          {Tools: []string{"Read"}, Calls: []config.ToolCallMatch{{Tools: []string{"Bash"}}}},
	} {
		if err := config.ValidateConversationRuleContract(rule(source)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	other := rule(config.ConversationSource{Match: "no_match"})
	other.Feature.Source.Type = "assistant_tool_call"
	if err := config.ValidateConversationRuleContract(other); err == nil {
		t.Error("match on assistant_tool_call accepted")
	}
	ok := rule(config.ConversationSource{Calls: []config.ToolCallMatch{{Tools: []string{"Bash"}, Argument: "command", Pattern: "^ls"}}, Match: "no_match"})
	if err := config.ValidateConversationRuleContract(ok); err != nil {
		t.Errorf("valid calls rejected: %v", err)
	}
}

// A field the call does not carry as a string never matches, not even a pattern
// that matches the empty string, so an unreadable call is never taken for a read.
func TestLastToolCallUnreadableArgumentsNeverMatch(t *testing.T) {
	for _, pattern := range []string{"^$", ".*"} {
		source := config.ConversationSource{Type: "last_tool_call", Calls: []config.ToolCallMatch{{Tools: []string{"Bash"}, Argument: "command", Pattern: pattern}}}
		for name, args := range map[string]string{
			"null":        `{"command":null}`,
			"number":      `{"command":3}`,
			"missing key": `{"cmd":"ls"}`,
			"not JSON":    `ls`,
			"JSON array":  `["ls"]`,
		} {
			facts := ConversationFacts{LastAssistantToolNames: []string{"Bash"}, LastAssistantToolArgs: []string{args}}
			if got := resolveConversationValue(config.ConversationFeature{Type: "count", Source: source}, facts); got != 0 {
				t.Errorf("pattern %q, %s: counted %v", pattern, name, got)
			}
		}
	}
}

// The compiled-pattern cache stops growing at its bound, even when requests
// compile new patterns at the same time, and still matches.
func TestToolArgumentPatternCacheIsBounded(t *testing.T) {
	var wg sync.WaitGroup
	for i := 0; i < maxToolArgumentPatterns+50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := toolArgumentPattern(fmt.Sprintf("^bounded-%d$", i)); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if n := len(toolArgumentPatterns.compiled); n != maxToolArgumentPatterns {
		t.Fatalf("cache holds %d patterns, bound is %d", n, maxToolArgumentPatterns)
	}
	pattern, err := toolArgumentPattern("^bounded-300$")
	if err != nil || !pattern.MatchString("bounded-300") {
		t.Fatalf("an uncached pattern must still compile and match: %v", err)
	}
}
