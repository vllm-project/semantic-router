package headers

import (
	"regexp"
	"strings"
	"testing"
)

var traceKeyPattern = regexp.MustCompile(`^[a-z][a-z0-9_]*$`)

func TestStructuredItemsSerializeAsRFC8941(t *testing.T) {
	for _, tt := range []struct {
		name string
		item Item
		want string
	}{
		{"token", Token("multi_factor"), "multi_factor"},
		{"token with a path", Token("text/plain"), "text/plain"},
		{"no token is a string", Token("computer science"), `"computer science"`},
		{"leading digit is a string", Token("3d"), `"3d"`},
		{"string escapes quotes and backslashes", String(`a "b" \c`), `"a \"b\" \\c"`},
		{"string percent-encodes", String("50% ü\n"), `"50%25 %C3%BC%0A"`},
		{"integer", Integer(42), "42"},
		{"integer is clamped", Integer(1 << 62), "999999999999999"},
		{"decimal keeps three digits", Decimal(0.91234), "0.912"},
		{"whole decimal", Decimal(2), "2.0"},
		{"negative zero", Decimal(-0.0001), "0.0"},
		{"booleans", List(Boolean(true), Boolean(false)), "(?1 ?0)"},
		{"inner list with parameters", List(String("a"), Token("b").With("n", Integer(1))).With("ok", Boolean(true)), `("a" b;n=1);ok`},
		{"false parameter", Token("x").With("drop", Boolean(false)), "x;drop=?0"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.item.String(); got != tt.want {
				t.Fatalf("String() = %s, want %s", got, tt.want)
			}
		})
	}
}

func TestTraceMembersAreWellFormed(t *testing.T) {
	seen := make(map[string]bool)
	for _, member := range TraceMembers() {
		if !traceKeyPattern.MatchString(member.Key) || seen[member.Key] {
			t.Errorf("trace member %q is no unique RFC 8941 key", member.Key)
		}
		seen[member.Key] = true
		if !ownerPattern.MatchString(member.Owner) {
			t.Errorf("%s: owner %q is not a feature name", member.Key, member.Owner)
		}
		if !tableSafeSentence(member.Doc) || strings.Contains(member.Doc, "*/") {
			t.Errorf("%s: doc %q must be one sentence a Markdown table, MDX and a TypeScript comment can hold", member.Key, member.Doc)
		}
	}
	if members := TraceMembers(); members[len(members)-1].Key != TraceTruncated {
		t.Errorf("the last trace member is %s, want %s", members[len(members)-1].Key, TraceTruncated)
	}
}

func TestTraceEncodesMembersInPriorityOrder(t *testing.T) {
	var trace Trace
	if got := trace.Encode(); got != "" {
		t.Fatalf("empty trace = %q, want none", got)
	}
	trace.Set(TraceSignals, List(String("keyword:code")))
	trace.Set(TraceSystemPrompt, Boolean(true))
	trace.Set(TraceAlgorithm, Token("static"))
	trace.Set("not_a_member", Integer(1))
	trace.Set(TraceTruncated, Boolean(true))
	want := `algorithm=static, signals=("keyword:code"), system_prompt`
	if got := trace.Encode(); got != want {
		t.Fatalf("Encode() = %s, want %s", got, want)
	}
}

func TestTraceDropsLowestPriorityMembersOverTheLimit(t *testing.T) {
	var trace Trace
	trace.Set(TraceAlgorithm, Token("static"))
	trace.Set(TraceCategory, String(strings.Repeat("c", 100)))
	trace.Set(TraceSignals, String(strings.Repeat("s", TraceLimitBytes-40)))
	got := trace.Encode()
	if len(got) > TraceLimitBytes {
		t.Fatalf("encoded trace is %d bytes, over %d", len(got), TraceLimitBytes)
	}
	if !strings.HasPrefix(got, "algorithm=static, signals=") || !strings.HasSuffix(got, ", truncated") || strings.Contains(got, "category") {
		t.Fatalf("Encode() kept the wrong members: %.80s...%s", got, got[len(got)-30:])
	}

	var oversized Trace
	oversized.Set(TraceAlgorithm, String(strings.Repeat("a", TraceLimitBytes)))
	if got := oversized.Encode(); got != TraceTruncated {
		t.Fatalf("a trace whose only member exceeds the limit = %.40s, want %s", got, TraceTruncated)
	}
}
