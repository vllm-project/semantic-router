package memory

import (
	"strings"
	"testing"
)

// A user id is interpolated into a Milvus boolean-expression string literal.
// If it is not escaped, expression metacharacters in an attacker-influenced id
// break out of the quoted literal and inject filter logic (CWE-943), which on
// the read paths leaks other users' memories and on the delete paths can erase
// them. The id must be confined to a single escaped string literal.
func TestMilvusUserScopeFilterEscapesInjection(t *testing.T) {
	if got := milvusUserScopeFilter("alice"); got != `user_id == "alice"` {
		t.Fatalf("benign id: got %q, want %q", got, `user_id == "alice"`)
	}

	// Classic break-out attempt: close the literal, OR in a match-everything
	// clause, reopen the literal.
	malicious := `x" || user_id != "y`
	got := milvusUserScopeFilter(malicious)
	if strings.Contains(got, `" || user_id != "`) {
		t.Fatalf("injection not neutralized, operators escaped to expression level: %s", got)
	}
	if got != `user_id == "x\" || user_id != \"y"` {
		t.Fatalf("unexpected escaping: got %q", got)
	}

	// Backslash must also be escaped so it cannot escape the closing quote.
	if got := milvusUserScopeFilter(`a\b`); got != `user_id == "a\\b"` {
		t.Fatalf("backslash: got %q, want %q", got, `user_id == "a\\b"`)
	}
}
