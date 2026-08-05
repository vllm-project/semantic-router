package memory

import "fmt"

// milvusUserScopeFilter returns a Milvus boolean-expression clause that matches
// a single user's records.
//
// The user id is rendered with %q so it is emitted as a properly escaped Milvus
// string literal. This prevents filter-expression injection (CWE-943) when the
// identifier contains expression metacharacters such as '"', '\\', '|' or '&':
// without escaping, a crafted id like `x" || user_id != "y` would break out of
// the quoted literal and inject filter logic, leaking other users' memories on
// the read/retrieve paths and deleting them on the forget paths.
//
// This mirrors the escaping the Valkey backend already applies
// (valkeyEscapeTagValue) and the %q convention used by buildTypeFilter for
// memory-type values.
func milvusUserScopeFilter(userID string) string {
	return fmt.Sprintf("user_id == %q", userID)
}
