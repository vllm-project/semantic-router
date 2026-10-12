package memory

import "fmt"

// milvusEqString returns a Milvus boolean-expression clause that matches one
// string field. Every interpolated scope value goes through this helper so
// user ids and project ids share one escaping path.
//
// The value is rendered with %q so it is emitted as a properly escaped Milvus
// string literal. This prevents filter-expression injection (CWE-943) when the
// identifier contains expression metacharacters such as '"', '\\', '|' or '&':
// without escaping, a crafted id like `x" || user_id != "y` would break out of
// the quoted literal and inject filter logic, leaking other users' memories on
// the read/retrieve paths and deleting them on the forget paths.
//
// This mirrors the escaping the Valkey backend already applies
// (valkeyEscapeTagValue) and the %q convention used by buildTypeFilter for
// memory-type values.
func milvusEqString(field, value string) string {
	return fmt.Sprintf("%s == %q", field, value)
}

func milvusUserScopeFilter(userID string) string {
	return milvusEqString("user_id", userID)
}
