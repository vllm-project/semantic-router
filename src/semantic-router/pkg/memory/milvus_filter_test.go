package memory

import (
	"strings"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
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

func TestRetrieveFilterExprProjectScope(t *testing.T) {
	without := retrieveFilterExpr("alice", "", nil)
	if without != `user_id == "alice"` {
		t.Fatalf("empty project: got %q", without)
	}

	withProject := retrieveFilterExpr("alice", "proj-1", []MemoryType{MemoryTypeSemantic})
	want := `user_id == "alice" && project_id == "proj-1" && (memory_type == "semantic")`
	if withProject != want {
		t.Fatalf("project scope: got %q, want %q", withProject, want)
	}

	malicious := `p" || user_id != "x`
	got := retrieveFilterExpr("alice", malicious, nil)
	if strings.Contains(got, `" || user_id != "`) {
		t.Fatalf("project id injection not neutralized: %s", got)
	}
	if !strings.Contains(got, `project_id == "p\" || user_id != \"x"`) {
		t.Fatalf("unexpected project escaping: %s", got)
	}
	if !strings.Contains(got, milvusEqString("project_id", malicious)) {
		t.Fatalf("project id did not use the shared escaping helper: %s", got)
	}
}

func TestStoredProjectScopeSeparatesExplicitDefault(t *testing.T) {
	unscoped := &Memory{ID: "unscoped", Content: "no project", UserID: "alice"}
	explicit := &Memory{ID: "named", Content: "default project", UserID: "alice", ProjectID: "default"}

	unscopedJSON, err := memoryMetadataJSON(unscoped)
	if err != nil {
		t.Fatalf("unscoped metadata: %v", err)
	}
	explicitJSON, err := memoryMetadataJSON(explicit)
	if err != nil {
		t.Fatalf("explicit metadata: %v", err)
	}

	unscopedRow := newMemoryRowColumns(unscoped, []float32{0.1}, string(unscopedJSON))
	explicitRow := newMemoryRowColumns(explicit, []float32{0.2}, string(explicitJSON))
	if got, want := varcharColumnValue(t, unscopedRow.projectID), ""; got != want {
		t.Fatalf("unscoped indexed project_id = %q, want empty", got)
	}
	if got := varcharColumnValue(t, explicitRow.projectID); got != "default" {
		t.Fatalf("explicit indexed project_id = %q, want default", got)
	}
	if strings.Contains(string(unscopedJSON), `"project_id":"default"`) {
		t.Fatalf("unscoped metadata was stored as project default: %s", unscopedJSON)
	}
	if !strings.Contains(string(explicitJSON), `"project_id":"default"`) {
		t.Fatalf("explicit metadata lost project default: %s", explicitJSON)
	}

	filter := retrieveFilterExpr("alice", "default", nil)
	want := `user_id == "alice" && project_id == "default"`
	if filter != want {
		t.Fatalf("default project filter = %q, want %q", filter, want)
	}
}

func TestRetainProjectMatchesKeepsExplicitDefaultOnly(t *testing.T) {
	candidates := []*RetrieveResult{
		{Memory: &Memory{ID: "unscoped", ProjectID: ""}, Score: 0.9},
		{Memory: &Memory{ID: "named", ProjectID: "default"}, Score: 0.8},
		{Memory: &Memory{ID: "other", ProjectID: "proj-1"}, Score: 0.7},
	}
	got := retainProjectMatches(candidates, "default")
	if len(got) != 1 || got[0].Memory.ID != "named" {
		t.Fatalf("kept %#v, want only the explicit default project", got)
	}
	if all := retainProjectMatches(candidates, ""); len(all) != len(candidates) {
		t.Fatalf("empty project filter kept %d, want %d", len(all), len(candidates))
	}
}

func varcharColumnValue(t *testing.T, col entity.Column) string {
	t.Helper()
	varchar, ok := col.(*entity.ColumnVarChar)
	if !ok {
		t.Fatalf("column %s is %T, want varchar", col.Name(), col)
	}
	value, err := varchar.ValueByIdx(0)
	if err != nil {
		t.Fatalf("read %s: %v", col.Name(), err)
	}
	return value
}
