package store

import (
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestPostgresReplayWhereParameters(t *testing.T) {
	// since in another zone: the timestamp column holds UTC wall-clock time.
	since := time.Date(2026, 11, 16, 0, 0, 0, 0, time.FixedZone("KST", 9*3600))
	filters := QueryFilters{Recipe: "recipe'", Decision: "decision", SessionID: "session", Model: "model", Search: "_%_'", CacheStatus: "cached", Since: since}
	where, args := postgresReplayWhere(filters)
	if strings.Contains(where, filters.Recipe) || strings.Contains(where, filters.Search) || strings.Contains(where, " LIKE ") {
		t.Fatalf("filters must remain literal parameters: %s", where)
	}
	if !reflect.DeepEqual(args, []interface{}{filters.Recipe, filters.Decision, filters.SessionID, filters.Model, filters.Search, since.UTC()}) {
		t.Fatalf("unexpected args: %#v", args)
	}
	for _, expression := range []string{"COALESCE(recipe, '') = $1", "decision = $2", "session_id = $3", "selected_model = $4 OR original_model = $4", "strpos(lower(COALESCE(session_id, '')), lower($5))", "timestamp >= $6", "from_cache = TRUE"} {
		if !strings.Contains(where, expression) {
			t.Fatalf("missing %s in %s", expression, where)
		}
	}
}

// A harness filter matches the harness in a record's routing metadata, each
// harness a parameter, and an API client as a record without one.
func TestPostgresReplayWhereHarness(t *testing.T) {
	where, args := postgresReplayWhere(QueryFilters{SessionID: "session", Harness: HarnessAPIClient + "," + HarnessCodex})
	if !reflect.DeepEqual(args, []interface{}{"session", "", HarnessCodex}) {
		t.Fatalf("unexpected args: %#v", args)
	}
	clause := "COALESCE(routing_metadata->'agent'->>'harness', '') IN ($2, $3)"
	if !strings.Contains(where, "session_id = $1") || !strings.Contains(where, clause) {
		t.Fatalf("missing %s in %s", clause, where)
	}
	if where, _ := postgresReplayWhere(QueryFilters{SessionID: "session"}); strings.Contains(where, "harness") {
		t.Fatalf("no harness filter must leave the harness alone: %s", where)
	}
}

func TestFilterHarnessNamesPlainClients(t *testing.T) {
	for _, test := range []struct {
		agent *AgentIdentity
		want  string
	}{
		{nil, HarnessAPIClient},
		{&AgentIdentity{ID: "agent-hash"}, HarnessAPIClient},
		{&AgentIdentity{Harness: HarnessClaudeCode, ID: "agent-hash"}, HarnessClaudeCode},
	} {
		if got := FilterHarness(Record{Agent: test.agent}); got != test.want {
			t.Fatalf("FilterHarness(%+v) = %q, want %q", test.agent, got, test.want)
		}
	}
	list := HarnessAPIClient + "," + HarnessCodex
	if !HarnessListed(list, HarnessCodex) || !HarnessListed(list, HarnessAPIClient) || HarnessListed(list, HarnessClaudeCode) || HarnessListed("", HarnessAPIClient) {
		t.Fatalf("HarnessListed misreads %q", list)
	}
}
