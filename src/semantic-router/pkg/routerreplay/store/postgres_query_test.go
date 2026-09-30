package store

import (
	"reflect"
	"strings"
	"testing"
)

func TestPostgresReplayWhereParameters(t *testing.T) {
	filters := QueryFilters{Recipe: "recipe'", Decision: "decision", SessionID: "session", Model: "model", Search: "_%_'", CacheStatus: "cached"}
	where, args := postgresReplayWhere(filters)
	if strings.Contains(where, filters.Recipe) || strings.Contains(where, filters.Search) || strings.Contains(where, " LIKE ") {
		t.Fatalf("filters must remain literal parameters: %s", where)
	}
	if !reflect.DeepEqual(args, []interface{}{filters.Recipe, filters.Decision, filters.SessionID, filters.Model, filters.Search}) {
		t.Fatalf("unexpected args: %#v", args)
	}
	for _, expression := range []string{"COALESCE(recipe, '') = $1", "decision = $2", "session_id = $3", "selected_model = $4 OR original_model = $4", "lower($5)", "from_cache = TRUE"} {
		if !strings.Contains(where, expression) {
			t.Fatalf("missing %s in %s", expression, where)
		}
	}
}
