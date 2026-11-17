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
