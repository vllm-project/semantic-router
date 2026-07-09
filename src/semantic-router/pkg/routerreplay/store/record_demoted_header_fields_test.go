package store

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestRecordJSONExposesDemotedHeaderFields locks the replay wire contract for
// the two v0.4 demoted-header values (#2200, #2254). The replay API marshals
// the full Record, so these exact snake_case keys are what GET
// /v1/router_replay/{id} and the dashboard replay view read. Asserting the raw
// JSON keys (not just a round trip) guards against a tag rename silently
// breaking that contract.
func TestRecordJSONExposesDemotedHeaderFields(t *testing.T) {
	rec := Record{
		ID:                "rid",
		Signals:           Signal{},
		CacheSimilarity:   0.875, // exactly representable as float32
		ContextTokenCount: 4096,
	}

	data, err := json.Marshal(rec)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	js := string(data)

	if !strings.Contains(js, `"cache_similarity":0.875`) {
		t.Fatalf("expected cache_similarity key in replay JSON, got %s", js)
	}
	if !strings.Contains(js, `"context_token_count":4096`) {
		t.Fatalf("expected context_token_count key in replay JSON, got %s", js)
	}

	decoded := roundTripRecord(t, rec)
	if decoded.CacheSimilarity != 0.875 {
		t.Fatalf("cache_similarity round-trip = %v", decoded.CacheSimilarity)
	}
	if decoded.ContextTokenCount != 4096 {
		t.Fatalf("context_token_count round-trip = %d", decoded.ContextTokenCount)
	}
}

// TestRecordJSONOmitsDemotedHeaderFieldsWhenZero confirms the omitempty tags
// keep the replay JSON clean for requests with no cache lookup / no context
// token count, matching the surrounding RAG/usage fields.
func TestRecordJSONOmitsDemotedHeaderFieldsWhenZero(t *testing.T) {
	data, err := json.Marshal(Record{ID: "rid", Signals: Signal{}})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	js := string(data)

	if strings.Contains(js, "cache_similarity") {
		t.Fatalf("expected cache_similarity omitted when zero, got %s", js)
	}
	if strings.Contains(js, "context_token_count") {
		t.Fatalf("expected context_token_count omitted when zero, got %s", js)
	}
}

// TestPostgresBackendWiresDemotedHeaderFields guards the Postgres column wiring
// for the two demoted-header values (#2254). Unlike the JSON backends
// (memory/redis/milvus/qdrant) which marshal the whole Record, Postgres maps
// columns explicitly — so a struct field with no matching column in the
// CREATE/INSERT/SELECT/scan chain is silently dropped under
// store_backend: postgres (the documented default). This pins every link.
func TestPostgresBackendWiresDemotedHeaderFields(t *testing.T) {
	for _, col := range []string{"cache_similarity", "context_token_count"} {
		if !strings.Contains(postgresCreateTableQueryTemplate, col) {
			t.Errorf("CREATE TABLE template missing %q — column never provisioned", col)
		}
		if !strings.Contains(postgresInsertQueryTemplate, col) {
			t.Errorf("INSERT template missing %q — value never written under store_backend: postgres", col)
		}
		if !strings.Contains(postgresRecordSelectColumns, col) {
			t.Errorf("SELECT column list missing %q — value never read back under store_backend: postgres", col)
		}
	}

	// SELECT column list and scan destinations must stay 1:1; otherwise every
	// scanned column shifts. The insert-side alignment test does not cover this.
	selectCols := splitSQLColumnList(postgresRecordSelectColumns)
	scanDests := (&postgresRecordRow{}).scanDestinations()
	if len(selectCols) != len(scanDests) {
		t.Fatalf("SELECT columns (%d) vs scan destinations (%d) count mismatch", len(selectCols), len(scanDests))
	}
}

func splitSQLColumnList(list string) []string {
	parts := strings.Split(list, ",")
	cols := make([]string, 0, len(parts))
	for _, p := range parts {
		if trimmed := strings.TrimSpace(p); trimmed != "" {
			cols = append(cols, trimmed)
		}
	}
	return cols
}
