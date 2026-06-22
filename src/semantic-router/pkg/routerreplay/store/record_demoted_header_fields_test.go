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
