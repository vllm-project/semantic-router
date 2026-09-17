package store

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"
)

func postgresMetadataFixture() Record {
	return Record{
		ID: "metadata-round-trip", Timestamp: time.Unix(1700000000, 0).UTC(),
		Recipe: "balance", SessionID: "metadata-session", LifecycleState: LifecycleCompleted,
		SelectionMethod: "multi_factor", ConfidenceScore: 0.83, ConfidenceScoreAvailable: true,
		FromCache: true, CacheHitKind: "semantic", CacheSource: "redis",
		CacheEntryAgeSeconds: 12.25, CacheTTLSeconds: 87,
		PIIEnabled: true, PIIDetected: true, PIIEntities: []string{"email", "phone"}, PIIBlocked: true,
	}
}

func assertPostgresMetadata(t *testing.T, got, want Record) {
	t.Helper()
	if got.ConfidenceScore != want.ConfidenceScore || got.ConfidenceScoreAvailable != want.ConfidenceScoreAvailable || got.SelectionMethod != want.SelectionMethod {
		t.Fatalf("routing metadata changed: score=%v available=%v method=%q", got.ConfidenceScore, got.ConfidenceScoreAvailable, got.SelectionMethod)
	}
	if got.CacheHitKind != want.CacheHitKind || got.CacheSource != want.CacheSource || got.CacheEntryAgeSeconds != want.CacheEntryAgeSeconds || got.CacheTTLSeconds != want.CacheTTLSeconds {
		t.Fatalf("cache metadata changed: kind=%q source=%q age=%v ttl=%v", got.CacheHitKind, got.CacheSource, got.CacheEntryAgeSeconds, got.CacheTTLSeconds)
	}
	if got.PIIDetected != want.PIIDetected || got.PIIBlocked != want.PIIBlocked || !reflect.DeepEqual(got.PIIEntities, want.PIIEntities) {
		t.Fatalf("PII metadata changed: detected=%v entities=%v blocked=%v", got.PIIDetected, got.PIIEntities, got.PIIBlocked)
	}
}

func TestPostgresMetadataRowRoundTrip(t *testing.T) {
	for _, test := range []struct {
		name      string
		score     float64
		available bool
	}{
		{"measured nonzero", 0.83, true},
		{"measured zero", 0, true},
		{"unavailable", 0, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			input := postgresMetadataFixture()
			input.ConfidenceScore, input.ConfidenceScoreAvailable = test.score, test.available
			insert, err := newPostgresInsertRecord(input)
			if err != nil {
				t.Fatal(err)
			}
			row := postgresRecordRow{signalsJSON: []byte("{}")}
			for _, column := range []string{"routing_metadata", "safety_evidence"} {
				index := columnIndex(t, column)
				destination, ok := row.scanDestinations()[index].(*[]byte)
				if !ok {
					t.Fatalf("%s scan destination is not JSON", column)
				}
				*destination = insert.args()[index].([]byte)
			}
			output, err := row.decode()
			if err != nil {
				t.Fatal(err)
			}
			assertPostgresMetadata(t, output, input)
		})
	}
}

func TestPostgresLegacyMissingConfidenceRemainsUnavailable(t *testing.T) {
	for _, encoded := range [][]byte{nil, []byte("null"), []byte("{}"), []byte(`{"confidence_score":null}`)} {
		row := postgresRecordRow{
			signalsJSON: []byte("{}"), routingMetadataJSON: encoded,
			safetyEvidenceJSON: []byte(`{"confidence_score_available":true}`),
		}
		record, err := row.decode()
		if err != nil {
			t.Fatal(err)
		}
		if record.ConfidenceScoreAvailable || record.SelectionMethod != "" {
			t.Fatal("legacy metadata must stay unknown when no value was stored")
		}
		data, err := json.Marshal(record)
		if err != nil || !strings.Contains(string(data), `"confidence_score":null`) {
			t.Fatalf("missing confidence must be JSON null: %s (%v)", data, err)
		}
	}
}

func TestPostgresRoutingMetadataMigrationIsAdditive(t *testing.T) {
	query := postgresCreateTableQuery(DefaultPostgresTableName)
	if !strings.Contains(query, "routing_metadata JSONB,") || !strings.Contains(query, "ADD COLUMN IF NOT EXISTS routing_metadata JSONB;") {
		t.Fatal("fresh and existing replay tables must gain nullable routing metadata")
	}
}
