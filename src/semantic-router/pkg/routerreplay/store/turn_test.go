package store

import (
	"context"
	"database/sql"
	"reflect"
	"strings"
	"testing"
	"time"
)

var turnColumns = map[string]string{"thread_id": "agent-thread", "turn_id": "agent-turn", "parent_turn_id": "parent-turn"}

// The stores that keep a record as JSON keep its agent turn with it.
func TestTurnRoundTripsThroughMemoryAndRedis(t *testing.T) {
	ctx := context.Background()
	for name, storage := range map[string]Storage{"memory": NewMemoryStore(10, 0), "redis": newMiniRedisStore(t)} {
		t.Run(name, func(t *testing.T) {
			id, err := storage.Add(ctx, Record{ID: "turn-" + name, ThreadID: "agent-thread", TurnID: "agent-turn", ParentTurnID: "parent-turn"})
			if err != nil {
				t.Fatal(err)
			}
			got, _, err := storage.Get(ctx, id)
			if err != nil || got.ThreadID != "agent-thread" || got.TurnID != "agent-turn" || got.ParentTurnID != "parent-turn" {
				t.Fatalf("turn = %q %q %q (%v)", got.ThreadID, got.TurnID, got.ParentTurnID, err)
			}
		})
	}
}

// A record keeps its agent turn in plain columns, and a record without one
// leaves them NULL.
func TestPostgresTurnColumnsRoundTrip(t *testing.T) {
	insert, err := newPostgresInsertRecord(Record{ID: "turn", ThreadID: "agent-thread", TurnID: "agent-turn", ParentTurnID: "parent-turn"})
	if err != nil {
		t.Fatal(err)
	}
	plain, err := newPostgresInsertRecord(Record{ID: "plain"})
	if err != nil {
		t.Fatal(err)
	}
	row := postgresRecordRow{signalsJSON: []byte("{}")}
	destinations := row.scanDestinations()
	for column, want := range turnColumns {
		if got := insert.args()[columnIndex(t, column)]; got != want {
			t.Fatalf("%s = %v, want %v", column, got, want)
		}
		if plain.args()[columnIndex(t, column)] != nil {
			t.Fatalf("a record without a turn stores %s", column)
		}
		destination, ok := destinations[columnIndex(t, column)].(*sql.NullString)
		if !ok {
			t.Fatalf("%s column scan misaligned", column)
		}
		*destination = sql.NullString{String: want, Valid: true}
	}
	output, err := row.decode()
	if err != nil || output.ThreadID != "agent-thread" || output.TurnID != "agent-turn" || output.ParentTurnID != "parent-turn" {
		t.Fatalf("decoded turn = %q %q %q (%v)", output.ThreadID, output.TurnID, output.ParentTurnID, err)
	}
}

func TestPostgresTurnFilterAndMigration(t *testing.T) {
	where, args := postgresReplayWhere(QueryFilters{SessionID: "s", TurnID: "agent-turn"})
	if !reflect.DeepEqual(args, []interface{}{"s", "agent-turn"}) || !strings.Contains(where, "session_id = $1") ||
		!strings.Contains(where, "(turn_id = $2 OR parent_turn_id = $2)") {
		t.Fatalf("where = %s args = %#v", where, args)
	}
	schema := postgresCreateTableQuery(DefaultPostgresTableName)
	for _, statement := range []string{
		"ADD COLUMN IF NOT EXISTS thread_id", "ADD COLUMN IF NOT EXISTS turn_id", "ADD COLUMN IF NOT EXISTS parent_turn_id",
		"idx_router_replay_records_turn_timestamp ON router_replay_records (turn_id, timestamp DESC)",
		"idx_router_replay_records_parent_turn_timestamp ON router_replay_records (parent_turn_id, timestamp DESC)",
	} {
		if !strings.Contains(schema, statement) {
			t.Fatalf("fresh and existing tables must get %q", statement)
		}
	}
}

// A turn's query holds its own requests and the review and subagent call it
// started, newest first, and the metadata scan the aggregate filters keeps
// both ids.
//
// StorageIntegration: postgres
func TestPostgresTurnQueriesIntegration(t *testing.T) {
	postgres := openPostgresReviewStore(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	start := time.Unix(1700000000, 0).UTC()
	records := []Record{
		{ID: "own-call", TurnID: "turn-1", ThreadID: "thread-1"},
		{ID: "review", TurnID: "review-turn", ThreadID: "review-thread", ParentTurnID: "turn-1"},
		{ID: "subagent", TurnID: "child-turn", ThreadID: "child-thread", ParentTurnID: "turn-1"},
		{ID: "next-turn", TurnID: "turn-2", ThreadID: "thread-1"},
		{ID: "no-turn"},
	}
	for index, record := range records {
		record.RequestID, record.Timestamp = record.ID, start.Add(time.Duration(index)*time.Second)
		if _, err := postgres.Add(ctx, record); err != nil {
			t.Fatal(err)
		}
	}
	page, err := postgres.QueryPage(ctx, QueryFilters{TurnID: "turn-1"}, 10, 0, true)
	if err != nil || page.Total != 3 {
		t.Fatalf("turn page = %+v (%v)", page, err)
	}
	if got := []string{page.Records[0].ID, page.Records[1].ID, page.Records[2].ID}; !reflect.DeepEqual(got, []string{"subagent", "review", "own-call"}) {
		t.Fatalf("turn records = %v", got)
	}
	if review := page.Records[1]; review.ThreadID != "review-thread" || review.TurnID != "review-turn" || review.ParentTurnID != "turn-1" {
		t.Fatalf("review turn = %+v", review)
	}
	scanned := map[string]Record{}
	if err = postgres.ScanMetadata(ctx, QueryFilters{}, func(record Record) error {
		scanned[record.RequestID] = record
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if scan := scanned["review"]; scan.TurnID != "review-turn" || scan.ParentTurnID != "turn-1" {
		t.Fatalf("metadata scan turn = %q %q", scan.TurnID, scan.ParentTurnID)
	}
}
