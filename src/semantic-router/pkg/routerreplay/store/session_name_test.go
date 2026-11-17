package store

import (
	"context"
	"testing"
	"time"
)

// A record keeps the session name it was written with until its answer names
// the session, in the memory and Redis stores alike.
func TestSessionNameRoundTripsThroughMemoryAndRedis(t *testing.T) {
	ctx := context.Background()
	for name, storage := range map[string]Storage{"memory": NewMemoryStore(10, 0), "redis": newMiniRedisStore(t)} {
		t.Run(name, func(t *testing.T) {
			id, err := storage.Add(ctx, Record{ID: "title-" + name, Timestamp: time.Unix(1700000000, 0).UTC(), SessionName: "Renamed"})
			if err != nil {
				t.Fatal(err)
			}
			if got, _, _ := storage.Get(ctx, id); got.SessionName != "Renamed" {
				t.Fatalf("written session name = %q", got.SessionName)
			}
			if err = storage.UpdateSessionName(ctx, id, "Average helper"); err != nil {
				t.Fatal(err)
			}
			if got, _, _ := storage.Get(ctx, id); got.SessionName != "Average helper" {
				t.Fatalf("updated session name = %q", got.SessionName)
			}
			if err = storage.UpdateSessionName(ctx, "missing", "x"); err == nil {
				t.Fatal("a session name for an unknown record is an error")
			}
		})
	}
}

// PostgreSQL keeps the session name with the record's run identity, so a
// metadata scan reads it with the agent.
func TestPostgresKeepsTheSessionNameWithTheRunIdentity(t *testing.T) {
	input := Record{ID: "named", SessionName: "Average helper", Agent: &AgentIdentity{Harness: HarnessClaudeCode}}
	insert, err := newPostgresInsertRecord(input)
	if err != nil {
		t.Fatal(err)
	}
	row := postgresRecordRow{signalsJSON: []byte("{}"), routingMetadataJSON: insert.args()[columnIndex(t, "routing_metadata")].([]byte)}
	output, err := row.decode()
	if err != nil {
		t.Fatal(err)
	}
	if output.SessionName != "Average helper" || output.Agent == nil || output.Agent.Harness != HarnessClaudeCode {
		t.Fatalf("decoded session name %q, agent %+v", output.SessionName, output.Agent)
	}
}
