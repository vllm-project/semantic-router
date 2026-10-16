package store

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
)

// ROUTER_REPLAY_TEST_POSTGRES_CONFIG is a JSON PostgresConfig for a disposable
// test database. Only a unique table owned by this test is altered or removed;
// an operator-supplied table_name is deliberately ignored.
// StorageIntegration: postgres
func TestPostgresMetadataIntegration(t *testing.T) {
	storagetest.Require(t, "postgres")
	raw := os.Getenv("ROUTER_REPLAY_TEST_POSTGRES_CONFIG")
	if raw == "" {
		storagetest.Unavailable(t, "postgres", "set ROUTER_REPLAY_TEST_POSTGRES_CONFIG to run against PostgreSQL")
	}
	var cfg PostgresConfig
	if err := json.Unmarshal([]byte(raw), &cfg); err != nil {
		t.Fatal("ROUTER_REPLAY_TEST_POSTGRES_CONFIG must be a JSON PostgresConfig")
	}
	cfg.TableName = fmt.Sprintf("replay_metadata_%d", time.Now().UnixNano())
	runtimeCfg, err := newPostgresRuntimeConfig(&cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	db, err := openConfiguredPostgresDB(ctx, runtimeCfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		//nolint:gosec // generated table name was validated above
		if _, dropErr := db.ExecContext(cleanupCtx, "DROP TABLE IF EXISTS "+cfg.TableName); dropErr != nil {
			t.Errorf("drop owned test table: %v", dropErr)
		}
		_ = db.Close()
	})

	initial, err := NewPostgresStore(&cfg, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = initial.Close() })
	legacy := postgresMetadataFixture()
	legacy.ID = "legacy"
	if _, addErr := initial.Add(ctx, legacy); addErr != nil {
		t.Fatal(addErr)
	}
	// Reproduce the schema before routing metadata existed, retaining its rows.
	//nolint:gosec // generated table name was validated above
	if _, dropErr := db.ExecContext(ctx, "ALTER TABLE "+cfg.TableName+" DROP COLUMN routing_metadata"); dropErr != nil {
		t.Fatal(dropErr)
	}
	if closeErr := initial.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}

	migrated, err := NewPostgresStore(&cfg, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = migrated.Close() })
	old, found, err := migrated.Get(ctx, legacy.ID)
	if err != nil || !found || old.ConfidenceScoreAvailable || old.SelectionMethod != "" || old.Recipe != legacy.Recipe {
		t.Fatalf("legacy row was lost or gained invented routing metadata: found=%v available=%v method=%q recipe=%q err=%v", found, old.ConfidenceScoreAvailable, old.SelectionMethod, old.Recipe, err)
	}
	input := postgresMetadataFixture()
	if id, addErr := migrated.Add(ctx, input); addErr != nil || id != input.ID {
		t.Fatalf("Add returned id=%q err=%v", id, addErr)
	}
	if closeErr := migrated.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}

	reopened, err := NewPostgresStore(&cfg, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reopened.Close() })
	got, found, err := reopened.Get(ctx, input.ID)
	if err != nil || !found {
		t.Fatalf("Get after reopen: found=%v err=%v", found, err)
	}
	assertPostgresMetadata(t, got, input)
	if got.ID != input.ID || !got.Timestamp.Equal(input.Timestamp) || got.Recipe != input.Recipe || got.SessionID != input.SessionID {
		t.Fatal("Add/Get must retain the supplied record identity and timestamp")
	}
	records, err := reopened.List(ctx)
	if err != nil || len(records) != 2 {
		t.Fatalf("List after migration: count=%d err=%v", len(records), err)
	}
	for _, record := range records {
		if record.ID == input.ID {
			assertPostgresMetadata(t, record, input)
		}
	}
}
