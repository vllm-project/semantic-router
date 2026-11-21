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

// A record routed to one model names the fallback candidate that answered,
// with the reasoning it was sent with, in the memory and Redis stores alike.
func TestServedModelReplacesTheRoutedModel(t *testing.T) {
	ctx := context.Background()
	for name, storage := range map[string]Storage{"memory": NewMemoryStore(10, 0), "redis": newMiniRedisStore(t)} {
		t.Run(name, func(t *testing.T) {
			id, err := storage.Add(ctx, Record{
				ID: "served-" + name, Timestamp: time.Unix(1700000000, 0).UTC(),
				SelectedModel: "fail-model", ReasoningMode: "on", ReasoningEffort: "high",
			})
			if err != nil {
				t.Fatal(err)
			}
			if err = storage.UpdateServedModel(ctx, id, ServedModel{Model: "cheap-model", ReasoningMode: "off"}); err != nil {
				t.Fatal(err)
			}
			got, _, _ := storage.Get(ctx, id)
			if got.SelectedModel != "cheap-model" || got.ReasoningMode != "off" || got.ReasoningEffort != "" {
				t.Fatalf("served model = %q, reasoning %q %q", got.SelectedModel, got.ReasoningMode, got.ReasoningEffort)
			}
			if err = storage.UpdateServedModel(ctx, "missing", ServedModel{Model: "x"}); err == nil {
				t.Fatal("a served model for an unknown record is an error")
			}
		})
	}
}

// PostgreSQL keeps the model and the mode in columns and the effort in the
// routing metadata, and a served model replaces all three.
// StorageIntegration: postgres
func TestPostgresServedModelIntegration(t *testing.T) {
	storagetest.Require(t, "postgres")
	raw := os.Getenv("ROUTER_REPLAY_TEST_POSTGRES_CONFIG")
	if raw == "" {
		storagetest.Unavailable(t, "postgres", "set ROUTER_REPLAY_TEST_POSTGRES_CONFIG to run against PostgreSQL")
	}
	var cfg PostgresConfig
	if err := json.Unmarshal([]byte(raw), &cfg); err != nil {
		t.Fatal("ROUTER_REPLAY_TEST_POSTGRES_CONFIG must be a JSON PostgresConfig")
	}
	cfg.TableName = fmt.Sprintf("replay_served_%d", time.Now().UnixNano())
	storage, err := NewPostgresStore(&cfg, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	t.Cleanup(func() {
		//nolint:gosec // the generated table name was validated by NewPostgresStore
		if _, dropErr := storage.db.ExecContext(ctx, "DROP TABLE IF EXISTS "+cfg.TableName); dropErr != nil {
			t.Errorf("drop owned test table: %v", dropErr)
		}
		_ = storage.Close()
	})
	id, err := storage.Add(ctx, Record{
		ID: "served", Timestamp: time.Unix(1700000000, 0).UTC(), SessionName: "kept",
		SelectedModel: "fail-model", ReasoningMode: "off",
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, served := range []ServedModel{
		{Model: "cheap-model", ReasoningMode: "on", ReasoningEffort: "low"},
		{Model: "main-model", ReasoningMode: "off"},
	} {
		if err = storage.UpdateServedModel(ctx, id, served); err != nil {
			t.Fatal(err)
		}
		got, _, getErr := storage.Get(ctx, id)
		if getErr != nil || got.SelectedModel != served.Model || got.ReasoningMode != served.ReasoningMode ||
			got.ReasoningEffort != served.ReasoningEffort || got.SessionName != "kept" {
			t.Fatalf("served model = %q, reasoning %q %q, session name %q, err %v",
				got.SelectedModel, got.ReasoningMode, got.ReasoningEffort, got.SessionName, getErr)
		}
	}
	if err = storage.UpdateServedModel(ctx, "missing", ServedModel{Model: "x"}); err == nil {
		t.Fatal("a served model for an unknown record is an error")
	}
}
