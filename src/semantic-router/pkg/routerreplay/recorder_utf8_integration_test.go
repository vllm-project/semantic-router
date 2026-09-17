package routerreplay

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestRecorderUTF8PostgresIntegration(t *testing.T) {
	raw := os.Getenv("ROUTER_REPLAY_TEST_POSTGRES_CONFIG")
	if raw == "" {
		t.Skip("set ROUTER_REPLAY_TEST_POSTGRES_CONFIG to run against PostgreSQL")
	}
	var cfg store.PostgresConfig
	if err := json.Unmarshal([]byte(raw), &cfg); err != nil {
		t.Fatal("ROUTER_REPLAY_TEST_POSTGRES_CONFIG must be a JSON PostgresConfig")
	}
	// Ignore any supplied table name; only this test-owned table is modified.
	cfg.TableName = fmt.Sprintf("replay_utf8_%d", time.Now().UnixNano())
	runtimeCfg, err := postgres.NewRuntimeConfig(&cfg, store.DefaultPostgresTableName)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	db, err := postgres.OpenDB(ctx, runtimeCfg)
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
	backend, err := store.NewPostgresStore(&cfg, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	recorder := NewRecorder(backend)
	t.Cleanup(func() { _ = recorder.Close() })
	exerciseRecorderUTF8Boundaries(t, recorder)
}
