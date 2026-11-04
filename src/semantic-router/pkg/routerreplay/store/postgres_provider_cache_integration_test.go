package store

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
)

// StorageIntegration: postgres
func TestPostgresProviderCacheIntegration(t *testing.T) {
	storagetest.Require(t, "postgres")
	raw := os.Getenv("ROUTER_REPLAY_TEST_POSTGRES_CONFIG")
	if raw == "" {
		storagetest.Unavailable(t, "postgres", "set ROUTER_REPLAY_TEST_POSTGRES_CONFIG to run against PostgreSQL")
	}
	var cfg PostgresConfig
	if err := json.Unmarshal([]byte(raw), &cfg); err != nil {
		t.Fatal("invalid PostgreSQL test config")
	}
	cfg.TableName = fmt.Sprintf("replay_provider_cache_%d", time.Now().UnixNano())
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	initial, err := NewPostgresStore(&cfg, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = initial.Close() })
	legacy := postgresMetadataFixture()
	legacy.ID = "legacy"
	if _, err = initial.Add(ctx, legacy); err != nil {
		t.Fatal(err)
	}
	// Reproduce a table created before provider_cache existed, retaining its rows.
	//nolint:gosec // generated table name is validated by NewPostgresStore
	if _, err = initial.db.ExecContext(ctx, "ALTER TABLE "+cfg.TableName+" DROP COLUMN provider_cache"); err != nil {
		t.Fatal(err)
	}
	if err = initial.Close(); err != nil {
		t.Fatal(err)
	}

	postgres, err := NewPostgresStore(&cfg, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		//nolint:gosec // generated table name is validated by NewPostgresStore
		if _, dropErr := postgres.db.Exec("DROP TABLE IF EXISTS " + cfg.TableName); dropErr != nil {
			t.Error(dropErr)
		}
		_ = postgres.Close()
	})
	if old, found, getErr := postgres.Get(ctx, legacy.ID); getErr != nil || !found || old.ProviderCache != nil {
		t.Fatalf("legacy row was lost or gained provider cache accounting: found=%v err=%v", found, getErr)
	}

	assertStored := func(id string, want *ProviderCacheAccounting) {
		t.Helper()
		got, found, getErr := postgres.Get(ctx, id)
		if getErr != nil || !found {
			t.Fatalf("Get %s: found=%v err=%v", id, found, getErr)
		}
		if !reflect.DeepEqual(got.ProviderCache, want) {
			t.Fatalf("provider cache accounting changed in PostgreSQL: got %+v want %+v", got.ProviderCache, want)
		}
	}

	zero, ten, eighty, two := 0, 10, 80, 2
	partial := &ProviderCacheAccounting{
		State:     "partial",
		CacheRead: TokenCostComponent{Tokens: &zero, Provenance: "authoritative"},
		Output:    TokenCostComponent{Tokens: &two, Provenance: "authoritative"},
	}
	record := postgresMetadataFixture()
	record.ID, record.RequestID = "provider-cache", "provider-cache-request"
	record.ProviderCache = partial
	if _, err = postgres.Add(ctx, record); err != nil {
		t.Fatal(err)
	}
	assertStored(record.ID, partial)

	writeCost, noCache, savings := 0.00052, 0.00022, -0.00035
	complete := &ProviderCacheAccounting{
		State:         "complete",
		UncachedInput: TokenCostComponent{Tokens: &ten, Provenance: "authoritative"},
		CacheRead:     TokenCostComponent{Tokens: &ten, Provenance: "authoritative"},
		CacheWrite:    TokenCostComponent{Tokens: &eighty, Provenance: "authoritative", Cost: &writeCost},
		Output:        TokenCostComponent{Tokens: &two, Provenance: "authoritative"},
		NoCacheCost:   &noCache,
		CacheSavings:  &savings,
	}
	if err = postgres.UpdateUsageCost(ctx, record.ID, UsageCost{ProviderCache: complete}); err != nil {
		t.Fatal(err)
	}
	assertStored(record.ID, complete)

	page, err := postgres.QueryPage(ctx, QueryFilters{SessionID: record.SessionID}, 10, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	listed := 0
	for _, summary := range page.Records {
		if summary.ID == record.ID {
			listed++
			if !reflect.DeepEqual(summary.ProviderCache, complete) {
				t.Fatalf("summary list lost provider cache accounting: %+v", summary.ProviderCache)
			}
		}
	}
	if listed != 1 {
		t.Fatalf("summary list returned the record %d times", listed)
	}
	scanned := 0
	if err = postgres.ScanMetadata(ctx, func(metadata Record) error {
		switch metadata.RequestID {
		case record.RequestID:
			scanned++
			if !reflect.DeepEqual(metadata.ProviderCache, complete) {
				t.Errorf("metadata scan lost provider cache accounting: %+v", metadata.ProviderCache)
			}
		default:
			if metadata.ProviderCache != nil {
				t.Errorf("metadata scan invented provider cache accounting: %+v", metadata.ProviderCache)
			}
		}
		return nil
	}); err != nil || scanned != 1 {
		t.Fatalf("metadata scan returned the record %d times: %v", scanned, err)
	}

	if err = postgres.UpdateUsageCost(ctx, record.ID, UsageCost{}); err != nil {
		t.Fatal(err)
	}
	assertStored(record.ID, nil)
}
