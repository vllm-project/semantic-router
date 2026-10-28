package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
)

// StorageIntegration: postgres
func TestPostgresQueryBeyondTenThousandIntegration(t *testing.T) {
	storagetest.Require(t, "postgres")
	raw := os.Getenv("ROUTER_REPLAY_TEST_POSTGRES_CONFIG")
	if raw == "" {
		storagetest.Unavailable(t, "postgres", "set ROUTER_REPLAY_TEST_POSTGRES_CONFIG to run against PostgreSQL")
	}
	var cfg PostgresConfig
	if err := json.Unmarshal([]byte(raw), &cfg); err != nil {
		t.Fatal("invalid PostgreSQL test config")
	}
	cfg.TableName = fmt.Sprintf("replay_query_%d", time.Now().UnixNano())
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
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	fixture := postgresMetadataFixture()
	fixture.ID, fixture.RequestID, fixture.Recipe = "record-00000", "benchmark_%_request", "evaluation"
	fixture.LifecycleState = LifecycleCompleted
	fixture.SessionID = "old-session"
	fixture.RequestBody, fixture.ResponseBody = strings.Repeat("request", 32), strings.Repeat("response", 32)
	fixture.ToolTrace = &ToolTrace{ToolNames: []string{"lookup"}, Steps: []ToolTraceStep{{ToolName: "calculate", Text: strings.Repeat("large result", 16)}}}
	if _, err = postgres.Add(ctx, fixture); err != nil {
		t.Fatal(err)
	}
	// The oldest matching rows lie beyond the former 10,000-row cutoff. Equal
	// timestamps also exercise stable ID ordering across page boundaries.
	//nolint:gosec // generated table name is validated by NewPostgresStore
	query := fmt.Sprintf(`INSERT INTO %[1]s SELECT copied.*
 FROM %[1]s seed CROSS JOIN generate_series(1,12249) i
 CROSS JOIN LATERAL jsonb_populate_record(NULL::%[1]s, to_jsonb(seed) || jsonb_build_object(
 'id', 'record-' || lpad(i::text,5,'0'), 'recipe', CASE WHEN i < 12230 THEN 'evaluation' ELSE 'other' END,
 'session_id', CASE WHEN i IN (1,12249) THEN 'old-session' ELSE 'unrelated-session' END)) copied
 WHERE seed.id='record-00000'`, cfg.TableName)
	if _, err = postgres.db.ExecContext(ctx, query); err != nil {
		t.Fatal(err)
	}
	filters := QueryFilters{Recipe: "evaluation", Search: "_%_", Model: fixture.SelectedModel}
	page, err := postgres.QueryPage(ctx, filters, 5, 12225, false)
	if err != nil {
		t.Fatal(err)
	}
	if page.Total != 12230 || page.Offset != 12225 || len(page.Records) != 5 || page.Records[0].ID != "record-00004" || page.Records[4].ID != fixture.ID {
		t.Fatalf("incorrect final page: total=%d offset=%d records=%v", page.Total, page.Offset, page.Records)
	}
	for _, record := range page.Records {
		if record.RequestBody != "" || record.ResponseBody != "" || record.ToolTrace == nil || len(record.ToolTrace.Steps) != 0 || strings.Join(record.ToolTrace.ToolNames, ",") != "calculate,lookup" {
			t.Fatal("summary query fetched captured bodies or lost tool names")
		}
	}
	past, err := postgres.QueryPage(ctx, filters, 5, 99999, false)
	if err != nil || len(past.Records) != 0 || past.Offset != 12230 || past.Total != 12230 {
		t.Fatalf("past-end page=%+v err=%v", past, err)
	}
	detail, err := postgres.QueryPage(ctx, filters, 1, 12229, true)
	if err != nil || detail.Records[0].RequestBody != fixture.RequestBody || len(detail.Records[0].ToolTrace.Steps) != 1 {
		t.Fatalf("detail page failed: %v", err)
	}
	count, matching := 0, 0
	err = postgres.ScanMetadata(ctx, func(record Record) error {
		count++
		if record.Recipe == "evaluation" {
			matching++
		}
		if record.RequestBody != "" || record.ResponseBody != "" || record.ToolTrace != nil || record.Prompt != "" {
			return errors.New("metadata scan fetched a captured payload")
		}
		return nil
	})
	if err != nil || count != 12250 || matching != 12230 {
		t.Fatalf("scan count=%d matching=%d err=%v", count, matching, err)
	}
	scoped, err := postgres.QueryPage(ctx, QueryFilters{SessionID: "old-session", Recipe: "evaluation", RecipeSet: true}, 100, 0, true)
	if err != nil || scoped.Total != 2 || len(scoped.Records) != 2 {
		t.Fatalf("old scoped session=%+v err=%v", scoped, err)
	}
	unscoped, err := postgres.QueryPage(ctx, QueryFilters{SessionID: "old-session"}, 100, 0, true)
	if err != nil || unscoped.Total != 3 {
		t.Fatalf("cross-recipe session total=%d err=%v", unscoped.Total, err)
	}
	//nolint:gosec // generated table name is validated by NewPostgresStore
	if _, err = postgres.db.ExecContext(ctx, "UPDATE "+cfg.TableName+" SET recipe=NULL WHERE id=$1", "record-12249"); err != nil {
		t.Fatal(err)
	}
	legacy, err := postgres.QueryPage(ctx, QueryFilters{SessionID: "old-session", RecipeSet: true}, 100, 0, true)
	if err != nil || legacy.Total != 1 || legacy.Records[0].ID != "record-12249" {
		t.Fatalf("legacy recipe session=%+v err=%v", legacy, err)
	}
	stopped := errors.New("stop scan")
	if err = postgres.ScanMetadata(ctx, func(Record) error { return stopped }); !errors.Is(err, stopped) {
		t.Fatalf("callback failure was hidden: %v", err)
	}
}
