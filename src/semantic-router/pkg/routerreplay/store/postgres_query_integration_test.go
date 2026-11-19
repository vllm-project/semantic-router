package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
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
	err = postgres.ScanMetadata(ctx, QueryFilters{}, func(record Record) error {
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
	// A filtered scan reads only the selected rows; the filter options still
	// come from every row.
	filtered := 0
	err = postgres.ScanMetadata(ctx, QueryFilters{Recipe: "other", SessionID: "unrelated-session"}, func(record Record) error {
		if record.Recipe != "other" || record.SessionID != "unrelated-session" {
			return errors.New("filtered scan returned an unselected record")
		}
		filtered++
		return nil
	})
	if err != nil || filtered != 19 {
		t.Fatalf("filtered scan count=%d err=%v", filtered, err)
	}
	recipes := map[string]bool{}
	if err = postgres.ScanFilterOptions(ctx, func(record Record) error {
		recipes[record.Recipe] = true
		return nil
	}); err != nil || len(recipes) != 2 || !recipes["evaluation"] || !recipes["other"] {
		t.Fatalf("filter options recipes=%v err=%v", recipes, err)
	}
	scoped, err := postgres.QueryPage(ctx, QueryFilters{SessionID: "old-session", Recipe: "evaluation", RecipeSet: true}, 100, 0, true)
	if err != nil || scoped.Total != 2 || len(scoped.Records) != 2 {
		t.Fatalf("old scoped session=%+v err=%v", scoped, err)
	}
	unscoped, err := postgres.QueryPage(ctx, QueryFilters{SessionID: "old-session"}, 100, 0, true)
	if err != nil || unscoped.Total != 3 {
		t.Fatalf("cross-recipe session total=%d err=%v", unscoped.Total, err)
	}
	// The search box matches a session id as the in-memory matcher does.
	searched, err := postgres.QueryPage(ctx, QueryFilters{Search: "OLD-SESS"}, 100, 0, false)
	if err != nil || searched.Total != 3 {
		t.Fatalf("session search total=%d err=%v", searched.Total, err)
	}
	// since compares the zone-less timestamp column in UTC: the rows' own
	// instant written in another zone keeps them, a second later drops them.
	for _, window := range []struct {
		since time.Time
		want  int
	}{{fixture.Timestamp.In(time.FixedZone("KST", 9*3600)), 3}, {fixture.Timestamp.Add(time.Second), 0}} {
		windowed, windowErr := postgres.QueryPage(ctx, QueryFilters{SessionID: "old-session", Since: window.since}, 100, 0, false)
		if windowErr != nil || windowed.Total != window.want {
			t.Fatalf("since %s total=%d err=%v", window.since, windowed.Total, windowErr)
		}
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
	if err = postgres.ScanMetadata(ctx, QueryFilters{}, func(Record) error { return stopped }); !errors.Is(err, stopped) {
		t.Fatalf("callback failure was hidden: %v", err)
	}
}

// A harness filter keeps the records of its harnesses, a record no harness
// sent as an API client, and a page counts every harness the other filters
// keep, in the snapshot its total comes from.
//
// StorageIntegration: postgres
func TestPostgresHarnessQueriesIntegration(t *testing.T) {
	postgres := openPostgresReviewStore(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	start := time.Unix(1700000000, 0).UTC()
	records := []Record{
		{ID: "claude", Agent: &AgentIdentity{Harness: HarnessClaudeCode}},
		{ID: "codex", Agent: &AgentIdentity{Harness: HarnessCodex, ID: "subagent"}},
		{ID: "plain"},
		{ID: "unnamed-agent", Agent: &AgentIdentity{ID: "agent-hash"}},
		{ID: "codex-elsewhere", Agent: &AgentIdentity{Harness: HarnessCodex}, Recipe: "other"},
	}
	for index, record := range records {
		record.RequestID, record.Timestamp = record.ID, start.Add(time.Duration(index)*time.Second)
		if record.Recipe == "" {
			record.Recipe = "evaluation"
		}
		if _, err := postgres.Add(ctx, record); err != nil {
			t.Fatal(err)
		}
	}
	page, err := postgres.QueryPage(ctx, QueryFilters{Recipe: "evaluation", Harness: HarnessAPIClient + "," + HarnessCodex}, 10, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]string, len(page.Records))
	for index, record := range page.Records {
		ids[index] = record.ID
	}
	if page.Total != 3 || !reflect.DeepEqual(ids, []string{"unnamed-agent", "plain", "codex"}) {
		t.Fatalf("harness page total=%d records=%v", page.Total, ids)
	}
	want := map[string]int{HarnessClaudeCode: 1, HarnessCodex: 1, HarnessAPIClient: 2}
	if !reflect.DeepEqual(page.HarnessCounts, want) {
		t.Fatalf("harness counts = %v, want %v", page.HarnessCounts, want)
	}
	all, err := postgres.QueryPage(ctx, QueryFilters{}, 1, 0, false)
	if err != nil || all.Total != 5 || all.HarnessCounts[HarnessCodex] != 2 || len(all.Records) != 1 {
		t.Fatalf("unfiltered page=%+v err=%v", all, err)
	}
	scanned := 0
	if err = postgres.ScanMetadata(ctx, QueryFilters{Harness: HarnessClaudeCode}, func(record Record) error {
		if record.Agent == nil || record.Agent.Harness != HarnessClaudeCode {
			return fmt.Errorf("harness scan returned %s", record.RequestID)
		}
		scanned++
		return nil
	}); err != nil || scanned != 1 {
		t.Fatalf("harness scan count=%d err=%v", scanned, err)
	}
}
