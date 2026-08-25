package usageledger

import (
	"context"
	"database/sql"
	"errors"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
	"github.com/redis/go-redis/v9"

	controlpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/postgres"
)

func waitUsageCondition(t *testing.T, ctx context.Context, condition func() bool) {
	t.Helper()
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	for {
		if condition() {
			return
		}
		select {
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		case <-ticker.C:
		}
	}
}

func namespaceEventCount(t *testing.T, ctx context.Context, db *sql.DB, namespaceID string) int {
	t.Helper()
	var count int
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM usage_events WHERE namespace_id = $1`, namespaceID).Scan(&count); err != nil {
		t.Fatal(err)
	}
	return count
}

func namespaceRollupRequests(t *testing.T, ctx context.Context, db *sql.DB, table, namespaceID string) int {
	t.Helper()
	var requests int
	statement := "SELECT COALESCE(sum(requests),0)::int FROM " + pq.QuoteIdentifier(table) +
		" WHERE namespace_id = $1 AND view = 'request'"
	if err := db.QueryRowContext(ctx, statement, namespaceID).Scan(&requests); err != nil {
		t.Fatal(err)
	}
	return requests
}

var errInjectedAckFailure = errors.New("injected acknowledgement failure")

var errInjectedRollupFailure = errors.New("injected rollup failure")

type failAckOnceStream struct {
	Stream
	failed bool
}

func (s *failAckOnceStream) Ack(ctx context.Context, ids []string) error {
	if !s.failed {
		s.failed = true
		return errInjectedAckFailure
	}
	return s.Stream.Ack(ctx, ids)
}

type failCommitHookOnce struct {
	delegate CommittedBatchHook
	failed   bool
}

func (hook *failCommitHookOnce) AfterCommit(ctx context.Context, events []TerminalEvent) error {
	if !hook.failed {
		hook.failed = true
		return errInjectedRollupFailure
	}
	return hook.delegate.AfterCommit(ctx, events)
}

func integrationStores(t *testing.T) (*sql.DB, *redis.Client) {
	t.Helper()
	databaseURL := os.Getenv("VLLM_SR_USAGE_LEDGER_TEST_DATABASE_URL")
	redisURL := os.Getenv("VLLM_SR_USAGE_LEDGER_TEST_REDIS_URL")
	if databaseURL == "" || redisURL == "" {
		t.Skip("usage ledger PostgreSQL and Redis integration stores are not configured")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	admin, integrationStoresErr := sql.Open("postgres", databaseURL)
	if integrationStoresErr != nil {
		t.Fatal(integrationStoresErr)
	}
	t.Cleanup(func() { _ = admin.Close() })
	if err := admin.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	schema := "vsr_usageledger_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	if _, err := admin.ExecContext(ctx, "CREATE SCHEMA "+pq.QuoteIdentifier(schema)); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, stop := context.WithTimeout(context.Background(), 15*time.Second)
		defer stop()
		_, _ = admin.ExecContext(cleanup, "DROP SCHEMA "+pq.QuoteIdentifier(schema)+" CASCADE")
	})
	scopedURL, integrationStoresErr := databaseURLWithSearchPath(databaseURL, schema)
	if integrationStoresErr != nil {
		t.Fatal(integrationStoresErr)
	}
	db, integrationStoresErr := sql.Open("postgres", scopedURL)
	if integrationStoresErr != nil {
		t.Fatal(integrationStoresErr)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := (controlpostgres.Migrator{DB: db}).Apply(ctx); err != nil {
		t.Fatalf("apply control-plane migrations: %v", err)
	}
	options, integrationStoresErr := redis.ParseURL(redisURL)
	if integrationStoresErr != nil {
		t.Fatal(integrationStoresErr)
	}
	client := redis.NewClient(options)
	t.Cleanup(func() { _ = client.Close() })
	if err := client.Ping(ctx).Err(); err != nil {
		t.Fatal(err)
	}
	return db, client
}

func seedNamespaceAndFencePolicy(t *testing.T, ctx context.Context, db *sql.DB) {
	t.Helper()
	statements := []struct {
		query string
		args  []any
	}{
		{`INSERT INTO access_namespaces(id,name,quota_partition_id,billing_currency,status)
VALUES ($1,'integration','partition-integration','USD','active')`, []any{testNamespaceID}},
		{`INSERT INTO access_subjects(namespace_id,id,kind) VALUES
($1,$2,'team'),($1,$3,'user'),($1,$4,'api_key')`, []any{testNamespaceID, testTeamID, testUserID, testKeyID}},
		{`INSERT INTO access_teams(id,namespace_id,name,status) VALUES ($1,$2,'integration-team','active')`, []any{testTeamID, testNamespaceID}},
		{`INSERT INTO access_users(id,namespace_id,email,display_name,status)
VALUES ($1,$2,'usage@example.invalid','Usage User','active')`, []any{testUserID, testNamespaceID}},
		{`INSERT INTO access_team_memberships(namespace_id,team_id,user_id,role,status)
VALUES ($1,$2,$3,'member','active')`, []any{testNamespaceID, testTeamID, testUserID}},
		{`INSERT INTO access_api_keys(id,namespace_id,name,owner_user_id,context_team_id,status)
VALUES ($1,$2,'usage-key',$3,$4,'active')`, []any{testKeyID, testNamespaceID, testUserID, testTeamID}},
		{`INSERT INTO rate_limit_policies(id,namespace_id,name,status)
VALUES ('dddddddd-dddd-4ddd-8ddd-dddddddddddd',$1,'integration-budget','active')`, []any{testNamespaceID}},
		{`INSERT INTO rate_limit_rules(
  id,policy_id,metric,algorithm,limit_value,window_seconds,accounting,enforcement,ordinal
) VALUES ('cccccccc-cccc-4ccc-8ccc-cccccccccccc','dddddddd-dddd-4ddd-8ddd-dddddddddddd',
  'total_tokens','sliding_log',1000,60,'response_actual','enforce',0)`, nil},
		{`INSERT INTO rate_limit_bindings(
  id,namespace_id,policy_id,subject_id,binding_mode,quota_partition_id,status
) VALUES ('bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb',$1,'dddddddd-dddd-4ddd-8ddd-dddddddddddd',
  $2,'allocation','partition-integration','active')`, []any{testNamespaceID, testTeamID}},
	}
	for _, statement := range statements {
		if _, err := db.ExecContext(ctx, statement.query, statement.args...); err != nil {
			t.Fatalf("seed integration schema: %v", err)
		}
	}
}

func addTerminalEvent(t *testing.T, ctx context.Context, client *redis.Client, key string, event TerminalEvent) {
	t.Helper()
	payload, err := EncodeTerminalEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	if err := client.XAdd(ctx, &redis.XAddArgs{Stream: key, Values: streamValues(event, payload)}).Err(); err != nil {
		t.Fatal(err)
	}
}

func assertLedgerCounts(t *testing.T, ctx context.Context, db *sql.DB, events, dispatches, attempts int) {
	t.Helper()
	assertCount(t, ctx, db, "usage_settlements", events)
	assertCount(t, ctx, db, "usage_events", events)
	assertCount(t, ctx, db, "usage_dispatches", dispatches)
	assertCount(t, ctx, db, "usage_dispatch_attempts", attempts)
}

func assertCount(t *testing.T, ctx context.Context, db *sql.DB, table string, want int) {
	t.Helper()
	var got int
	if err := db.QueryRowContext(ctx, "SELECT count(*) FROM "+pq.QuoteIdentifier(table)).Scan(&got); err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("%s count = %d, want %d", table, got, want)
	}
}

func databaseURLWithSearchPath(databaseURL, schema string) (string, error) {
	if !strings.Contains(databaseURL, "://") {
		return databaseURL + " search_path=" + schema, nil
	}
	parsed, err := url.Parse(databaseURL)
	if err != nil {
		return "", err
	}
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	return parsed.String(), nil
}

func deleteRedisPrefix(client *redis.Client, prefix string) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	iterator := client.Scan(ctx, 0, prefix+":*", 100).Iterator()
	keys := make([]string, 0)
	for iterator.Next(ctx) {
		keys = append(keys, iterator.Val())
	}
	if len(keys) != 0 {
		_ = client.Del(ctx, keys...).Err()
	}
}
