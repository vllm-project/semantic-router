package postgres

import (
	"context"
	"database/sql"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	controlpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementstatistics"
)

func TestRepositoryCountsTenThousandKeysWithoutMaterializingRows(t *testing.T) {
	databaseURL := os.Getenv("VLLM_SR_CONTROL_PLANE_TEST_DATABASE_URL")
	if databaseURL == "" {
		databaseURL = os.Getenv("VLLM_SR_ACCESS_CONTROL_TEST_DATABASE_URL")
	}
	if databaseURL == "" {
		t.Skip("PostgreSQL Management statistics test database is not configured")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	database := isolatedStatisticsDatabase(t, ctx, databaseURL)
	if err := (controlpostgres.Migrator{DB: database}).Apply(ctx); err != nil {
		t.Fatal(err)
	}

	const (
		namespaceID = statisticsTestNamespace
		userID      = "22222222-2222-4222-8222-222222222222"
		teamID      = "33333333-3333-4333-8333-333333333333"
		accessOne   = "50000000-0000-4000-8000-000000000001"
		accessTwo   = "50000000-0000-4000-8000-000000000002"
		rateOne     = "60000000-0000-4000-8000-000000000001"
		rateTwo     = "60000000-0000-4000-8000-000000000002"
	)
	asOf := time.Date(2026, 8, 23, 10, 0, 0, 0, time.UTC)
	seedStatisticsScale(t, ctx, database, asOf, namespaceID, userID, teamID, accessOne, accessTwo, rateOne, rateTwo)
	repository, err := New(database)
	if err != nil {
		t.Fatal(err)
	}
	all := accesscontrol.ResultScope{NamespaceID: namespaceID, All: true}
	snapshot, err := repository.Snapshot(ctx, managementstatistics.Query{
		NamespaceID: namespaceID, AsOf: asOf, ExpiringBefore: asOf.Add(30 * 24 * time.Hour),
		Scopes: managementstatistics.Scopes{
			Users: &all, Teams: &all, APIKeys: &all, AccessPolicies: &all, RatePolicies: &all,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	assertStatisticsCount(t, "users", snapshot.Users, "1")
	assertStatisticsCount(t, "teams", snapshot.Teams, "1")
	assertStatisticsCount(t, "active API keys", snapshot.ActiveAPIKeys, "9500")
	assertStatisticsCount(t, "expiring API keys", snapshot.ExpiringAPIKeys, "500")
	assertStatisticsCount(t, "access policies", snapshot.AccessPolicies, "2")
	assertStatisticsCount(t, "active rate policies", snapshot.ActiveRatePolicies, "1")

	userScope := accesscontrol.ResultScope{NamespaceID: namespaceID, UserIDs: []accesscontrol.UserID{userID}}
	accessScope := accesscontrol.ResultScope{NamespaceID: namespaceID, ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
		accesscontrol.ScopeResourceAccessPolicy: {accesscontrol.ResourceID(accessOne)},
	}}
	narrow, err := repository.Snapshot(ctx, managementstatistics.Query{
		NamespaceID: namespaceID, AsOf: asOf, ExpiringBefore: asOf.Add(30 * 24 * time.Hour),
		Scopes: managementstatistics.Scopes{APIKeys: &userScope, AccessPolicies: &accessScope},
	})
	if err != nil {
		t.Fatal(err)
	}
	assertStatisticsCount(t, "user-owned active API keys", narrow.ActiveAPIKeys, "5000")
	assertStatisticsCount(t, "user-owned expiring API keys", narrow.ExpiringAPIKeys, "500")
	assertStatisticsCount(t, "narrow access policies", narrow.AccessPolicies, "1")
	if narrow.Users != nil || narrow.Teams != nil || narrow.ActiveRatePolicies != nil {
		t.Fatalf("unrequested statistics leaked: %#v", narrow)
	}
}

func seedStatisticsScale(
	t *testing.T,
	ctx context.Context,
	database *sql.DB,
	asOf time.Time,
	namespaceID, userID, teamID, accessOne, accessTwo, rateOne, rateTwo string,
) {
	t.Helper()
	statements := []struct {
		query string
		args  []any
	}{
		{`INSERT INTO access_namespaces
  (id,name,quota_partition_id,billing_currency,status)
VALUES ($1,'statistics-scale','statistics-scale-partition','USD','active')`, []any{namespaceID}},
		{`INSERT INTO access_subjects(namespace_id,id,kind) VALUES
  ($1,$2,'user'),($1,$3,'team')`, []any{namespaceID, userID, teamID}},
		{`INSERT INTO access_users(id,namespace_id,email,display_name,status)
VALUES ($1,$2,'statistics@example.com','Statistics User','active')`, []any{userID, namespaceID}},
		{`INSERT INTO access_teams(id,namespace_id,name,status)
VALUES ($1,$2,'Statistics Team','active')`, []any{teamID, namespaceID}},
		{`INSERT INTO access_policies(id,namespace_id,name,status) VALUES
  ($1,$3,'Access One','active'),($2,$3,'Access Two','disabled')`, []any{accessOne, accessTwo, namespaceID}},
		{`INSERT INTO rate_limit_policies(id,namespace_id,name,status) VALUES
  ($1,$3,'Rate One','active'),($2,$3,'Rate Two','disabled')`, []any{rateOne, rateTwo, namespaceID}},
		{`INSERT INTO access_subjects(namespace_id,id,kind)
SELECT $1, ('40000000-0000-4000-8000-' || lpad(value::text,12,'0'))::uuid, 'api_key'
FROM generate_series(1,10000) AS value`, []any{namespaceID}},
		{`INSERT INTO access_api_keys
  (id,namespace_id,name,owner_user_id,owner_team_id,status,expires_at)
SELECT ('40000000-0000-4000-8000-' || lpad(value::text,12,'0'))::uuid,
       $1, 'key-' || value,
       CASE WHEN value % 2 = 1 THEN $2::uuid END,
       CASE WHEN value % 2 = 0 THEN $3::uuid END,
       CASE WHEN value % 20 = 0 THEN 'disabled' ELSE 'active' END,
       CASE WHEN value % 20 = 1 THEN $4::timestamptz + interval '7 days' END
FROM generate_series(1,10000) AS value`, []any{namespaceID, userID, teamID, asOf}},
	}
	transaction, err := database.BeginTx(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = transaction.Rollback() }()
	for _, statement := range statements {
		if _, err := transaction.ExecContext(ctx, statement.query, statement.args...); err != nil {
			t.Fatal(err)
		}
	}
	if err := transaction.Commit(); err != nil {
		t.Fatal(err)
	}
}

func isolatedStatisticsDatabase(t *testing.T, ctx context.Context, databaseURL string) *sql.DB {
	t.Helper()
	admin, isolatedStatisticsDatabaseErr := sql.Open("postgres", databaseURL)
	if isolatedStatisticsDatabaseErr != nil {
		t.Fatal(isolatedStatisticsDatabaseErr)
	}
	t.Cleanup(func() { _ = admin.Close() })
	if err := admin.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	schema := "vsr_statistics_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	if _, err := admin.ExecContext(ctx, "CREATE SCHEMA "+pq.QuoteIdentifier(schema)); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		_, _ = admin.ExecContext(cleanup, "DROP SCHEMA "+pq.QuoteIdentifier(schema)+" CASCADE")
	})
	scopedURL, isolatedStatisticsDatabaseErr := statisticsDatabaseURL(databaseURL, schema)
	if isolatedStatisticsDatabaseErr != nil {
		t.Fatal(isolatedStatisticsDatabaseErr)
	}
	database, isolatedStatisticsDatabaseErr := sql.Open("postgres", scopedURL)
	if isolatedStatisticsDatabaseErr != nil {
		t.Fatal(isolatedStatisticsDatabaseErr)
	}
	t.Cleanup(func() { _ = database.Close() })
	return database
}

func statisticsDatabaseURL(databaseURL, schema string) (string, error) {
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

func assertStatisticsCount(t *testing.T, name string, got *managementstatistics.Count, want managementstatistics.Count) {
	t.Helper()
	if got == nil || *got != want {
		t.Fatalf("%s = %v, want %s", name, got, want)
	}
}
