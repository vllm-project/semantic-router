package postgres

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementstatistics"
)

const statisticsTestNamespace = "11111111-1111-4111-8111-111111111111"

func TestRepositoryUsesOneBoundedStatementAndPreservesUnavailableFields(t *testing.T) {
	database, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherEqual))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = database.Close() })
	repository, err := New(database)
	if err != nil {
		t.Fatal(err)
	}
	asOf := time.Date(2026, 8, 23, 10, 0, 0, 0, time.UTC)
	expiringBefore := asOf.Add(managementstatistics.DefaultExpiringWindow)
	mock.ExpectQuery(statisticsQuery).WithArgs(
		statisticsTestNamespace, asOf, expiringBefore,
		true, true, sqlmock.AnyArg(),
		false, false, sqlmock.AnyArg(),
		true, false, sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg(),
		false, false, sqlmock.AnyArg(),
		true, true, sqlmock.AnyArg(),
	).WillReturnRows(sqlmock.NewRows([]string{
		"users", "teams", "active_keys", "expiring_keys", "access_policies", "active_rate_policies",
	}).AddRow("10001", nil, "9000", "32", nil, "8"))

	all := accesscontrol.ResultScope{NamespaceID: statisticsTestNamespace, All: true}
	keyScope := accesscontrol.ResultScope{
		NamespaceID: statisticsTestNamespace,
		APIKeyIDs:   []accesscontrol.APIKeyID{"22222222-2222-4222-8222-222222222222"},
	}
	snapshot, err := repository.Snapshot(context.Background(), managementstatistics.Query{
		NamespaceID: statisticsTestNamespace, AsOf: asOf, ExpiringBefore: expiringBefore,
		Scopes: managementstatistics.Scopes{Users: &all, APIKeys: &keyScope, RatePolicies: &all},
	})
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Users == nil || *snapshot.Users != "10001" || snapshot.Teams != nil ||
		snapshot.ActiveAPIKeys == nil || *snapshot.ActiveAPIKeys != "9000" || snapshot.AccessPolicies != nil ||
		snapshot.ActiveRatePolicies == nil || *snapshot.ActiveRatePolicies != "8" {
		t.Fatalf("snapshot = %#v", snapshot)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestStatisticsQueryRemainsScaleBounded(t *testing.T) {
	for _, required := range []string{
		"count(*)::text", "u.id=ANY($6::uuid[])", "t.id=ANY($9::uuid[])",
		"k.id=ANY($12::uuid[])", "k.owner_user_id=ANY($13::uuid[])",
		"k.owner_team_id=ANY($14::uuid[])", "p.id=ANY($17::uuid[])", "p.id=ANY($20::uuid[])",
	} {
		if !strings.Contains(statisticsQuery, required) {
			t.Errorf("statistics query is missing %q", required)
		}
	}
	for _, forbidden := range []string{" LIMIT ", " OFFSET ", "SELECT *"} {
		if strings.Contains(statisticsQuery, forbidden) {
			t.Errorf("statistics query contains unbounded shape %q", forbidden)
		}
	}
}
