package auth

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
)

type auditListStoreFixture struct {
	ctx   context.Context
	logs  []AuditLog
	store *Store
}

func newAuditListStoreFixture(t *testing.T, userIDs []string, logs []AuditLog) *auditListStoreFixture {
	t.Helper()

	store, err := NewStore(filepath.Join(t.TempDir(), "auth.db"))
	if err != nil {
		t.Fatalf("NewStore() error = %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	seedAuditFixtureUsers(t, store, userIDs...)

	ctx := context.Background()
	for _, log := range logs {
		if err := store.AddAuditLog(ctx, log); err != nil {
			t.Fatalf("AddAuditLog(%q) error = %v", log.Action, err)
		}
	}

	return &auditListStoreFixture{ctx: ctx, logs: logs, store: store}
}

func enterpriseAuditLogFixtures() []AuditLog {
	return []AuditLog{
		{UserID: "user-a", Action: "user.create", Resource: "users", Method: "POST", Path: "/api/admin/users", IP: "10.0.0.1", UserAgent: "console", StatusCode: 201, CreatedAt: 1_704_067_200},
		{UserID: "user-a", Action: "user.update", Resource: "users/user-b", Method: "PATCH", Path: "/api/admin/users/user-b", IP: "10.0.0.2", UserAgent: "console", StatusCode: 204, CreatedAt: 1_704_153_600},
		{UserID: "user-b", Action: "router.deploy", Resource: "config", Method: "POST", Path: "/api/config/deploy", IP: "10.0.0.3", UserAgent: "automation", StatusCode: 503, CreatedAt: 1_704_240_000},
		{Action: "auth.login", Resource: "session", Method: "POST", Path: "/api/auth/login", IP: "10.0.0.4", UserAgent: "browser", StatusCode: 401, CreatedAt: 1_704_326_400},
		{UserID: "user-c", Action: "user.delete", Resource: "users/user-c", Method: "DELETE", Path: "/api/admin/users/percent%literal", IP: "10.0.0.5", UserAgent: "console", StatusCode: 204, CreatedAt: 1_704_412_800},
	}
}

func TestQueryAuditLogsSupportsEnterpriseFiltersPagingAndSafeSort(t *testing.T) {
	t.Parallel()

	fixture := newAuditListStoreFixture(
		t,
		[]string{"user-a", "user-b", "user-c"},
		enterpriseAuditLogFixtures(),
	)
	t.Run("combines text, field, status, and date filters", fixture.testCombinedFilters)
	t.Run("returns total independently from the requested page", fixture.testPagingTotal)
	t.Run("treats LIKE wildcards as literal search text", fixture.testLiteralWildcardSearch)
	t.Run("supports response status classes and exact codes", fixture.testStatusFilters)
	t.Run("rejects invalid status and safely falls back from unknown sort", fixture.testInvalidFiltersAndSafeSort)
}

func (fixture *auditListStoreFixture) testCombinedFilters(t *testing.T) {
	logs, total, err := fixture.store.QueryAuditLogs(fixture.ctx, AuditLogListOptions{
		Query:    "UPDATE",
		UserID:   "user-a",
		Action:   "user.update",
		Resource: "users/user-b",
		Status:   "success",
		From:     1_704_153_600,
		To:       1_704_153_600,
		Limit:    20,
	})
	if err != nil {
		t.Fatalf("QueryAuditLogs() error = %v", err)
	}
	if total != 1 || len(logs) != 1 || logs[0].Action != "user.update" {
		t.Fatalf("QueryAuditLogs() = %#v, total %d; want user.update", logs, total)
	}
}

func (fixture *auditListStoreFixture) testPagingTotal(t *testing.T) {
	logs, total, err := fixture.store.QueryAuditLogs(fixture.ctx, AuditLogListOptions{
		Sort:   "createdAt",
		Order:  "asc",
		Limit:  2,
		Offset: 2,
	})
	if err != nil {
		t.Fatalf("QueryAuditLogs() error = %v", err)
	}
	if total != len(fixture.logs) || len(logs) != 2 || logs[0].Action != "router.deploy" {
		t.Fatalf("QueryAuditLogs() = %#v, total %d; want middle page", logs, total)
	}
}

func (fixture *auditListStoreFixture) testLiteralWildcardSearch(t *testing.T) {
	logs, total, err := fixture.store.QueryAuditLogs(fixture.ctx, AuditLogListOptions{Query: "%", Limit: 20})
	if err != nil {
		t.Fatalf("QueryAuditLogs() error = %v", err)
	}
	if total != 1 || len(logs) != 1 || logs[0].Action != "user.delete" {
		t.Fatalf("QueryAuditLogs() = %#v, total %d; want literal percent match", logs, total)
	}
}

func (fixture *auditListStoreFixture) testStatusFilters(t *testing.T) {
	logs, total, err := fixture.store.QueryAuditLogs(fixture.ctx, AuditLogListOptions{Status: "server_error", Limit: 20})
	if err != nil || total != 1 || len(logs) != 1 || logs[0].StatusCode != 503 {
		t.Fatalf("server_error query = %#v, total %d, err %v", logs, total, err)
	}
	logs, total, err = fixture.store.QueryAuditLogs(fixture.ctx, AuditLogListOptions{Status: "401", Limit: 20})
	if err != nil || total != 1 || len(logs) != 1 || logs[0].StatusCode != 401 {
		t.Fatalf("401 query = %#v, total %d, err %v", logs, total, err)
	}
}

func (fixture *auditListStoreFixture) testInvalidFiltersAndSafeSort(t *testing.T) {
	if _, _, err := fixture.store.QueryAuditLogs(fixture.ctx, AuditLogListOptions{Status: "not-a-status"}); !errors.Is(err, ErrInvalidAuditLogFilter) {
		t.Fatalf("invalid status error = %v, want ErrInvalidAuditLogFilter", err)
	}
	logs, total, err := fixture.store.QueryAuditLogs(fixture.ctx, AuditLogListOptions{
		Query: "' OR 1=1 --",
		Limit: 20,
	})
	if err != nil || total != 0 || len(logs) != 0 {
		t.Fatalf("injection-shaped search = %#v, total %d, err %v; want no matches", logs, total, err)
	}

	logs, total, err = fixture.store.QueryAuditLogs(fixture.ctx, AuditLogListOptions{
		Sort:  "created_at; DROP TABLE user_audit_logs; --",
		Order: "asc; DROP TABLE users; --",
		Limit: 1,
	})
	if err != nil || total != len(fixture.logs) || len(logs) != 1 || logs[0].Action != "user.delete" {
		t.Fatalf("safe fallback query = %#v, total %d, err %v", logs, total, err)
	}
}

func TestListAuditLogsRetainsLegacyStoreContract(t *testing.T) {
	t.Parallel()

	fixture := newAuditListStoreFixture(
		t,
		[]string{"legacy-user"},
		[]AuditLog{{UserID: "legacy-user", Action: "user.update", Resource: "users", StatusCode: 200}},
	)

	logs, err := fixture.store.ListAuditLogs(fixture.ctx, "legacy-user", "user.update", "users", 10, 0)
	if err != nil || len(logs) != 1 {
		t.Fatalf("ListAuditLogs() = %#v, %v; want one row", logs, err)
	}
}
