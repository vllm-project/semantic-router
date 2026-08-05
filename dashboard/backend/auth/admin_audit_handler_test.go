package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestAdminAuditLogsHandlerReturnsFilteredPage(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	ctx := context.Background()
	fixtures := []AuditLog{
		{UserID: "admin-a", Action: "user.create", Resource: "users", Method: "POST", Path: "/api/admin/users", StatusCode: 201, CreatedAt: 1_704_067_200},
		{UserID: "admin-a", Action: "user.update", Resource: "users/user-b", Method: "PATCH", Path: "/api/admin/users/user-b", StatusCode: 204, CreatedAt: 1_704_153_600},
		{UserID: "admin-b", Action: "router.deploy", Resource: "config", Method: "POST", Path: "/api/config/deploy", StatusCode: 503, CreatedAt: 1_704_240_000},
	}
	for _, fixture := range fixtures {
		if err := svc.store.AddAuditLog(ctx, fixture); err != nil {
			t.Fatalf("AddAuditLog() error = %v", err)
		}
	}

	request := httptest.NewRequest(
		http.MethodGet,
		"/api/admin/audit-logs?user=admin-a&status=success&from=2024-01-01&to=2024-01-02&sort=createdAt&order=asc&page=2&limit=1",
		nil,
	)
	request = request.WithContext(WithAuthContext(request.Context(), AuthContext{
		UserID: "admin-a",
		Perms:  map[string]bool{PermUsersManage: true},
	}))
	recorder := httptest.NewRecorder()

	adminAuditLogsHandler(svc).ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %q", recorder.Code, recorder.Body.String())
	}
	var payload AuditLogPageResponse
	if err := json.NewDecoder(recorder.Body).Decode(&payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if payload.Total != 2 || payload.Page != 2 || payload.Limit != 1 || len(payload.Logs) != 1 {
		t.Fatalf("response = %#v, want second page of two", payload)
	}
	if payload.Logs[0].Action != "user.update" {
		t.Fatalf("action = %q, want user.update", payload.Logs[0].Action)
	}
}

func TestAdminAuditLogsHandlerPreservesLegacyArrayResponse(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	ctx := context.Background()
	for index := 0; index < 27; index++ {
		if err := svc.store.AddAuditLog(ctx, AuditLog{
			UserID:    "filler",
			Action:    fmt.Sprintf("filler.%02d", index),
			Resource:  "compatibility-test",
			CreatedAt: int64(index + 1),
		}); err != nil {
			t.Fatalf("AddAuditLog(filler %d) error = %v", index, err)
		}
	}
	fixtures := []AuditLog{
		{UserID: "admin-a", Action: "user.create", Resource: "users", CreatedAt: 100},
		{UserID: "admin-a", Action: "user.update", Resource: "users/user-b", CreatedAt: 200},
		{UserID: "admin-b", Action: "router.deploy", Resource: "config", CreatedAt: 300},
	}
	for _, fixture := range fixtures {
		if err := svc.store.AddAuditLog(ctx, fixture); err != nil {
			t.Fatalf("AddAuditLog() error = %v", err)
		}
	}

	for _, test := range []struct {
		name       string
		path       string
		wantLength int
		wantAction string
	}{
		{name: "no query", path: "/api/admin/audit-logs", wantLength: 30, wantAction: "router.deploy"},
		{name: "legacy filters", path: "/api/admin/audit-logs?userId=admin-a&limit=1", wantLength: 1, wantAction: "user.update"},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodGet, test.path, nil)
			request = request.WithContext(WithAuthContext(request.Context(), AuthContext{
				UserID: "admin-a",
				Perms:  map[string]bool{PermUsersManage: true},
			}))
			recorder := httptest.NewRecorder()

			adminAuditLogsHandler(svc).ServeHTTP(recorder, request)
			if recorder.Code != http.StatusOK {
				t.Fatalf("status = %d, body = %q", recorder.Code, recorder.Body.String())
			}

			var payload []AuditLog
			if err := json.NewDecoder(recorder.Body).Decode(&payload); err != nil {
				t.Fatalf("decode legacy response: %v", err)
			}
			if len(payload) != test.wantLength {
				t.Fatalf("len(response) = %d, want %d", len(payload), test.wantLength)
			}
			if payload[0].Action != test.wantAction {
				t.Fatalf("first action = %q, want %q", payload[0].Action, test.wantAction)
			}
		})
	}
}

func TestAdminAuditLogsHandlerValidatesFiltersAndPermission(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	tests := []struct {
		name  string
		path  string
		perms map[string]bool
		want  int
	}{
		{name: "requires users manage", path: "/api/admin/audit-logs", perms: map[string]bool{PermUsersView: true}, want: http.StatusForbidden},
		{name: "rejects unknown sort", path: "/api/admin/audit-logs?sort=created_at%3BDROP+TABLE+users", perms: map[string]bool{PermUsersManage: true}, want: http.StatusBadRequest},
		{name: "rejects invalid order", path: "/api/admin/audit-logs?order=sideways", perms: map[string]bool{PermUsersManage: true}, want: http.StatusBadRequest},
		{name: "rejects invalid page", path: "/api/admin/audit-logs?page=0", perms: map[string]bool{PermUsersManage: true}, want: http.StatusBadRequest},
		{name: "rejects invalid status", path: "/api/admin/audit-logs?status=unknown", perms: map[string]bool{PermUsersManage: true}, want: http.StatusBadRequest},
		{name: "rejects reversed dates", path: "/api/admin/audit-logs?from=2024-02-02&to=2024-01-01", perms: map[string]bool{PermUsersManage: true}, want: http.StatusBadRequest},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodGet, test.path, nil)
			request = request.WithContext(WithAuthContext(request.Context(), AuthContext{Perms: test.perms}))
			recorder := httptest.NewRecorder()

			adminAuditLogsHandler(svc).ServeHTTP(recorder, request)
			if recorder.Code != test.want {
				t.Fatalf("status = %d, want %d; body = %q", recorder.Code, test.want, recorder.Body.String())
			}
		})
	}
}
