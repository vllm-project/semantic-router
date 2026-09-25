package auth

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestAdminUserMutationRechecksPermissionBeforeStoreWrite(t *testing.T) {
	svc := newTestAuthService(t)
	admin := newTestUser(t, svc, "admin-revocation@example.test", RoleAdmin, "active")
	for _, test := range []struct {
		name, method, body string
	}{
		{"patch", http.MethodPatch, `{"status":"inactive"}`},
		{"delete", http.MethodDelete, ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			target := newTestUser(t, svc, test.name+"-target@example.test", RoleRead, "active")
			request := httptest.NewRequest(test.method, "/api/admin/users/"+target.ID, strings.NewReader(test.body))
			ctx := WithAuthContext(request.Context(), AuthContext{
				UserID: admin.ID,
				Perms:  map[string]bool{PermUsersManage: true},
			})
			var checks atomic.Int32
			ctx = WithPermissionRevalidator(ctx, func(context.Context) error {
				checks.Add(1)
				return errors.New("users.manage revoked")
			})
			response := httptest.NewRecorder()
			adminUserItemHandler(svc)(response, request.WithContext(ctx))
			if response.Code != http.StatusForbidden || checks.Load() != 1 {
				t.Fatalf("status = %d, permission checks = %d", response.Code, checks.Load())
			}
			stored, err := svc.store.GetUserByID(t.Context(), target.ID)
			if err != nil || stored.Role != RoleRead || stored.Status != "active" {
				t.Fatalf("revoked user mutation changed target: user=%+v err=%v", stored, err)
			}
		})
	}
}
