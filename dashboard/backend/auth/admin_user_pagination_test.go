package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestAdminUsersHandlerKeepsLegacyDefaultAndHonorsExplicitLimit(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	ctx := context.Background()
	for index := 0; index < 25; index++ {
		_, err := svc.store.CreateUser(
			ctx,
			fmt.Sprintf("user-%02d@example.com", index),
			fmt.Sprintf("User %02d", index),
			"hash",
			RoleRead,
			"active",
		)
		if err != nil {
			t.Fatalf("CreateUser(%d) error = %v", index, err)
		}
	}

	for _, test := range []struct {
		name       string
		path       string
		wantLength int
		wantLimit  int
	}{
		{name: "omitted limit keeps legacy default", path: "/api/admin/users", wantLength: 25, wantLimit: 100},
		{name: "explicit UI limit is honored", path: "/api/admin/users?page=1&limit=10", wantLength: 10, wantLimit: 10},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodGet, test.path, nil)
			request = request.WithContext(WithAuthContext(request.Context(), AuthContext{
				Perms: map[string]bool{PermUsersView: true},
			}))
			recorder := httptest.NewRecorder()

			handleAdminUsersList(recorder, request, svc)
			if recorder.Code != http.StatusOK {
				t.Fatalf("status = %d, body = %q", recorder.Code, recorder.Body.String())
			}

			var payload ListUsersResponse
			if err := json.NewDecoder(recorder.Body).Decode(&payload); err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if len(payload.Users) != test.wantLength {
				t.Fatalf("len(users) = %d, want %d", len(payload.Users), test.wantLength)
			}
			if payload.Limit != test.wantLimit {
				t.Fatalf("limit = %d, want %d", payload.Limit, test.wantLimit)
			}
		})
	}
}
