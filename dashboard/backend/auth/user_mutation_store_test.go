package auth

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

func TestUserMutationRejectsStaleRoleAndStatusSnapshots(t *testing.T) {
	t.Parallel()

	t.Run("role-only request cannot reactivate a concurrently deactivated user", func(t *testing.T) {
		t.Parallel()
		svc := newTestAuthService(t)
		user := newTestUser(t, svc, "stale-status@example.com", RoleRead, defaultUserStatusActive)

		if _, err := svc.store.UpdateUserRoleOrStatus(
			context.Background(),
			user.ID,
			"",
			"inactive",
		); err != nil {
			t.Fatalf("deactivate user: %v", err)
		}
		_, err := svc.store.UpdateUserRoleOrStatusIfCurrent(
			context.Background(),
			user.ID,
			RoleRead,
			defaultUserStatusActive,
			RoleWrite,
			defaultUserStatusActive,
		)
		if !errors.Is(err, ErrUserStateChanged) {
			t.Fatalf("stale role update error = %v, want ErrUserStateChanged", err)
		}
		assertStoredUserState(t, svc, user.ID, RoleRead, "inactive")
	})

	t.Run("status-only request cannot restore a concurrently changed role", func(t *testing.T) {
		t.Parallel()
		svc := newTestAuthService(t)
		user := newTestUser(t, svc, "stale-role@example.com", RoleRead, defaultUserStatusActive)

		if _, err := svc.store.UpdateUserRoleOrStatus(
			context.Background(),
			user.ID,
			RoleWrite,
			"",
		); err != nil {
			t.Fatalf("change role: %v", err)
		}
		_, err := svc.store.UpdateUserRoleOrStatusIfCurrent(
			context.Background(),
			user.ID,
			RoleRead,
			defaultUserStatusActive,
			RoleRead,
			"inactive",
		)
		if !errors.Is(err, ErrUserStateChanged) {
			t.Fatalf("stale status update error = %v, want ErrUserStateChanged", err)
		}
		assertStoredUserState(t, svc, user.ID, RoleWrite, defaultUserStatusActive)
	})
}

func TestConcurrentManagerMutationsKeepOneActiveManager(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name   string
		mutate func(context.Context, *Store, *User) error
	}{
		{
			name: "mutual deactivation",
			mutate: func(ctx context.Context, store *Store, user *User) error {
				_, err := store.UpdateUserRoleOrStatusIfCurrent(
					ctx,
					user.ID,
					RoleAdmin,
					defaultUserStatusActive,
					RoleAdmin,
					"inactive",
				)
				return err
			},
		},
		{
			name: "mutual deletion",
			mutate: func(ctx context.Context, store *Store, user *User) error {
				return store.DeleteUserIfCurrent(
					ctx,
					user.ID,
					RoleAdmin,
					defaultUserStatusActive,
				)
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			svc := newTestAuthService(t)
			first := newTestUser(t, svc, "first-manager@example.com", RoleAdmin, defaultUserStatusActive)
			second := newTestUser(t, svc, "second-manager@example.com", RoleAdmin, defaultUserStatusActive)

			start := make(chan struct{})
			errorsByMutation := make(chan error, 2)
			var ready sync.WaitGroup
			ready.Add(2)
			for _, user := range []*User{first, second} {
				user := user
				go func() {
					ready.Done()
					<-start
					errorsByMutation <- test.mutate(context.Background(), svc.store, user)
				}()
			}
			ready.Wait()
			close(start)

			var succeeded, protected int
			for range 2 {
				err := <-errorsByMutation
				switch {
				case err == nil:
					succeeded++
				case errors.Is(err, ErrLastActiveUserManager):
					protected++
				default:
					t.Fatalf("manager mutation error = %v", err)
				}
			}
			if succeeded != 1 || protected != 1 {
				t.Fatalf("mutation results: succeeded=%d protected=%d, want 1 each", succeeded, protected)
			}
			assertActiveManagerCount(t, svc, 1)
		})
	}
}

func TestDeleteAndRoleUpdatePreserveLastManagerInEitherOrder(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name        string
		firstDelete bool
	}{
		{name: "delete then demote", firstDelete: true},
		{name: "demote then delete", firstDelete: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			svc := newTestAuthService(t)
			first := newTestUser(t, svc, "cross-first@example.com", RoleAdmin, defaultUserStatusActive)
			second := newTestUser(t, svc, "cross-second@example.com", RoleAdmin, defaultUserStatusActive)

			if test.firstDelete {
				if err := svc.store.DeleteUser(context.Background(), first.ID); err != nil {
					t.Fatalf("first DeleteUser() error = %v", err)
				}
				_, err := svc.store.UpdateUserRoleOrStatus(
					context.Background(), second.ID, RoleRead, "",
				)
				if !errors.Is(err, ErrLastActiveUserManager) {
					t.Fatalf("second demotion error = %v, want ErrLastActiveUserManager", err)
				}
			} else {
				if _, err := svc.store.UpdateUserRoleOrStatus(
					context.Background(), first.ID, RoleRead, "",
				); err != nil {
					t.Fatalf("first demotion error = %v", err)
				}
				if err := svc.store.DeleteUser(context.Background(), second.ID); !errors.Is(err, ErrLastActiveUserManager) {
					t.Fatalf("second deletion error = %v, want ErrLastActiveUserManager", err)
				}
			}
			assertActiveManagerCount(t, svc, 1)
		})
	}
}

func TestUserMutationEnforcesManagerPermissionSemantics(t *testing.T) {
	t.Parallel()

	t.Run("sole role manager cannot be demoted or deleted", func(t *testing.T) {
		t.Parallel()
		svc := newTestAuthService(t)
		manager := newTestUser(t, svc, "sole-manager@example.com", RoleAdmin, defaultUserStatusActive)

		if _, err := svc.store.UpdateUserRoleOrStatus(
			context.Background(), manager.ID, RoleRead, "",
		); !errors.Is(err, ErrLastActiveUserManager) {
			t.Fatalf("demotion error = %v, want ErrLastActiveUserManager", err)
		}
		if err := svc.store.DeleteUser(context.Background(), manager.ID); !errors.Is(err, ErrLastActiveUserManager) {
			t.Fatalf("deletion error = %v, want ErrLastActiveUserManager", err)
		}
		assertStoredUserState(t, svc, manager.ID, RoleAdmin, defaultUserStatusActive)
	})

	t.Run("direct permission counts as a remaining manager", func(t *testing.T) {
		t.Parallel()
		svc := newTestAuthService(t)
		admin := newTestUser(t, svc, "role-manager@example.com", RoleAdmin, defaultUserStatusActive)
		direct := newTestUser(t, svc, "direct-manager@example.com", RoleRead, defaultUserStatusActive)
		grantUserPermission(t, svc, direct.ID, PermUsersManage)

		if _, err := svc.store.UpdateUserRoleOrStatus(
			context.Background(), admin.ID, RoleRead, "",
		); err != nil {
			t.Fatalf("demote role manager: %v", err)
		}
		assertActiveManagerCount(t, svc, 1)
	})

	t.Run("target direct permission survives a role demotion", func(t *testing.T) {
		t.Parallel()
		svc := newTestAuthService(t)
		manager := newTestUser(t, svc, "direct-target@example.com", RoleAdmin, defaultUserStatusActive)
		grantUserPermission(t, svc, manager.ID, PermUsersManage)

		if _, err := svc.store.UpdateUserRoleOrStatus(
			context.Background(), manager.ID, RoleRead, "",
		); err != nil {
			t.Fatalf("demote direct manager: %v", err)
		}
		assertStoredUserState(t, svc, manager.ID, RoleRead, defaultUserStatusActive)
		assertActiveManagerCount(t, svc, 1)
	})

	t.Run("ordinary user mutation does not require a manager", func(t *testing.T) {
		t.Parallel()
		svc := newTestAuthService(t)
		user := newTestUser(t, svc, "ordinary@example.com", RoleRead, defaultUserStatusActive)

		if _, err := svc.store.UpdateUserRoleOrStatus(
			context.Background(), user.ID, RoleWrite, "inactive",
		); err != nil {
			t.Fatalf("ordinary mutation error = %v", err)
		}
		assertStoredUserState(t, svc, user.ID, RoleWrite, "inactive")
	})
}

func TestNormalizeUserUpdatePreservesOmittedFields(t *testing.T) {
	t.Parallel()

	role := RoleWrite
	status := defaultUserStatusActive
	target := &User{Role: RoleRead, Status: "inactive"}

	nextRole, nextStatus, err := normalizeUserUpdate(target, UpdateUserRequest{Role: &role})
	if err != nil {
		t.Fatalf("normalize role-only update: %v", err)
	}
	if nextRole != RoleWrite || nextStatus != "inactive" {
		t.Fatalf("role-only update = (%q, %q), want (%q, inactive)", nextRole, nextStatus, RoleWrite)
	}

	target = &User{Role: RoleRead, Status: "inactive"}
	nextRole, nextStatus, err = normalizeUserUpdate(target, UpdateUserRequest{Status: &status})
	if err != nil {
		t.Fatalf("normalize status-only update: %v", err)
	}
	if nextRole != RoleRead || nextStatus != defaultUserStatusActive {
		t.Fatalf("status-only update = (%q, %q), want (%q, %q)", nextRole, nextStatus, RoleRead, defaultUserStatusActive)
	}
}

func TestWriteUserMutationErrorUsesConflictWithoutLeakingInternalErrors(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name string
		err  error
		want int
	}{
		{name: "stale state", err: ErrUserStateChanged, want: http.StatusConflict},
		{name: "last manager", err: ErrLastActiveUserManager, want: http.StatusConflict},
		{name: "internal", err: errors.New("sensitive database detail"), want: http.StatusInternalServerError},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			recorder := httptest.NewRecorder()
			writeUserMutationError(recorder, test.err, "update")
			if recorder.Code != test.want {
				t.Fatalf("status = %d, want %d", recorder.Code, test.want)
			}
			if test.want == http.StatusInternalServerError &&
				recorder.Body.String() != "user update failed\n" {
				t.Fatalf("internal response = %q", recorder.Body.String())
			}
		})
	}
}

func TestAdminUserHandlersDoNotMisclassifyOrLeakLookupFailures(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	if err := svc.store.Close(); err != nil {
		t.Fatalf("close auth store: %v", err)
	}
	actor := AuthContext{
		UserID: "actor",
		Perms:  map[string]bool{PermUsersManage: true},
	}
	tests := []struct {
		name   string
		method string
		serve  func(http.ResponseWriter, *http.Request)
	}{
		{
			name:   "get",
			method: http.MethodGet,
			serve: func(w http.ResponseWriter, r *http.Request) {
				handleAdminUserGet(w, r, svc, "target")
			},
		},
		{
			name:   "patch",
			method: http.MethodPatch,
			serve: func(w http.ResponseWriter, r *http.Request) {
				handleAdminUserPatch(w, r, svc, actor, "target")
			},
		},
		{
			name:   "delete",
			method: http.MethodDelete,
			serve: func(w http.ResponseWriter, r *http.Request) {
				handleAdminUserDelete(w, r, svc, actor, "target")
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(test.method, "/api/admin/users/target", nil)
			test.serve(recorder, request)
			if recorder.Code != http.StatusInternalServerError {
				t.Fatalf("status = %d, want 500", recorder.Code)
			}
			if recorder.Body.String() != "user lookup failed\n" {
				t.Fatalf("response = %q, want generic lookup failure", recorder.Body.String())
			}
		})
	}
}

func grantUserPermission(t *testing.T, svc *Service, userID, permission string) {
	t.Helper()
	if _, err := svc.store.db.ExecContext(
		context.Background(),
		`INSERT INTO user_permissions(user_id, permission_key, allowed) VALUES (?, ?, 1)`,
		userID,
		permission,
	); err != nil {
		t.Fatalf("grant direct permission: %v", err)
	}
}

func assertStoredUserState(t *testing.T, svc *Service, userID, role, status string) {
	t.Helper()
	user, err := svc.store.GetUserByID(context.Background(), userID)
	if err != nil {
		t.Fatalf("GetUserByID() error = %v", err)
	}
	if user.Role != role || user.Status != status {
		t.Fatalf("stored user state = (%q, %q), want (%q, %q)", user.Role, user.Status, role, status)
	}
}

func assertActiveManagerCount(t *testing.T, svc *Service, want int) {
	t.Helper()
	count, err := svc.store.CountActiveUsersWithPermission(
		context.Background(),
		PermUsersManage,
		"",
	)
	if err != nil {
		t.Fatalf("CountActiveUsersWithPermission() error = %v", err)
	}
	if count != want {
		t.Fatalf("active manager count = %d, want %d", count, want)
	}
}
