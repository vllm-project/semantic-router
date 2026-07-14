package auth

import (
	"context"
	"database/sql"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestRevokedAdminCannotFinishBlockedAccountMutation(t *testing.T) {
	t.Run("create user", func(t *testing.T) {
		svc, actor, token := newMutationActor(t)
		body := `{"email":"created-after-revoke@example.com","name":"Blocked","password":"created user password value 2026","role":"read"}`
		recorder := runBlockedAdminMutation(
			t,
			svc,
			actor,
			token,
			http.MethodPost,
			"/api/admin/users",
			body,
			adminUsersCollectionHandler(svc),
		)
		assertAuthorizationChangedResponse(t, recorder)
		if _, _, _, _, _, _, _, _, _, err := svc.store.GetUserByEmail(
			context.Background(),
			"created-after-revoke@example.com",
		); !errors.Is(err, sql.ErrNoRows) {
			t.Fatalf("revoked actor created a user; lookup error = %v", err)
		}
	})

	t.Run("reset password", func(t *testing.T) {
		svc, actor, token := newMutationActor(t)
		target := newTestUser(t, svc, "reset-after-revoke@example.com", RoleRead, defaultUserStatusActive)
		body := `{"userId":"` + target.ID + `","password":"reset after revoke password 2026"}`
		recorder := runBlockedAdminMutation(
			t,
			svc,
			actor,
			token,
			http.MethodPost,
			"/api/admin/users/password",
			body,
			adminUserPasswordHandler(svc),
		)
		assertAuthorizationChangedResponse(t, recorder)
		_, hash, err := svc.store.GetUserWithPasswordHash(context.Background(), target.ID)
		if err != nil {
			t.Fatalf("read target password: %v", err)
		}
		if !svc.VerifyPassword(hash, "secret-password") ||
			svc.VerifyPassword(hash, "reset after revoke password 2026") {
			t.Fatal("revoked actor changed the target password")
		}
	})

	t.Run("patch user", func(t *testing.T) {
		svc, actor, token := newMutationActor(t)
		target := newTestUser(t, svc, "patch-after-revoke@example.com", RoleRead, defaultUserStatusActive)
		recorder := runBlockedAdminMutation(
			t,
			svc,
			actor,
			token,
			http.MethodPatch,
			"/api/admin/users/"+target.ID,
			`{"role":"write"}`,
			adminUserItemHandler(svc),
		)
		assertAuthorizationChangedResponse(t, recorder)
		assertStoredUserState(t, svc, target.ID, RoleRead, defaultUserStatusActive)
	})
}

func TestRevokedAdminCannotDeleteUserWithStaleAuthorization(t *testing.T) {
	t.Parallel()

	svc, actor, token := newMutationActor(t)
	target := newTestUser(t, svc, "delete-after-revoke@example.com", RoleRead, defaultUserStatusActive)
	claims, err := svc.ParseToken(token)
	if err != nil {
		t.Fatalf("ParseToken() error = %v", err)
	}
	deactivateMutationActor(t, svc, actor.ID)

	err = svc.store.deleteUserAuthorizedIfCurrent(
		context.Background(),
		usersManageAuthorization(AuthContext{
			UserID:    actor.ID,
			SessionID: claims.ID,
		}),
		target.ID,
		target.Role,
		target.Status,
	)
	if !errors.Is(err, ErrAuthorizationChanged) {
		t.Fatalf("delete error = %v, want ErrAuthorizationChanged", err)
	}
	assertStoredUserState(t, svc, target.ID, RoleRead, defaultUserStatusActive)
}

func newMutationActor(t *testing.T) (*Service, *User, string) {
	t.Helper()
	svc := newTestAuthService(t)
	actor := newTestUser(t, svc, "mutation-actor@example.com", RoleAdmin, defaultUserStatusActive)
	newTestUser(t, svc, "mutation-backup@example.com", RoleAdmin, defaultUserStatusActive)
	token, err := svc.issueToken(actor)
	if err != nil {
		t.Fatalf("issue actor token: %v", err)
	}
	return svc, actor, token
}

func runBlockedAdminMutation(
	t *testing.T,
	svc *Service,
	actor *User,
	token string,
	method string,
	path string,
	body string,
	handler http.Handler,
) *httptest.ResponseRecorder {
	t.Helper()
	bodyGate := newGatedRequestBody(body)
	request := httptest.NewRequest(method, path, nil)
	request.Body = bodyGate
	request.ContentLength = int64(len(body))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+token)
	recorder := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		AuthenticateRequest(svc)(handler).ServeHTTP(recorder, request)
		close(done)
	}()

	select {
	case <-bodyGate.entered:
	case <-time.After(20 * time.Second):
		bodyGate.releaseRead()
		t.Fatal("handler did not reach the blocked request body")
	}
	deactivateMutationActor(t, svc, actor.ID)
	bodyGate.releaseRead()
	select {
	case <-done:
	case <-time.After(20 * time.Second):
		t.Fatal("handler did not finish after releasing the request body")
	}
	return recorder
}

func deactivateMutationActor(t *testing.T, svc *Service, actorID string) {
	t.Helper()
	if _, err := svc.store.UpdateUserRoleOrStatus(
		context.Background(),
		actorID,
		"",
		"inactive",
	); err != nil {
		t.Fatalf("deactivate mutation actor: %v", err)
	}
}

func assertAuthorizationChangedResponse(t *testing.T, recorder *httptest.ResponseRecorder) {
	t.Helper()
	if recorder.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401; body=%s", recorder.Code, recorder.Body.String())
	}
	if recorder.Body.String() != "Unauthorized\n" {
		t.Fatalf("response = %q, want generic Unauthorized", recorder.Body.String())
	}
}

type gatedRequestBody struct {
	reader  io.Reader
	entered chan struct{}
	release chan struct{}
	once    sync.Once
}

func newGatedRequestBody(body string) *gatedRequestBody {
	return &gatedRequestBody{
		reader:  strings.NewReader(body),
		entered: make(chan struct{}),
		release: make(chan struct{}),
	}
}

func (b *gatedRequestBody) Read(destination []byte) (int, error) {
	b.once.Do(func() { close(b.entered) })
	<-b.release
	return b.reader.Read(destination)
}

func (b *gatedRequestBody) Close() error {
	b.releaseRead()
	return nil
}

func (b *gatedRequestBody) releaseRead() {
	select {
	case <-b.release:
	default:
		close(b.release)
	}
}
