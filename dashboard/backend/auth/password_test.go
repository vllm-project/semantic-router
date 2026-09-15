package auth

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestValidatePasswordMeasuresBytesNotCharacters(t *testing.T) {
	t.Parallel()

	// 25 CJK runes are 75 bytes, so a rune count would wrongly accept this.
	multibyte := strings.Repeat("好", 25)

	if len([]rune(multibyte)) > MaxPasswordBytes {
		t.Fatalf("test input has %d runes, expected it to be short in characters", len([]rune(multibyte)))
	}

	tests := []struct {
		name     string
		password string
		wantErr  error
	}{
		{name: "empty", password: "", wantErr: ErrPasswordTooShort},
		{name: "at the limit", password: strings.Repeat("a", MaxPasswordBytes), wantErr: nil},
		{name: "one byte over", password: strings.Repeat("a", MaxPasswordBytes+1), wantErr: ErrPasswordTooLong},
		{name: "short in runes but long in bytes", password: multibyte, wantErr: ErrPasswordTooLong},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			if err := ValidatePassword(tt.password); !errors.Is(err, tt.wantErr) {
				t.Fatalf("ValidatePassword() error = %v, want %v", err, tt.wantErr)
			}
		})
	}
}

func TestHashPasswordRejectsOversizedInputBeforeBcrypt(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)

	if _, err := svc.HashPassword(strings.Repeat("a", MaxPasswordBytes+1)); !errors.Is(err, ErrPasswordTooLong) {
		t.Fatalf("HashPassword() error = %v, want %v", err, ErrPasswordTooLong)
	}

	// Must not reject anything bcrypt would have accepted.
	if _, err := svc.HashPassword(strings.Repeat("a", MaxPasswordBytes)); err != nil {
		t.Fatalf("HashPassword() at the limit error = %v, want nil", err)
	}
}

func TestPasswordRotationRejectsOversizedPasswordWithBadRequest(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	admin := newTestUser(t, svc, "admin@example.com", RoleAdmin, "active")
	target := newTestUser(t, svc, "target@example.com", RoleRead, "active")

	mux := http.NewServeMux()
	RegisterAdminRoutes(mux, svc)
	handler := AuthenticateRequest(svc)(mux)

	body, err := json.Marshal(map[string]string{
		"userId":   target.ID,
		"password": strings.Repeat("a", MaxPasswordBytes+1),
	})
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}

	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, newAuthenticatedRequest(
		t, svc, admin, http.MethodPost, "/api/admin/users/password", string(body),
	))

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadRequest)
	}
	if got := recorder.Body.String(); !strings.Contains(got, "at most 72 bytes") {
		t.Fatalf("body = %q, want it to state the limit", got)
	}
	if got := recorder.Body.String(); strings.Contains(got, "bcrypt") {
		t.Fatalf("body = %q, want no bcrypt internals in the response", got)
	}
}

func TestPasswordRotationAcceptsPasswordAtTheLimit(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	admin := newTestUser(t, svc, "admin@example.com", RoleAdmin, "active")
	target := newTestUser(t, svc, "target@example.com", RoleRead, "active")

	mux := http.NewServeMux()
	RegisterAdminRoutes(mux, svc)
	handler := AuthenticateRequest(svc)(mux)

	password := strings.Repeat("a", MaxPasswordBytes)
	body, err := json.Marshal(map[string]string{"userId": target.ID, "password": password})
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}

	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, newAuthenticatedRequest(
		t, svc, admin, http.MethodPost, "/api/admin/users/password", string(body),
	))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d: %s", recorder.Code, http.StatusOK, recorder.Body.String())
	}

	if _, _, _, _, _, _, _, _, hash, err := svc.store.GetUserByEmail(t.Context(), target.Email); err != nil {
		t.Fatalf("GetUserByEmail() error = %v", err)
	} else if !svc.VerifyPassword(hash, password) {
		t.Fatal("stored hash does not verify against the new password")
	}
}

func TestInvitationAcceptanceRejectsOversizedPasswordWithBadRequest(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	admin := newTestUser(t, svc, "admin@example.com", RoleAdmin, "active")

	mux := http.NewServeMux()
	RegisterAdminRoutes(mux, svc)
	handler := AuthenticateRequest(svc)(mux)

	createBody, err := json.Marshal(map[string]string{
		"email": "new-user@example.com",
		"name":  "New User",
		"role":  RoleRead,
	})
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}

	createRecorder := httptest.NewRecorder()
	handler.ServeHTTP(createRecorder, newAuthenticatedRequest(
		t, svc, admin, http.MethodPost, "/api/admin/invitations", string(createBody),
	))
	if createRecorder.Code != http.StatusCreated {
		t.Fatalf("create invitation status = %d, want %d: %s", createRecorder.Code, http.StatusCreated, createRecorder.Body.String())
	}
	var created invitationMutationResponse
	if unmarshalErr := json.Unmarshal(createRecorder.Body.Bytes(), &created); unmarshalErr != nil {
		t.Fatalf("Unmarshal() error = %v", unmarshalErr)
	}

	body, err := json.Marshal(map[string]string{"password": strings.Repeat("a", MaxPasswordBytes+1)})
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	recorder := httptest.NewRecorder()
	publicInvitationHandler(svc).ServeHTTP(recorder, httptest.NewRequest(
		http.MethodPost,
		"/api/auth/invitations/"+created.Token+"/accept",
		strings.NewReader(string(body)),
	))

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadRequest)
	}
}
