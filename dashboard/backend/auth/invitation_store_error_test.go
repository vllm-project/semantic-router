package auth

import (
	"bytes"
	"fmt"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestPublicInvitationRoutesReportStoreFailures(t *testing.T) {
	for _, failure := range []struct {
		name       string
		breakStore func(*testing.T, *Service)
		message    string
	}{
		{
			name: "closed database",
			breakStore: func(t *testing.T, svc *Service) {
				if err := svc.store.Close(); err != nil {
					t.Fatal(err)
				}
			},
			message: "sql: database is closed",
		},
		{
			name: "missing invitations table",
			breakStore: func(t *testing.T, svc *Service) {
				if _, err := svc.store.db.Exec("DROP TABLE dashboard_invitations"); err != nil {
					t.Fatal(err)
				}
			},
			message: "no such table: dashboard_invitations",
		},
	} {
		t.Run(failure.name, func(t *testing.T) {
			svc, _, token := newStoreErrorTestInvitation(t)
			failure.breakStore(t, svc)
			for _, request := range []struct {
				method string
				path   string
				body   string
			}{
				{http.MethodGet, "/api/auth/invitations/" + token, ""},
				{http.MethodPost, "/api/auth/invitations/" + token + "/accept", `{"password":"secret-password"}`},
			} {
				t.Run(request.method, func(t *testing.T) {
					var logs bytes.Buffer
					previous := log.Writer()
					log.SetOutput(&logs)
					defer log.SetOutput(previous)
					response := httptest.NewRecorder()
					AuthRoutes(svc).ServeHTTP(response, httptest.NewRequest(request.method, request.path, strings.NewReader(request.body)))
					if response.Code != http.StatusInternalServerError {
						t.Errorf("status = %d, want 500; body = %q", response.Code, response.Body.String())
					}
					if want := "failed to process invitation\n"; response.Body.String() != want {
						t.Errorf("body = %q, want %q", response.Body.String(), want)
					}
					if strings.Contains(response.Body.String(), failure.message) {
						t.Errorf("body leaks store error: %q", response.Body.String())
					}
					if !strings.Contains(logs.String(), failure.message) {
						t.Errorf("log = %q, want store error %q", logs.String(), failure.message)
					}
				})
			}
		})
	}
}

func newStoreErrorTestInvitation(t *testing.T) (*Service, *Invitation, string) {
	t.Helper()
	svc := newTestAuthService(t)
	admin, err := svc.store.CreateUser(t.Context(), "admin@example.com", "Admin", "unused", RoleAdmin, "active")
	if err != nil {
		t.Fatal(err)
	}
	item, token, err := svc.CreateInvitation(t.Context(), InvitationSpec{
		Email: "invited@example.com", Name: "Invited User", Role: RoleRead,
	}, admin.ID)
	if err != nil {
		t.Fatal(err)
	}
	return svc, item, token
}

func TestPublicInvitationRoutesKeepUnavailableResponse(t *testing.T) {
	for _, state := range []string{"missing", "expired", "revoked", "accepted"} {
		t.Run(state, func(t *testing.T) {
			svc, item, token := newStoreErrorTestInvitation(t)
			switch state {
			case "missing":
				token = "unknown-token"
			case "expired":
				if _, err := svc.store.db.Exec("UPDATE dashboard_invitations SET expires_at=0 WHERE id=?", item.ID); err != nil {
					t.Fatal(err)
				}
			default:
				if _, err := svc.store.db.Exec("UPDATE dashboard_invitations SET status=? WHERE id=?", state, item.ID); err != nil {
					t.Fatal(err)
				}
			}
			for _, method := range []string{http.MethodGet, http.MethodPost} {
				path := "/api/auth/invitations/" + token
				if method == http.MethodPost {
					path += "/accept"
				}
				response := httptest.NewRecorder()
				AuthRoutes(svc).ServeHTTP(response, httptest.NewRequest(method, path, strings.NewReader(`{"password":"secret-password"}`)))
				if response.Code != http.StatusGone || response.Body.String() != "invitation is no longer available\n" {
					t.Errorf("%s: status = %d, body = %q; want 410 with unavailable message", method, response.Code, response.Body.String())
				}
			}
		})
	}
}

func TestInvitationValidationErrorsKeepSafeMessages(t *testing.T) {
	for _, validationErr := range []error{ErrInvalidInvitationEmail, ErrInvalidInvitationName, ErrInvitationUserExists} {
		t.Run(validationErr.Error(), func(t *testing.T) {
			response := httptest.NewRecorder()
			writeInvitationError(response, fmt.Errorf("internal details: %w", validationErr))
			if response.Code != http.StatusBadRequest || response.Body.String() != validationErr.Error()+"\n" {
				t.Fatalf("status = %d, body = %q; want 400 with %q", response.Code, response.Body.String(), validationErr)
			}
		})
	}
}
