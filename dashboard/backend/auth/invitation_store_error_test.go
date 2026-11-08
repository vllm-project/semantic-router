package auth

import (
	"bytes"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func newLookupTestInvitation(t *testing.T) (*Service, *Invitation, string) {
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

func recordInvitationRequest(svc *Service, token, method string) (*httptest.ResponseRecorder, string) {
	var logs bytes.Buffer
	previous := log.Writer()
	log.SetOutput(&logs)
	defer log.SetOutput(previous)
	path := "/api/auth/invitations/" + token
	if method == http.MethodPost {
		path += "/accept"
	}
	response := httptest.NewRecorder()
	AuthRoutes(svc).ServeHTTP(response, httptest.NewRequest(method, path, strings.NewReader(`{"password":"secret-password"}`)))
	return response, logs.String()
}

func TestPublicInvitationRoutesReportLookupStoreFailures(t *testing.T) {
	// Keep these tests sequential while capturing the package logger.
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
			svc, _, token := newLookupTestInvitation(t)
			failure.breakStore(t, svc)
			for _, method := range []string{http.MethodGet, http.MethodPost} {
				t.Run(method, func(t *testing.T) {
					response, logs := recordInvitationRequest(svc, token, method)
					if response.Code != http.StatusInternalServerError {
						t.Errorf("status = %d, want 500; body = %q", response.Code, response.Body.String())
					}
					if want := "failed to process invitation\n"; response.Body.String() != want {
						t.Errorf("body = %q, want %q", response.Body.String(), want)
					}
					assertNoDriverText(t, response.Body.String())
					if !strings.Contains(logs, failure.message) {
						t.Errorf("log = %q, want store error %q", logs, failure.message)
					}
				})
			}
		})
	}
}

func TestPublicInvitationRoutesKeepUnavailableResponse(t *testing.T) {
	for _, state := range []string{"missing", "expired", "revoked", "accepted", "exhausted"} {
		t.Run(state, func(t *testing.T) {
			svc, item, token := newLookupTestInvitation(t)
			switch state {
			case "missing":
				token = "unknown-token"
			case "expired":
				if _, err := svc.store.db.Exec("UPDATE dashboard_invitations SET expires_at=0 WHERE id=?", item.ID); err != nil {
					t.Fatal(err)
				}
			case "exhausted":
				if _, err := svc.store.db.Exec("UPDATE dashboard_invitations SET used_count=max_uses WHERE id=?", item.ID); err != nil {
					t.Fatal(err)
				}
			default:
				if _, err := svc.store.db.Exec("UPDATE dashboard_invitations SET status=? WHERE id=?", state, item.ID); err != nil {
					t.Fatal(err)
				}
			}
			for _, method := range []string{http.MethodGet, http.MethodPost} {
				response, logs := recordInvitationRequest(svc, token, method)
				if response.Code != http.StatusGone || response.Body.String() != "invitation is no longer available\n" {
					t.Errorf("%s: status = %d, body = %q; want 410 with unavailable message", method, response.Code, response.Body.String())
				}
				if logs != "" {
					t.Errorf("%s: unexpected diagnostic log for unavailable invitation: %q", method, logs)
				}
			}
		})
	}
}

func TestPublicInvitationRoutesAcceptAvailableInvitation(t *testing.T) {
	svc, _, token := newLookupTestInvitation(t)
	for _, method := range []string{http.MethodGet, http.MethodPost} {
		response, logs := recordInvitationRequest(svc, token, method)
		if response.Code != http.StatusOK {
			t.Errorf("%s: status = %d, body = %q; want 200", method, response.Code, response.Body.String())
		}
		if !strings.Contains(response.Body.String(), `"email":"invited@example.com"`) {
			t.Errorf("%s: body = %q, want invited user's email", method, response.Body.String())
		}
		if logs != "" {
			t.Errorf("%s: unexpected diagnostic log: %q", method, logs)
		}
	}
}
