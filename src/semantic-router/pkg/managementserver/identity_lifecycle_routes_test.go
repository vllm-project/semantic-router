package managementserver

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	lifecyclePrincipalID = "90000000-0000-4000-8000-000000000001"
	lifecycleSessionID   = "90000000-0000-4000-8000-000000000002"
	lifecycleIssuerID    = "90000000-0000-4000-8000-000000000003"
)

type lifecycleServiceStub struct {
	logoutIssuerID string
	logoutToken    string
	logoutResult   managementidentity.BackchannelLogoutResult
	logoutErr      error
	selfPrincipal  string
	selfSession    string
	selfMutation   managementauth.SessionMutation
	adminSession   managementidentity.SessionRevocationCommand
	updateIssuer   managementidentity.UpdateTrustedIdentityIssuer
}

func (*lifecycleServiceStub) Ready(context.Context) error { return nil }
func (*lifecycleServiceStub) Me(context.Context, managementauth.AuthenticatedSession) (managementidentity.SelfView, error) {
	return managementidentity.SelfView{}, errors.New("not called")
}

func (*lifecycleServiceStub) ListManagementSessions(context.Context, string, managementidentity.ListRequest) (managementidentity.ManagementSessionPage, error) {
	return managementidentity.ManagementSessionPage{}, errors.New("not called")
}

func (stub *lifecycleServiceStub) RevokeSelfManagementSession(_ context.Context, principalID, sessionID string, _ managementidentity.MutationActor) (managementauth.SessionMutation, error) {
	stub.selfPrincipal, stub.selfSession = principalID, sessionID
	return stub.selfMutation, nil
}

func (stub *lifecycleServiceStub) RevokeManagementSession(_ context.Context, request managementidentity.SessionRevocationCommand) (managementauth.SessionMutation, managementidentity.MutationResult, error) {
	stub.adminSession = request
	return managementauth.SessionMutation{
			SessionID: request.SessionID, Changed: true,
			ChangedAt: time.Date(2026, 8, 23, 3, 4, 5, 0, time.UTC),
		}, managementidentity.MutationResult{
			Kind: "management_session", ID: request.SessionID, Revision: 1, ResponseStatus: http.StatusOK,
		}, nil
}

func (*lifecycleServiceStub) RevokePrincipalManagementSessions(context.Context, managementidentity.PrincipalSessionRevocationCommand) (managementidentity.PrincipalSessionRevocation, error) {
	return managementidentity.PrincipalSessionRevocation{}, errors.New("not called")
}

func (*lifecycleServiceStub) GetTrustedIdentityIssuer(context.Context, string) (managementidentity.TrustedIdentityIssuer, error) {
	return managementidentity.TrustedIdentityIssuer{}, errors.New("not called")
}

func (*lifecycleServiceStub) ListTrustedIdentityIssuers(context.Context, managementidentity.ListRequest) (managementidentity.TrustedIdentityIssuerPage, error) {
	return managementidentity.TrustedIdentityIssuerPage{}, errors.New("not called")
}

func (*lifecycleServiceStub) CreateTrustedIdentityIssuer(context.Context, managementidentity.CreateTrustedIdentityIssuer) (managementidentity.IssuerMutation, error) {
	return managementidentity.IssuerMutation{}, errors.New("not called")
}

func (stub *lifecycleServiceStub) UpdateTrustedIdentityIssuer(_ context.Context, request managementidentity.UpdateTrustedIdentityIssuer) (managementidentity.IssuerMutation, error) {
	stub.updateIssuer = request
	return managementidentity.IssuerMutation{Result: managementidentity.MutationResult{
		Kind: "trusted_identity_issuer", ID: request.ID, Revision: request.ExpectedRevision + 1,
		ResponseStatus: http.StatusOK,
	}}, nil
}

func (*lifecycleServiceStub) DeleteTrustedIdentityIssuer(context.Context, string, uint64, managementidentity.MutationActor) (managementidentity.IssuerMutation, error) {
	return managementidentity.IssuerMutation{}, errors.New("not called")
}

func (*lifecycleServiceStub) RefreshTrustedIdentityIssuer(context.Context, managementidentity.RefreshTrustedIdentityIssuer) (managementidentity.IssuerMutation, error) {
	return managementidentity.IssuerMutation{}, errors.New("not called")
}

func (stub *lifecycleServiceStub) BackchannelLogout(_ context.Context, issuerID, token, _ string, _ time.Time) (managementidentity.BackchannelLogoutResult, error) {
	stub.logoutIssuerID, stub.logoutToken = issuerID, token
	return stub.logoutResult, stub.logoutErr
}

type lifecycleSessionStub struct{}

func (lifecycleSessionStub) Authenticate(context.Context, string, string, time.Time) (managementauth.AuthenticatedSession, error) {
	return managementauth.AuthenticatedSession{
		Claims: managementauth.Claims{Subject: lifecyclePrincipalID, SessionID: lifecycleSessionID},
		Session: managementauth.LiveSession{Session: managementauth.Session{
			ID: lifecycleSessionID, PrincipalID: lifecyclePrincipalID,
		}},
	}, nil
}

func TestIdentityBackchannelLogoutUsesOnlySignedBodyCredential(t *testing.T) {
	service := &lifecycleServiceStub{logoutResult: managementidentity.BackchannelLogoutResult{Replayed: true}}
	routes := newLifecycleRoutesForTest(t, service)
	mux := http.NewServeMux()
	routes.Register(mux)
	request := httptest.NewRequest(http.MethodPost, backchannelLogoutPath,
		strings.NewReader(`{"issuerId":"`+lifecycleIssuerID+`","logoutToken":"signed.logout.token"}`))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, request)
	if response.Code != http.StatusOK || service.logoutIssuerID != lifecycleIssuerID || service.logoutToken != "signed.logout.token" ||
		!strings.Contains(response.Body.String(), `"replayed":true`) {
		t.Fatalf("status=%d body=%s issuer=%q token=%q", response.Code, response.Body.String(), service.logoutIssuerID, service.logoutToken)
	}

	request = httptest.NewRequest(http.MethodPost, backchannelLogoutPath,
		strings.NewReader(`{"issuerId":"`+lifecycleIssuerID+`","logoutToken":"signed.logout.token"}`))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set("Authorization", "Bearer forbidden")
	response = httptest.NewRecorder()
	mux.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("bearer-authenticated logout status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestIdentitySelfSessionRevokeBindsAuthenticatedOwner(t *testing.T) {
	revokedAt := time.Date(2026, 8, 23, 3, 4, 5, 0, time.UTC)
	service := &lifecycleServiceStub{selfMutation: managementauth.SessionMutation{
		SessionID: lifecycleSessionID, Changed: true, ChangedAt: revokedAt,
	}}
	routes := newLifecycleRoutesForTest(t, service)
	mux := http.NewServeMux()
	routes.Register(mux)
	request := httptest.NewRequest(http.MethodDelete, selfManagementSessionPath+"/"+lifecycleSessionID, nil)
	request.Header.Set("Authorization", "Bearer management-token")
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, request)
	if response.Code != http.StatusNoContent || service.selfPrincipal != lifecyclePrincipalID || service.selfSession != lifecycleSessionID ||
		response.Body.Len() != 0 {
		t.Fatalf("status=%d body=%s principal=%q session=%q", response.Code, response.Body.String(), service.selfPrincipal, service.selfSession)
	}
}

func TestIdentityAdminSessionRevokeParsesActionPathAndIdempotency(t *testing.T) {
	service := &lifecycleServiceStub{}
	routes := newLifecycleRoutesForTest(t, service)
	mux := http.NewServeMux()
	routes.Register(mux)
	request := httptest.NewRequest(http.MethodPost, managementSessionPath+"/"+lifecycleSessionID+":revoke",
		strings.NewReader(`{"reason":"incident response"}`))
	request.Header.Set("Authorization", "Bearer management-token")
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set(managementapi.HeaderIdempotencyKey, "session-revoke-0001")
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, request)
	if response.Code != http.StatusOK || service.adminSession.SessionID != lifecycleSessionID ||
		service.adminSession.Actor.Reason != "incident response" {
		t.Fatalf("status=%d body=%s request=%+v", response.Code, response.Body.String(), service.adminSession)
	}
}

func TestIdentityIssuerPatchRequiresCASAndStrictBody(t *testing.T) {
	service := &lifecycleServiceStub{}
	routes := newLifecycleRoutesForTest(t, service)
	mux := http.NewServeMux()
	routes.Register(mux)
	target := trustedIssuerPath + "/" + lifecycleIssuerID
	for _, test := range []struct {
		name    string
		body    string
		ifMatch string
		status  int
	}{
		{name: "missing CAS", body: `{"status":"disabled","reason":"incident"}`, status: http.StatusPreconditionRequired},
		{name: "unknown field", body: `{"status":"disabled","reason":"incident","issuer":"https://other.example"}`, ifMatch: `"tii:7"`, status: http.StatusBadRequest},
		{name: "valid", body: `{"status":"disabled","reason":"incident"}`, ifMatch: `"tii:7"`, status: http.StatusOK},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPatch, target, strings.NewReader(test.body))
			request.Header.Set("Authorization", "Bearer management-token")
			request.Header.Set("Content-Type", managementapi.JSONMediaType)
			if test.ifMatch != "" {
				request.Header.Set(managementapi.HeaderIfMatch, test.ifMatch)
			}
			response := httptest.NewRecorder()
			mux.ServeHTTP(response, request)
			if response.Code != test.status {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
		})
	}
	if service.updateIssuer.ID != lifecycleIssuerID || service.updateIssuer.ExpectedRevision != 7 ||
		service.updateIssuer.Status == nil || *service.updateIssuer.Status != managementauth.ResourceDisabled {
		t.Fatalf("issuer update request = %+v", service.updateIssuer)
	}
}

func newLifecycleRoutesForTest(t *testing.T, service IdentityLifecycleService) *IdentityLifecycleRoutes {
	t.Helper()
	codec, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1", Keys: map[string][]byte{"v1": bytes.Repeat([]byte{0x71}, 32)},
	})
	if err != nil {
		t.Fatal(err)
	}
	routes, err := NewIdentityLifecycleRoutes(IdentityLifecycleRoutesOptions{
		Service: service, Sessions: lifecycleSessionStub{},
		Authorization: AuthorizerFunc(func(context.Context, AuthorizationRequest) (AuthorizationDecision, error) {
			return AuthorizationDecision{AuthorityDigest: "sha256:test"}, nil
		}),
		Commands: codec, AllowPlaintextForTests: true,
		Now: func() time.Time { return time.Date(2026, 8, 23, 3, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatal(err)
	}
	return routes
}
