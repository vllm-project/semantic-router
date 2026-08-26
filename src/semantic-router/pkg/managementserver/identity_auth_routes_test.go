package managementserver

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

type identityAuthenticationStub struct{}

func (identityAuthenticationStub) Ready(context.Context) error { return nil }
func (identityAuthenticationStub) CreateChallenge(context.Context, string, string) (managementauth.ExchangeChallenge, error) {
	return managementauth.ExchangeChallenge{}, errors.New("not called")
}

func (identityAuthenticationStub) Exchange(context.Context, string, string, string, managementauth.SubjectTokenType, string, string) (managementauth.IdentityExchangeResult, error) {
	return managementauth.IdentityExchangeResult{}, errors.New("not called")
}

func (identityAuthenticationStub) ServiceToken(context.Context, string) (managementauth.IssuedToken, error) {
	return managementauth.IssuedToken{}, errors.New("not called")
}

func (identityAuthenticationStub) MTLSToken(context.Context, managementauth.VerifiedMTLSEvidence) (managementauth.IssuedToken, error) {
	return managementauth.IssuedToken{}, errors.New("not called")
}

type exchangeAuthenticationStub struct {
	invitationToken string
	rateIdentity    string
	result          managementauth.IdentityExchangeResult
	err             error
	calls           int
}

type challengeAuthenticationStub struct {
	identityAuthenticationStub
	challenge    managementauth.ExchangeChallenge
	issuerID     string
	rateIdentity string
	err          error
}

func (stub *challengeAuthenticationStub) CreateChallenge(
	_ context.Context,
	issuerID string,
	rateIdentity string,
) (managementauth.ExchangeChallenge, error) {
	stub.issuerID = issuerID
	stub.rateIdentity = rateIdentity
	return stub.challenge, stub.err
}

func (stub *exchangeAuthenticationStub) Ready(context.Context) error { return nil }
func (stub *exchangeAuthenticationStub) CreateChallenge(context.Context, string, string) (managementauth.ExchangeChallenge, error) {
	return managementauth.ExchangeChallenge{}, errors.New("not called")
}

func (stub *exchangeAuthenticationStub) Exchange(_ context.Context, _, _, rateIdentity string,
	_ managementauth.SubjectTokenType, _ string, invitationToken string,
) (managementauth.IdentityExchangeResult, error) {
	stub.calls++
	stub.invitationToken = invitationToken
	stub.rateIdentity = rateIdentity
	return stub.result, stub.err
}

func (stub *exchangeAuthenticationStub) ServiceToken(context.Context, string) (managementauth.IssuedToken, error) {
	return managementauth.IssuedToken{}, errors.New("not called")
}

func (stub *exchangeAuthenticationStub) MTLSToken(context.Context, managementauth.VerifiedMTLSEvidence) (managementauth.IssuedToken, error) {
	return managementauth.IssuedToken{}, errors.New("not called")
}

type bootstrapStub struct {
	request managementidentity.BootstrapRequest
	token   string
	result  managementidentity.BootstrapResult
	err     error
}

type recoveryStub struct {
	request managementidentity.RecoveryRequest
	token   string
	result  managementidentity.RecoveryResult
	err     error
}

func (stub *recoveryStub) Ready(context.Context) error { return nil }
func (stub *recoveryStub) Recover(_ context.Context, request managementidentity.RecoveryRequest, token string) (managementidentity.RecoveryResult, error) {
	stub.request = request
	stub.token = token
	return stub.result, stub.err
}

func (stub *bootstrapStub) Ready(context.Context) error { return nil }
func (stub *bootstrapStub) Bootstrap(_ context.Context, request managementidentity.BootstrapRequest, token string) (managementidentity.BootstrapResult, error) {
	stub.request = request
	stub.token = token
	return stub.result, stub.err
}

func TestIdentityExchangeChallengeReturnsCreated(t *testing.T) {
	issuerID := "10000000-0000-4000-8000-000000000001"
	service := &challengeAuthenticationStub{challenge: managementauth.ExchangeChallenge{
		ID: "10000000-0000-4000-8000-000000000002", Nonce: "one-time-nonce",
		ExpiresAt: time.Date(2026, time.August, 23, 12, 1, 0, 0, time.UTC),
	}}
	routes, err := NewIdentityAuthRoutes(IdentityAuthRoutesOptions{
		Service: service, Bootstrap: &bootstrapStub{}, AllowPlaintextForTests: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, managementapi.BasePath+"/auth/exchange-challenges",
		strings.NewReader(`{"issuerId":"`+issuerID+`"}`))
	request.RemoteAddr = "192.0.2.1:12345"
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set("Accept", managementapi.JSONMediaType)
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)

	if response.Code != http.StatusCreated || service.issuerID != issuerID || service.rateIdentity != "192.0.2.1" {
		t.Fatalf("status=%d issuer=%q rateIdentity=%q body=%s", response.Code, service.issuerID, service.rateIdentity, response.Body.String())
	}
	if response.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("Cache-Control = %q", response.Header().Get("Cache-Control"))
	}
}

func TestIdentityBootstrapRouteReturnsOnlyOneTimeSecretEnvelope(t *testing.T) {
	bootstrap := &bootstrapStub{result: managementidentity.BootstrapResult{
		PrincipalID: "10000000-0000-4000-8000-000000000001", RoleBindingID: "10000000-0000-4000-8000-000000000002",
		ServiceAccountID: "10000000-0000-4000-8000-000000000003", ServiceCredentialID: "10000000-0000-4000-8000-000000000004",
		ServiceCredential: "vsm_test_secret", ServiceCredentialExpiresAt: time.Date(2026, 9, 21, 0, 0, 0, 0, time.UTC),
		FinalizationRequired: true, Replayed: true, ResponseStatus: http.StatusCreated,
	}}
	routes, err := NewIdentityAuthRoutes(IdentityAuthRoutesOptions{
		Service: identityAuthenticationStub{}, Bootstrap: bootstrap, AllowPlaintextForTests: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, managementapi.BasePath+"/auth/bootstrap",
		strings.NewReader(`{"kind":"service_account","displayName":"First administrator"}`))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set("Authorization", "VSR-Bootstrap bootstrap-secret")
	request.Header.Set(managementapi.HeaderIdempotencyKey, "bootstrap-key-0001")
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)

	if response.Code != http.StatusCreated {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if response.Header().Get("Cache-Control") != "no-store" ||
		response.Header().Get(managementapi.HeaderIdempotencyReplayed) != "true" {
		t.Fatalf("unsafe response headers: %v", response.Header())
	}
	var payload managementapi.BootstrapResponse
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if payload.ServiceCredential == nil || payload.ServiceCredential.Secret != "vsm_test_secret" || payload.ServiceCredential.ExpiresAt == nil ||
		payload.ServiceCredential.Kind != managementapi.SecretKindServiceCredential {
		t.Fatalf("unexpected secret envelope: %+v", payload.ServiceCredential)
	}
	if bootstrap.token != "bootstrap-secret" || bootstrap.request.IdempotencyKey != "bootstrap-key-0001" ||
		bootstrap.request.Kind != managementidentity.BootstrapServiceAccount || len(bootstrap.request.CanonicalRequest) == 0 {
		t.Fatalf("unexpected bootstrap domain request: %+v", bootstrap.request)
	}
	for _, forbidden := range []string{"ciphertext", "nonce", "kekVersion", "secretHmac", "pepperVersion"} {
		if strings.Contains(response.Body.String(), forbidden) {
			t.Fatalf("bootstrap response exposed %q", forbidden)
		}
	}
}

func TestIdentityRecoveryRouteRestoresExistingPrincipalFromLoopback(t *testing.T) {
	recovery := &recoveryStub{result: managementidentity.RecoveryResult{
		PrincipalID:   "10000000-0000-4000-8000-000000000001",
		RoleBindingID: "10000000-0000-4000-8000-000000000002",
		Replayed:      true, ResponseStatus: http.StatusCreated,
	}}
	routes, err := NewIdentityAuthRoutes(IdentityAuthRoutesOptions{
		Service: identityAuthenticationStub{}, Bootstrap: &bootstrapStub{}, Recovery: recovery,
		AllowPlaintextForTests: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, managementapi.BasePath+"/auth/recovery", strings.NewReader(`{
  "principalId":"10000000-0000-4000-8000-000000000001",
  "reason":"Restore cluster administration after issuer lockout"
}`))
	request.RemoteAddr = "127.0.0.1:12345"
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set("Authorization", "VSR-Recovery recovery-secret")
	request.Header.Set(managementapi.HeaderIdempotencyKey, "recovery-key-0001")
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)

	if response.Code != http.StatusCreated || response.Header().Get(managementapi.HeaderIdempotencyReplayed) != "true" {
		t.Fatalf("status=%d headers=%v body=%s", response.Code, response.Header(), response.Body.String())
	}
	if recovery.token != "recovery-secret" || recovery.request.PrincipalID != recovery.result.PrincipalID ||
		recovery.request.Reason == "" || recovery.request.RequestID == "" || len(recovery.request.CanonicalRequest) == 0 {
		t.Fatalf("recovery request = %+v token=%q", recovery.request, recovery.token)
	}
	var payload managementapi.RecoveryResponse
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if payload.RoleBindingID != recovery.result.RoleBindingID || !payload.RecoveryDisableRequired {
		t.Fatalf("recovery payload = %+v", payload)
	}
}

func TestIdentityRecoveryRouteIsHiddenFromNonLoopbackClients(t *testing.T) {
	recovery := &recoveryStub{}
	routes, err := NewIdentityAuthRoutes(IdentityAuthRoutesOptions{
		Service: identityAuthenticationStub{}, Bootstrap: &bootstrapStub{}, Recovery: recovery,
		AllowPlaintextForTests: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, managementapi.BasePath+"/auth/recovery", strings.NewReader(`{}`))
	request.RemoteAddr = "192.0.2.1:12345"
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusNotFound || recovery.token != "" {
		t.Fatalf("status=%d token=%q body=%s", response.Code, recovery.token, response.Body.String())
	}
}

func TestIdentityRecoveryRouteIsRegisteredOnlyWhenConfigured(t *testing.T) {
	for _, test := range []struct {
		name       string
		recovery   RecoveryService
		wantStatus int
	}{
		{name: "disabled", wantStatus: http.StatusNotFound},
		{name: "enabled", recovery: &recoveryStub{}, wantStatus: http.StatusUnauthorized},
	} {
		t.Run(test.name, func(t *testing.T) {
			routes, err := NewIdentityAuthRoutes(IdentityAuthRoutesOptions{
				Service: identityAuthenticationStub{}, Bootstrap: &bootstrapStub{}, Recovery: test.recovery,
				AllowPlaintextForTests: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			mux := http.NewServeMux()
			routes.Register(mux)
			request := httptest.NewRequest(http.MethodPost, managementapi.BasePath+"/auth/recovery", strings.NewReader(`{}`))
			request.RemoteAddr = "127.0.0.1:12345"
			response := httptest.NewRecorder()
			mux.ServeHTTP(response, request)
			if response.Code != test.wantStatus {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
		})
	}
}

func TestDisabledRecoveryRouteRemainsAbsentThroughManagementTransport(t *testing.T) {
	routes, err := NewIdentityAuthRoutes(IdentityAuthRoutesOptions{
		Service: identityAuthenticationStub{}, Bootstrap: &bootstrapStub{},
		AllowPlaintextForTests: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	server, err := NewServer(&catalogRuntimeStub{}, routes)
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	server.Register(mux)
	request := httptest.NewRequest(http.MethodPost, managementapi.BasePath+"/auth/recovery", strings.NewReader(`{}`))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set("Accept", managementapi.JSONMediaType)
	request.RemoteAddr = "127.0.0.1:12345"
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, request)
	if response.Code != http.StatusNotFound {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestIdentityBootstrapRouteRejectsUnknownFieldsBeforeSecretUse(t *testing.T) {
	bootstrap := &bootstrapStub{}
	routes, err := NewIdentityAuthRoutes(IdentityAuthRoutesOptions{
		Service: identityAuthenticationStub{}, Bootstrap: bootstrap, AllowPlaintextForTests: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, managementapi.BasePath+"/auth/bootstrap",
		strings.NewReader(`{"kind":"service_account","displayName":"Administrator","unknown":true}`))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set("Authorization", "VSR-Bootstrap bootstrap-secret")
	request.Header.Set(managementapi.HeaderIdempotencyKey, "bootstrap-key-0002")
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest || bootstrap.token != "" {
		t.Fatalf("status = %d, bootstrap called with token %q", response.Code, bootstrap.token)
	}
}

func TestIdentityBootstrapRouteMapsExpiredReplayToGone(t *testing.T) {
	bootstrap := &bootstrapStub{err: managementidentity.ErrBootstrapResultExpired}
	routes, err := NewIdentityAuthRoutes(IdentityAuthRoutesOptions{
		Service: identityAuthenticationStub{}, Bootstrap: bootstrap, AllowPlaintextForTests: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, managementapi.BasePath+"/auth/bootstrap",
		strings.NewReader(`{"kind":"service_account","displayName":"Administrator"}`))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set("Authorization", "VSR-Bootstrap bootstrap-secret")
	request.Header.Set(managementapi.HeaderIdempotencyKey, "bootstrap-key-0003")
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusGone {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
}

func TestIdentityTokenExchangeReturnsNestedOneTimeOnboarding(t *testing.T) {
	stub := &exchangeAuthenticationStub{result: managementauth.IdentityExchangeResult{
		Issued: managementauth.IssuedToken{
			AccessToken: "management-token", TokenType: "Bearer",
			ExpiresIn: time.Minute, ManagementSessionID: "10000000-0000-4000-8000-000000000004",
		},
		Onboarding: &managementauth.InvitationOnboarding{
			InvitationID: "10000000-0000-4000-8000-000000000005",
			PrincipalID:  "10000000-0000-4000-8000-000000000006",
			UserID:       "10000000-0000-4000-8000-000000000007",
			APIKeyID:     "10000000-0000-4000-8000-000000000008",
			APIKey:       "vsk_one_time", DeliveryExpiresAt: time.Now().UTC().Add(time.Minute),
		},
	}}
	routes, err := NewIdentityAuthRoutes(IdentityAuthRoutesOptions{
		Service: stub, Bootstrap: &bootstrapStub{}, AllowPlaintextForTests: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, managementapi.BasePath+"/auth/token-exchange", strings.NewReader(`{
  "issuerId":"10000000-0000-4000-8000-000000000001",
  "exchangeChallengeId":"10000000-0000-4000-8000-000000000002",
  "subjectToken":"verified-assertion",
  "subjectTokenType":"oidc_id_token",
  "invitationToken":"vsi_10000000-0000-4000-8000-000000000003_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
}`))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.RemoteAddr = "192.0.2.9:12345"
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusOK || stub.calls != 1 || stub.invitationToken == "" || stub.rateIdentity != "192.0.2.9" ||
		response.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("exchange status=%d calls=%d headers=%v body=%s", response.Code, stub.calls, response.Header(), response.Body.String())
	}
	var payload managementapi.TokenExchangeResponse
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if payload.Onboarding == nil || payload.Onboarding.APIKey != "vsk_one_time" ||
		payload.ManagementSessionID != stub.result.Issued.ManagementSessionID {
		t.Fatalf("exchange payload = %+v", payload)
	}
}

func TestIdentityChallengeCapacityReturnsRetryable429(t *testing.T) {
	stub := &challengeAuthenticationStub{err: &managementauth.ChallengeCapacityError{RetryAfter: 90 * time.Second}}
	routes, err := NewIdentityAuthRoutes(IdentityAuthRoutesOptions{
		Service: stub, Bootstrap: &bootstrapStub{}, AllowPlaintextForTests: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, managementapi.BasePath+"/auth/exchange-challenges",
		strings.NewReader(`{"issuerId":"10000000-0000-4000-8000-000000000001"}`))
	request.RemoteAddr = "192.0.2.9:12345"
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusTooManyRequests || response.Header().Get("Retry-After") != "90" ||
		!strings.Contains(response.Body.String(), "challenge_capacity_exceeded") {
		t.Fatalf("status=%d headers=%v body=%s", response.Code, response.Header(), response.Body.String())
	}
}

func TestIdentityTokenExchangeRejectsEmptyInvitationVariant(t *testing.T) {
	stub := &exchangeAuthenticationStub{}
	routes, err := NewIdentityAuthRoutes(IdentityAuthRoutesOptions{
		Service: stub, Bootstrap: &bootstrapStub{}, AllowPlaintextForTests: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, managementapi.BasePath+"/auth/token-exchange", strings.NewReader(`{
  "issuerId":"10000000-0000-4000-8000-000000000001",
  "exchangeChallengeId":"10000000-0000-4000-8000-000000000002",
  "subjectToken":"verified-assertion",
  "subjectTokenType":"oidc_id_token",
  "invitationToken":""
}`))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest || stub.calls != 0 {
		t.Fatalf("exchange status=%d calls=%d body=%s", response.Code, stub.calls, response.Body.String())
	}
}
