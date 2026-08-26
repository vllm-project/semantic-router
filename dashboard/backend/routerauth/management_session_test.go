package routerauth

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

type recordingAssertionSigner struct {
	mu     sync.Mutex
	claims jwt.MapClaims
}

func (*recordingAssertionSigner) KeyID() string { return "dashboard-key-1" }
func (*recordingAssertionSigner) PublicJWK() PublicJWK {
	return PublicJWK{KeyType: "OKP", Curve: "Ed25519", Algorithm: "EdDSA", KeyID: "dashboard-key-1", X: "public-key"}
}

func (signer *recordingAssertionSigner) Sign(claims jwt.Claims) (string, error) {
	value, ok := claims.(jwt.MapClaims)
	if !ok {
		return "", errors.New("unexpected claim type")
	}
	signer.mu.Lock()
	signer.claims = value
	signer.mu.Unlock()
	return "signed-dashboard-assertion", nil
}

func TestManagementSessionProviderExchangesPrincipalAssertionAndCachesToken(t *testing.T) {
	t.Parallel()
	now := time.Date(2026, time.August, 23, 12, 0, 0, 0, time.UTC)
	issuerID := "10000000-0000-4000-8000-000000000001"
	challengeID := "10000000-0000-4000-8000-000000000002"
	var challengeCalls, exchangeCalls int
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Content-Type") != managementMediaType || request.Header.Get("Accept") != managementMediaType {
			t.Errorf("Management media headers = %v", request.Header)
		}
		response.Header().Set("Content-Type", managementMediaType)
		switch request.URL.Path {
		case managementBasePath + "/auth/exchange-challenges":
			challengeCalls++
			response.WriteHeader(http.StatusCreated)
			_ = json.NewEncoder(response).Encode(map[string]any{
				"exchangeChallengeId": challengeID, "nonce": "router-nonce",
				"expiresAt": now.Add(time.Minute), "futureMetadata": "ignored",
			})
		case managementBasePath + "/auth/token-exchange":
			exchangeCalls++
			var body map[string]string
			if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			if body["issuerId"] != issuerID || body["exchangeChallengeId"] != challengeID ||
				body["subjectToken"] != "signed-dashboard-assertion" || body["subjectTokenType"] != "router_local_assertion" {
				t.Errorf("exchange body = %#v", body)
			}
			_ = json.NewEncoder(response).Encode(managementTokenEnvelope{
				AccessToken: "principal-management-token", TokenType: "Bearer", ExpiresIn: 300,
				ManagementSessionID: "10000000-0000-4000-8000-000000000003",
			})
		default:
			http.NotFound(response, request)
		}
	}))
	defer server.Close()

	signer := &recordingAssertionSigner{}
	provider, err := NewManagementSessionProvider(ManagementSessionOptions{
		RouterURL: server.URL, IssuerURL: "https://dashboard.example.test", IssuerID: issuerID,
		Signer: signer, Client: server.Client(), Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	principal := dashboardauth.AuthContext{
		UserID: "dashboard-user-1", SessionID: "dashboard-session-1",
		Email: "user@example.test", Name: "User One",
		AuthenticatedAt: now.Add(-time.Minute),
		ExpiresAt:       now.Add(12 * time.Hour),
	}
	for index := 0; index < 2; index++ {
		token, tokenErr := provider.ManagementAccessToken(context.Background(), principal)
		if tokenErr != nil || token != "principal-management-token" {
			t.Fatalf("ManagementAccessToken() token=%q error=%v", token, tokenErr)
		}
	}
	credential, credentialErr := provider.(*managementSessionProvider).managementCredential(
		context.Background(), principal,
	)
	if credentialErr != nil || credential.managementSessionID != "10000000-0000-4000-8000-000000000003" {
		t.Fatalf("management session ID = %q, error = %v", credential.managementSessionID, credentialErr)
	}
	if challengeCalls != 1 || exchangeCalls != 1 {
		t.Fatalf("challenge calls=%d exchange calls=%d", challengeCalls, exchangeCalls)
	}
	signer.mu.Lock()
	claims := signer.claims
	signer.mu.Unlock()
	for key, want := range map[string]any{
		"iss": "https://dashboard.example.test", "sub": "dashboard-user-1",
		"aud": managementAudience, "nonce": "router-nonce", "sid": "dashboard-session-1",
		"email": "user@example.test", "email_verified": true, "name": "User One",
		"auth_time":             now.Add(-time.Minute).Unix(),
		"exp":                   now.Add(assertionLifetime).Unix(),
		routerSourceExpiryClaim: now.Add(12 * time.Hour).Unix(),
	} {
		if claims[key] != want {
			t.Errorf("claim %s = %#v, want %#v", key, claims[key], want)
		}
	}
	for _, required := range []string{"iat", "exp", "jti", "aal", "amr", "auth_time"} {
		if _, ok := claims[required]; !ok {
			t.Errorf("missing claim %s", required)
		}
	}
}

func TestManagementSessionProviderStopsRetryingRejectedDashboardSession(t *testing.T) {
	now := time.Date(2026, time.August, 23, 12, 0, 0, 0, time.UTC)
	var challengeCalls int
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != managementBasePath+"/auth/exchange-challenges" {
			http.NotFound(response, request)
			return
		}
		challengeCalls++
		response.WriteHeader(http.StatusUnauthorized)
	}))
	defer server.Close()
	providerValue, err := NewManagementSessionProvider(ManagementSessionOptions{
		RouterURL: server.URL, IssuerURL: "https://dashboard.example.test",
		IssuerID: "10000000-0000-4000-8000-000000000001", Signer: &recordingAssertionSigner{},
		Client: server.Client(), Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	principal := dashboardauth.AuthContext{
		UserID: "dashboard-user", SessionID: "dashboard-session", ExpiresAt: now.Add(12 * time.Hour),
	}
	for attempt := 0; attempt < 2; attempt++ {
		_, err = providerValue.ManagementAccessToken(context.Background(), principal)
		if !errors.Is(err, ErrManagementSessionReauthentication) {
			t.Fatalf("attempt %d error = %v", attempt, err)
		}
	}
	if challengeCalls != 1 {
		t.Fatalf("rejected Dashboard session challenge calls = %d", challengeCalls)
	}
	principal.SessionID = "new-dashboard-session"
	_, err = providerValue.ManagementAccessToken(context.Background(), principal)
	if !errors.Is(err, ErrManagementSessionReauthentication) || challengeCalls != 2 {
		t.Fatalf("new Dashboard session error=%v challenge calls=%d", err, challengeCalls)
	}
}

func TestManagementSessionProviderAppliesSharedBoundedChallengeBackoff(t *testing.T) {
	current := time.Date(2026, time.August, 23, 12, 0, 0, 0, time.UTC)
	var challengeCalls int
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != managementBasePath+"/auth/exchange-challenges" {
			http.NotFound(response, request)
			return
		}
		challengeCalls++
		response.Header().Set("Retry-After", "30")
		response.WriteHeader(http.StatusTooManyRequests)
	}))
	defer server.Close()
	providerValue, err := NewManagementSessionProvider(ManagementSessionOptions{
		RouterURL: server.URL, IssuerURL: "https://dashboard.example.test",
		IssuerID: "10000000-0000-4000-8000-000000000001", Signer: &recordingAssertionSigner{},
		Client: server.Client(), Now: func() time.Time { return current },
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, sessionID := range []string{"dashboard-session-1", "dashboard-session-2"} {
		_, err = providerValue.ManagementAccessToken(context.Background(), dashboardauth.AuthContext{
			UserID: "dashboard-user", SessionID: sessionID, ExpiresAt: current.Add(12 * time.Hour),
		})
		var sessionErr *ManagementSessionError
		if !errors.As(err, &sessionErr) || sessionErr.Status != http.StatusTooManyRequests ||
			sessionErr.RetryAfter <= 0 || sessionErr.RetryAfter > 30*time.Second {
			t.Fatalf("session %s error = %#v", sessionID, err)
		}
	}
	if challengeCalls != 1 {
		t.Fatalf("challenge calls during shared backoff = %d", challengeCalls)
	}
	current = current.Add(31 * time.Second)
	_, _ = providerValue.ManagementAccessToken(context.Background(), dashboardauth.AuthContext{
		UserID: "dashboard-user", SessionID: "dashboard-session-3", ExpiresAt: current.Add(12 * time.Hour),
	})
	if challengeCalls != 2 {
		t.Fatalf("challenge calls after backoff = %d", challengeCalls)
	}
}

func TestManagementResponseDecodingClosesOnlySecretBearingVariants(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", managementMediaType)
		_ = json.NewEncoder(response).Encode(map[string]any{
			"accessToken": "token", "tokenType": "Bearer", "expiresIn": 60,
			"managementSessionId": "10000000-0000-4000-8000-000000000003",
			"onboarding":          map[string]any{},
		})
	}))
	defer server.Close()

	provider := &managementSessionProvider{routerURL: server.URL, client: server.Client()}
	var envelope managementTokenEnvelope
	if err := provider.request(
		context.Background(), http.MethodPost, "/response", map[string]string{},
		http.StatusOK, &envelope, true,
	); err == nil {
		t.Fatal("closed token envelope accepted an onboarding variant")
	}
	if err := provider.request(
		context.Background(), http.MethodPost, "/response", map[string]string{},
		http.StatusOK, &envelope, false,
	); err != nil {
		t.Fatalf("additive resource response rejected a future field: %v", err)
	}
}

func TestManagementSessionProviderRejectsInvalidSourceSessionExpiry(t *testing.T) {
	t.Parallel()
	now := time.Date(2026, time.August, 23, 12, 0, 0, 0, time.UTC)
	provider := &managementSessionProvider{
		issuerURL: "https://dashboard.example.test",
		signer:    &recordingAssertionSigner{},
	}
	for _, test := range []struct {
		name      string
		expiresAt time.Time
	}{
		{name: "missing", expiresAt: time.Time{}},
		{name: "expired", expiresAt: now.Add(-time.Second)},
		{name: "before assertion expiry", expiresAt: now.Add(assertionLifetime - time.Second)},
		{name: "beyond maximum lifetime", expiresAt: now.Add(maximumSourceSessionLifetime + time.Second)},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := provider.assertion(dashboardauth.AuthContext{
				UserID: "dashboard-user", SessionID: "dashboard-session",
				ExpiresAt: test.expiresAt,
			}, "router-nonce", now)
			if !errors.Is(err, ErrManagementSessionUnavailable) {
				t.Fatalf("assertion() error = %v, want ErrManagementSessionUnavailable", err)
			}
		})
	}
}

func TestManagementSessionProviderRetiresDerivedSessionByIssuerSessionID(t *testing.T) {
	t.Parallel()
	now := time.Date(2026, time.August, 23, 12, 0, 0, 0, time.UTC)
	issuerID := "10000000-0000-4000-8000-000000000001"
	var requestBody map[string]string
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Content-Type", managementMediaType)
		if request.URL.Path != managementBasePath+"/auth/backchannel-logout" {
			http.NotFound(response, request)
			return
		}
		if err := json.NewDecoder(request.Body).Decode(&requestBody); err != nil {
			t.Error(err)
		}
		_ = json.NewEncoder(response).Encode(map[string]any{
			"applied": true, "replayed": false, "futureMetadata": "ignored",
		})
	}))
	defer server.Close()

	signer := &recordingAssertionSigner{}
	providerValue, err := NewManagementSessionProvider(ManagementSessionOptions{
		RouterURL: server.URL, IssuerURL: "https://dashboard.example.test", IssuerID: issuerID,
		Signer: signer, Client: server.Client(), Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	provider := providerValue.(*managementSessionProvider)
	provider.cache["dashboard-session-1"] = cachedManagementToken{
		accessToken: "cached", managementSessionID: "router-session", expiresAt: now.Add(time.Minute),
	}
	if err := provider.RetireDashboardSession(context.Background(), "dashboard-session-1"); err != nil {
		t.Fatal(err)
	}
	if requestBody["issuerId"] != issuerID || requestBody["logoutToken"] != "signed-dashboard-assertion" {
		t.Fatalf("logout body = %#v", requestBody)
	}
	provider.mu.Lock()
	_, cached := provider.cache["dashboard-session-1"]
	provider.mu.Unlock()
	if cached {
		t.Fatal("retired Dashboard session remained in Management token cache")
	}
	signer.mu.Lock()
	claims := signer.claims
	signer.mu.Unlock()
	if claims["iss"] != "https://dashboard.example.test" || claims["aud"] != managementAudience ||
		claims["sid"] != "dashboard-session-1" {
		t.Fatalf("logout claims = %#v", claims)
	}
	events, ok := claims["events"].(map[string]any)
	if !ok || len(events) != 1 {
		t.Fatalf("logout events = %#v", claims["events"])
	}
	if payload, exists := events[backchannelLogoutEvent]; !exists {
		t.Fatalf("logout event is absent: %#v", events)
	} else if details, valid := payload.(map[string]any); !valid || len(details) != 0 {
		t.Fatalf("logout event payload = %#v", payload)
	}
}

func TestManagementSessionProviderRejectsNonCanonicalIssuerIdentity(t *testing.T) {
	t.Parallel()
	for _, testCase := range []struct {
		name, issuer, issuerID string
	}{
		{name: "plaintext issuer", issuer: "http://dashboard.example.test", issuerID: "10000000-0000-4000-8000-000000000001"},
		{name: "noncanonical issuer", issuer: "https://dashboard.example.test/", issuerID: "10000000-0000-4000-8000-000000000001"},
		{name: "noncanonical UUID", issuer: "https://dashboard.example.test", issuerID: "10000000000040008000000000000001"},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			_, err := NewManagementSessionProvider(ManagementSessionOptions{
				RouterURL: "http://router.example.test", IssuerURL: testCase.issuer,
				IssuerID: testCase.issuerID, Signer: &recordingAssertionSigner{},
			})
			if err == nil {
				t.Fatal("NewManagementSessionProvider() unexpectedly succeeded")
			}
		})
	}
}

type fixedManagementSessionProvider struct {
	token string
	err   error
}

func (provider fixedManagementSessionProvider) ManagementAccessToken(context.Context, dashboardauth.AuthContext) (string, error) {
	return provider.token, provider.err
}

func TestRewriteManagementAuthorizationUsesOnlyExchangedPrincipalToken(t *testing.T) {
	t.Parallel()
	request := httptest.NewRequest(http.MethodGet, "https://router.example.test/management/v1/me?authToken=query-secret&access_token=oauth-secret&api_key=key-secret&cursor=next", nil)
	request.Header.Set("Authorization", "Bearer browser-secret")
	request.Header.Set("Proxy-Authorization", "Bearer proxy-secret")
	request.Header.Set("Cookie", "vsr_session=cookie-secret")
	request.Header.Set("X-API-Key", "header-secret")
	request.Header.Set("X-VLLM-SR-Principal", "forged-principal")
	request.Header.Set("X-VLLM-SR-User", "forged-user")
	request.Header.Set("X-VLLM-SR-Team", "forged-team")
	request = request.WithContext(dashboardauth.WithAuthContext(request.Context(), dashboardauth.AuthContext{
		UserID: "dashboard-user-1", SessionID: "dashboard-session-1",
	}))
	if err := RewriteManagementAuthorization(request, fixedManagementSessionProvider{token: "router-token"}); err != nil {
		t.Fatal(err)
	}
	if request.Header.Get("Authorization") != "Bearer router-token" ||
		request.Header.Get("Proxy-Authorization") != "" || request.Header.Get("Cookie") != "" ||
		request.Header.Get("X-API-Key") != "" || request.Header.Get("X-VLLM-SR-Principal") != "" ||
		request.Header.Get("X-VLLM-SR-User") != "" || request.Header.Get("X-VLLM-SR-Team") != "" ||
		request.URL.Query().Get("authToken") != "" || request.URL.Query().Get("access_token") != "" ||
		request.URL.Query().Get("api_key") != "" || request.URL.Query().Get("cursor") != "next" {
		t.Fatalf("rewritten request headers=%v query=%q", request.Header, request.URL.RawQuery)
	}
}

func TestRewriteManagementAuthorizationFailsClosed(t *testing.T) {
	t.Parallel()
	request := httptest.NewRequest(http.MethodGet, "https://router.example.test/management/v1/me", nil)
	request.Header.Set("Authorization", "Bearer browser-secret")
	err := RewriteManagementAuthorization(request, fixedManagementSessionProvider{err: errors.New("unavailable")})
	if !errors.Is(err, ErrManagementSessionUnavailable) || request.Header.Get("Authorization") != "" {
		t.Fatalf("error=%v authorization=%q", err, request.Header.Get("Authorization"))
	}
}

func TestManagementResponseRejectsWrongMediaType(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "application/json")
		_, _ = response.Write([]byte(`{"nonce":"not-trusted"}`))
	}))
	defer server.Close()
	provider, err := NewManagementSessionProvider(ManagementSessionOptions{
		RouterURL: server.URL, IssuerURL: "https://dashboard.example.test",
		IssuerID: "10000000-0000-4000-8000-000000000001", Signer: &recordingAssertionSigner{},
		Client: server.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = provider.ManagementAccessToken(context.Background(), dashboardauth.AuthContext{
		UserID: "user", SessionID: "session", ExpiresAt: time.Now().UTC().Add(time.Hour),
	})
	if !errors.Is(err, ErrManagementSessionUnavailable) || !strings.Contains(err.Error(), "unavailable") {
		t.Fatalf("error = %v", err)
	}
}
