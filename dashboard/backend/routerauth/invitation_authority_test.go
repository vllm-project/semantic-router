package routerauth

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

const (
	invitationIssuerID   = "30000000-0000-4000-8000-000000000001"
	invitationChallenge  = "30000000-0000-4000-8000-000000000002"
	invitationNamespace  = "30000000-0000-4000-8000-000000000003"
	invitationPrincipal  = "30000000-0000-4000-8000-000000000004"
	invitationUser       = "30000000-0000-4000-8000-000000000005"
	invitationAPIKey     = "30000000-0000-4000-8000-000000000006"
	invitationResourceID = "30000000-0000-4000-8000-000000000007"
	invitationSubject    = "30000000-0000-4000-8000-000000000008"
	invitationSession    = "30000000-0000-4000-8000-000000000009"
	invitationTeam       = "30000000-0000-4000-8000-000000000010"
	operatorRoleID       = "10000000-0000-5000-8000-000000000003"
	consumerRoleID       = "10000000-0000-5000-8000-000000000008"
)

func writeManagementJSON(t *testing.T, response http.ResponseWriter, value any) {
	t.Helper()
	response.Header().Set("Content-Type", managementMediaType)
	if err := json.NewEncoder(response).Encode(value); err != nil {
		t.Fatal(err)
	}
}

func writeCreatedManagementJSON(t *testing.T, response http.ResponseWriter, value any) {
	t.Helper()
	response.Header().Set("Content-Type", managementMediaType)
	response.WriteHeader(http.StatusCreated)
	if err := json.NewEncoder(response).Encode(value); err != nil {
		t.Fatal(err)
	}
}

func newInvitationProvider(t *testing.T, server *httptest.Server, now time.Time, signer AssertionSigner) *managementSessionProvider {
	t.Helper()
	value, err := NewManagementSessionProvider(ManagementSessionOptions{
		RouterURL: server.URL, IssuerURL: "https://dashboard.example.test",
		IssuerID: invitationIssuerID, Signer: signer, Client: server.Client(), Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	provider, ok := value.(*managementSessionProvider)
	if !ok {
		t.Fatalf("provider type = %T", value)
	}
	return provider
}

func TestInvitationAuthorityCreateCarriesHumanSessionNamespaceAndIdempotency(t *testing.T) {
	t.Parallel()
	now := time.Date(2026, time.August, 23, 12, 0, 0, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != managementBasePath+"/invitations" || request.Method != http.MethodPost {
			http.NotFound(response, request)
			return
		}
		if request.Header.Get("Authorization") != "Bearer human-management-token" ||
			request.Header.Get(managementapi.HeaderNamespaceID) != invitationNamespace ||
			request.Header.Get(managementapi.HeaderIdempotencyKey) != "invite-create-123456" {
			t.Errorf("headers = %#v", request.Header)
		}
		var body managementapi.InvitationCreateRequest
		decoder := json.NewDecoder(request.Body)
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&body); err != nil {
			t.Fatal(err)
		}
		response.Header().Set("Content-Type", managementMediaType)
		response.WriteHeader(http.StatusCreated)
		writeManagementJSON(t, response, managementapi.InvitationIssuedSecret{
			Data: managementapi.Invitation{
				InvitationID: invitationResourceID, NamespaceID: invitationNamespace,
				ExpectedIdentity: body.ExpectedIdentity, DisplayName: body.DisplayName,
				Onboarding: managementapi.InvitationOnboardingSnapshot{AutomaticFirstKey: true},
				ExpiresAt:  body.ExpiresAt, Status: "pending", Revision: 1, CreatedAt: now, UpdatedAt: now,
			},
			Token: "router-invitation-token", DeliveryExpiresAt: now.Add(time.Hour),
		})
	}))
	defer server.Close()
	provider := newInvitationProvider(t, server, now, &recordingAssertionSigner{})
	provider.cache["dashboard-session"] = cachedManagementToken{
		accessToken: "human-management-token", expiresAt: now.Add(time.Hour),
	}
	request := managementapi.InvitationCreateRequest{
		ExpectedIdentity: managementapi.InvitationExpectedIdentity{
			Issuer:  "https://dashboard.example.test",
			Subject: invitationSubject, Email: "member@example.test",
		},
		DisplayName: "Member", ExpiresAt: now.Add(24 * time.Hour),
	}
	issued, err := provider.CreateInvitation(context.Background(), dashboardauth.AuthContext{
		UserID: "dashboard-user", SessionID: "dashboard-session", ExpiresAt: now.Add(time.Hour),
	}, invitationNamespace, "invite-create-123456", request)
	if err != nil || issued.Token != "router-invitation-token" {
		t.Fatalf("CreateInvitation() = %#v, %v", issued, err)
	}
}

func TestInvitationAuthorityRevokeReturnsRouterRevision(t *testing.T) {
	t.Parallel()
	now := time.Date(2026, time.August, 23, 12, 0, 0, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != managementBasePath+"/invitations/"+invitationResourceID ||
			request.Method != http.MethodDelete ||
			request.Header.Get(managementapi.HeaderIfMatch) != `"invitation:7"` {
			t.Fatalf("revoke request = %s %s headers=%#v", request.Method, request.URL.Path, request.Header)
		}
		response.Header().Set(managementapi.HeaderETag, `"invitation:11"`)
		response.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()
	provider := newInvitationProvider(t, server, now, &recordingAssertionSigner{})
	provider.cache["dashboard-session"] = cachedManagementToken{
		accessToken: "human-management-token", expiresAt: now.Add(time.Hour),
	}
	revision, err := provider.RevokeInvitation(context.Background(), dashboardauth.AuthContext{
		UserID: "dashboard-user", SessionID: "dashboard-session", ExpiresAt: now.Add(time.Hour),
	}, invitationNamespace, invitationResourceID, 7)
	if err != nil || revision != 11 {
		t.Fatalf("RevokeInvitation() = %d, %v", revision, err)
	}
}

func TestInvitationAcceptReturnsBoundedOnboardingWithoutCachingSecret(t *testing.T) {
	t.Parallel()
	now := time.Date(2026, time.August, 23, 12, 0, 0, 0, time.UTC)
	signer := &recordingAssertionSigner{}
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case managementBasePath + "/auth/exchange-challenges":
			writeCreatedManagementJSON(t, response, exchangeChallenge{
				ExchangeChallengeID: invitationChallenge,
				Nonce:               "invitation-nonce", ExpiresAt: now.Add(time.Minute),
			})
		case managementBasePath + "/auth/token-exchange":
			var body managementapi.TokenExchangeRequest
			if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			if body.InvitationToken == nil || *body.InvitationToken != "router-invitation-token" {
				t.Fatalf("token exchange body = %#v", body)
			}
			writeManagementJSON(t, response, managementapi.TokenExchangeResponse{
				ManagementTokenEnvelope: managementapi.ManagementTokenEnvelope{
					AccessToken: "invited-session-token",
					TokenType:   "Bearer", ExpiresIn: 300, ManagementSessionID: invitationSession,
				},
				Onboarding: &managementapi.OnboardingResult{
					InvitationID: invitationResourceID,
					PrincipalID:  invitationPrincipal, UserID: invitationUser, TeamID: invitationTeam,
					APIKeyID: invitationAPIKey, APIKey: "sk-one-time-onboarding",
					DeliveryExpiresAt: now.Add(time.Hour),
				},
			})
		case managementBasePath + "/me":
			if request.Header.Get("Authorization") != "Bearer invited-session-token" {
				t.Fatalf("invited /me authorization = %q", request.Header.Get("Authorization"))
			}
			writeManagementJSON(t, response, managementapi.Me{
				Principal: managementapi.MePrincipal{PrincipalID: invitationPrincipal, DisplayName: "Member", Kind: "human", Status: "active"},
				Session: managementapi.MeSession{
					SessionID: invitationSession, AuthenticatedAt: now,
					ExpiresAt: now.Add(time.Hour), EvidenceKind: "human",
				},
				ClusterPermissions: []string{}, Namespaces: []managementapi.MeNamespaceScope{{
					Namespace:   managementapi.MeNamespace{NamespaceID: invitationNamespace, Name: "default", Status: "active", DesiredRevision: 1, AppliedRevision: 1},
					Permissions: []string{}, RoleBindings: []managementapi.ManagementRoleBinding{{
						BindingID: "30000000-0000-4000-8000-000000000011", PrincipalID: invitationPrincipal,
						RoleID: operatorRoleID, Scope: managementapi.ManagementScope{Kind: "namespace", NamespaceID: invitationNamespace},
						Status: "active", Revision: 1, CreatedAt: now, UpdatedAt: now,
					}},
					User:              &managementapi.MeUser{UserID: invitationUser, Email: "member@example.test", DisplayName: "Member", Status: "active"},
					Teams:             []managementapi.MeTeamMembership{{TeamID: invitationTeam, Name: "Team", Role: "member", Status: "active"}},
					SelfServicePolicy: managementapi.MeSelfServicePolicy{AutomaticFirstKey: true, Revision: 1},
				}},
			})
		default:
			http.NotFound(response, request)
		}
	}))
	defer server.Close()
	provider := newInvitationProvider(t, server, now, signer)
	result, err := provider.AcceptInvitation(context.Background(), dashboardauth.RouterInvitationAcceptance{
		NamespaceID: invitationNamespace, InvitationToken: "router-invitation-token",
		PlannedSubject: invitationSubject, Email: "member@example.test", DisplayName: "Member",
		SessionExpiresAt: now.Add(12 * time.Hour),
	})
	if err != nil || result.DashboardRole != dashboardauth.RoleWrite || result.Onboarding.APIKey != "sk-one-time-onboarding" {
		t.Fatalf("AcceptInvitation() = %#v, %v", result, err)
	}
	signer.mu.Lock()
	claims := signer.claims
	signer.mu.Unlock()
	if claims["sub"] != invitationSubject || claims["sid"] != nil ||
		claims[routerSourceExpiryClaim] != now.Add(12*time.Hour).Unix() {
		t.Fatalf("invitation assertion claims = %#v", claims)
	}
	provider.mu.Lock()
	defer provider.mu.Unlock()
	for _, cached := range provider.cache {
		if cached.accessToken == result.Onboarding.APIKey {
			t.Fatal("one-time onboarding key entered the normal Management session cache")
		}
	}
}

func TestInvitedDashboardRoleAcceptsOnlyTheInvitedUsersConsumerGrant(t *testing.T) {
	now := time.Now().UTC()
	identity := managementapi.Me{Namespaces: []managementapi.MeNamespaceScope{{
		Namespace: managementapi.MeNamespace{NamespaceID: invitationNamespace},
		User:      &managementapi.MeUser{UserID: invitationUser},
		RoleBindings: []managementapi.ManagementRoleBinding{{
			BindingID: "consumer-binding", PrincipalID: invitationPrincipal, RoleID: consumerRoleID,
			Scope: managementapi.ManagementScope{
				Kind: "user", NamespaceID: invitationNamespace, UserID: invitationUser,
			},
			Status: "active", Revision: 1, CreatedAt: now, UpdatedAt: now,
		}},
	}}}
	role, err := invitedDashboardRole(identity, invitationNamespace, managementapi.OnboardingResult{
		UserID: invitationUser,
	})
	if err != nil || role != dashboardauth.RoleRead {
		t.Fatalf("consumer invitation role = %q, %v", role, err)
	}
	identity.Namespaces[0].RoleBindings[0].Scope.UserID = invitationSubject
	if _, err := invitedDashboardRole(identity, invitationNamespace, managementapi.OnboardingResult{
		UserID: invitationUser,
	}); err == nil {
		t.Fatal("cross-user consumer binding granted a Dashboard role")
	}
}

func TestNormalManagementExchangeRejectsOnboardingEnvelope(t *testing.T) {
	t.Parallel()
	now := time.Date(2026, time.August, 23, 12, 0, 0, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case managementBasePath + "/auth/exchange-challenges":
			writeCreatedManagementJSON(t, response, exchangeChallenge{
				ExchangeChallengeID: invitationChallenge,
				Nonce:               "normal-nonce", ExpiresAt: now.Add(time.Minute),
			})
		case managementBasePath + "/auth/token-exchange":
			writeManagementJSON(t, response, managementapi.TokenExchangeResponse{
				ManagementTokenEnvelope: managementapi.ManagementTokenEnvelope{
					AccessToken: "normal-token",
					TokenType:   "Bearer", ExpiresIn: 300, ManagementSessionID: invitationSession,
				},
				Onboarding: &managementapi.OnboardingResult{APIKey: "must-not-leak"},
			})
		default:
			http.NotFound(response, request)
		}
	}))
	defer server.Close()
	provider := newInvitationProvider(t, server, now, &recordingAssertionSigner{})
	_, err := provider.ManagementAccessToken(context.Background(), dashboardauth.AuthContext{
		UserID: "dashboard-user", SessionID: "dashboard-session", ExpiresAt: now.Add(time.Hour),
	})
	if err == nil {
		t.Fatal("normal Management exchange accepted an onboarding envelope")
	}
}

func TestInvitationAcceptRejectsCrossNamespaceIdentity(t *testing.T) {
	t.Parallel()
	now := time.Date(2026, time.August, 23, 12, 0, 0, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case managementBasePath + "/auth/exchange-challenges":
			writeCreatedManagementJSON(t, response, exchangeChallenge{
				ExchangeChallengeID: invitationChallenge,
				Nonce:               "cross-namespace", ExpiresAt: now.Add(time.Minute),
			})
		case managementBasePath + "/auth/token-exchange":
			writeManagementJSON(t, response, managementapi.TokenExchangeResponse{
				ManagementTokenEnvelope: managementapi.ManagementTokenEnvelope{AccessToken: "invited-token", TokenType: "Bearer", ExpiresIn: 300, ManagementSessionID: invitationSession},
				Onboarding: &managementapi.OnboardingResult{
					InvitationID: invitationResourceID, PrincipalID: invitationPrincipal,
					UserID: invitationUser, APIKeyID: invitationAPIKey, APIKey: "sk-bounded", DeliveryExpiresAt: now.Add(time.Hour),
				},
			})
		case managementBasePath + "/me":
			writeManagementJSON(t, response, managementapi.Me{
				Principal:  managementapi.MePrincipal{PrincipalID: invitationPrincipal},
				Namespaces: []managementapi.MeNamespaceScope{{Namespace: managementapi.MeNamespace{NamespaceID: "30000000-0000-4000-8000-000000000099"}}},
			})
		default:
			http.NotFound(response, request)
		}
	}))
	defer server.Close()
	provider := newInvitationProvider(t, server, now, &recordingAssertionSigner{})
	_, err := provider.AcceptInvitation(context.Background(), dashboardauth.RouterInvitationAcceptance{
		NamespaceID: invitationNamespace, InvitationToken: "router-token", PlannedSubject: invitationSubject,
		Email: "member@example.test", DisplayName: "Member", SessionExpiresAt: now.Add(12 * time.Hour),
	})
	if err == nil {
		t.Fatal("cross-namespace onboarding identity was accepted")
	}
}
