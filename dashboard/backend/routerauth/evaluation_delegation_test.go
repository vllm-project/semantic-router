package routerauth

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

func TestIssueEvaluationInferenceTokenUsesEligibleTeamKey(t *testing.T) {
	t.Parallel()
	now := time.Date(2026, time.August, 24, 0, 0, 0, 0, time.UTC)
	const (
		namespaceID         = "11111111-1111-4111-8111-111111111111"
		teamID              = "22222222-2222-4222-8222-222222222222"
		keyID               = "33333333-3333-4333-8333-333333333333"
		managementSessionID = "55555555-5555-4555-8555-555555555555"
	)
	identity := managementapi.Me{Namespaces: []managementapi.MeNamespaceScope{{
		Namespace: managementapi.MeNamespace{NamespaceID: namespaceID, Status: "active"},
		User:      &managementapi.MeUser{UserID: "current-router-user", Status: "active"},
		Teams:     []managementapi.MeTeamMembership{{TeamID: teamID, Status: "active"}},
	}}}
	expiresAt := now.Add(5 * time.Minute)
	delegationCalls := 0
	delegationIdempotencyKeys := make([]string, 0, 2)
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Authorization") != "Bearer management-token" {
			t.Errorf("Authorization = %q", request.Header.Get("Authorization"))
		}
		response.Header().Set("Content-Type", managementMediaType)
		switch request.Method + " " + request.URL.Path {
		case http.MethodGet + " " + managementBasePath + "/me":
			_ = json.NewEncoder(response).Encode(identity)
		case http.MethodGet + " " + managementBasePath + "/self/inference-keys":
			if request.Header.Get(managementapi.HeaderNamespaceID) != namespaceID ||
				request.URL.Query().Get("pageSize") != "200" {
				t.Errorf("eligible-key request headers=%v query=%v", request.Header, request.URL.Query())
			}
			_ = json.NewEncoder(response).Encode(managementapi.EligibleInferenceKeyPage{
				Data: []managementapi.EligibleInferenceKey{
					{KeyID: "unrelated", Owner: managementapi.APIKeyOwner{Type: "user", ID: "current-router-user"}},
					{KeyID: keyID, Owner: managementapi.APIKeyOwner{Type: "team", ID: teamID}, ContextTeamID: teamID},
				},
				Page: managementapi.PageInfo{PageSize: 200},
			})
		case http.MethodPost + " " + managementBasePath + "/self/inference-sessions":
			delegationCalls++
			delegationIdempotencyKeys = append(
				delegationIdempotencyKeys,
				request.Header.Get(managementapi.HeaderIdempotencyKey),
			)
			if request.Header.Get(managementapi.HeaderNamespaceID) != namespaceID ||
				request.Header.Get(managementapi.HeaderIdempotencyKey) == "" {
				t.Errorf("delegation request headers=%v", request.Header)
			}
			var create managementapi.DelegatedInferenceSessionCreateRequest
			if err := json.NewDecoder(request.Body).Decode(&create); err != nil || create.KeyID != keyID {
				t.Errorf("delegation request=%+v err=%v", create, err)
			}
			if delegationCalls == 1 {
				http.Error(response, "publication is still converging", http.StatusServiceUnavailable)
				return
			}
			response.WriteHeader(http.StatusCreated)
			_ = json.NewEncoder(response).Encode(managementapi.SecretEnvelope{
				ResourceID: "44444444-4444-4444-8444-444444444444",
				Kind:       managementapi.SecretKindDelegatedCredential,
				Secret:     "vsd_evaluation_delegated_secret",
				ExpiresAt:  &expiresAt,
			})
		default:
			http.Error(response, "unexpected request", http.StatusNotFound)
		}
	}))
	defer server.Close()

	provider := &managementSessionProvider{
		routerURL: server.URL,
		client:    server.Client(),
		now:       func() time.Time { return now },
		cache: map[string]cachedManagementToken{
			"dashboard-session": {
				accessToken: "management-token", managementSessionID: managementSessionID,
				expiresAt: now.Add(time.Minute),
			},
		},
		inflight: make(map[string]*managementTokenExchange),
	}
	principal := dashboardauth.AuthContext{
		UserID: "dashboard-local-user", SessionID: "dashboard-session", ExpiresAt: now.Add(time.Hour),
	}
	assertEvaluationDelegationLifecycle(
		t, provider, principal, teamID, &delegationIdempotencyKeys,
	)
}

func assertEvaluationDelegationLifecycle(
	t *testing.T,
	provider *managementSessionProvider,
	principal dashboardauth.AuthContext,
	teamID string,
	delegationIdempotencyKeys *[]string,
) {
	t.Helper()
	if _, err := provider.IssueEvaluationInferenceToken(
		t.Context(), principal, "different-team-member", teamID,
	); !errors.Is(err, errEvaluationInferenceUnavailable) {
		t.Fatalf("first IssueEvaluationInferenceToken() error = %v", err)
	}
	token, err := provider.IssueEvaluationInferenceToken(
		t.Context(),
		principal,
		"different-team-member",
		teamID,
	)
	if err != nil {
		t.Fatal(err)
	}
	if token != "vsd_evaluation_delegated_secret" {
		t.Fatalf("token = %q", token)
	}
	cachedToken, err := provider.IssueEvaluationInferenceToken(
		t.Context(), principal, "different-team-member", teamID,
	)
	if err != nil || cachedToken != token {
		t.Fatalf("cached IssueEvaluationInferenceToken() matched=%t error=%v", cachedToken == token, err)
	}
	keys := *delegationIdempotencyKeys
	if len(keys) != 2 || keys[0] == "" || keys[0] != keys[1] {
		t.Fatalf("delegation idempotency keys = %v", keys)
	}
}
