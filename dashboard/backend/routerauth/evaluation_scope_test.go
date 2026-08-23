package routerauth

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

func TestResolveEvaluationScopeUsesRouterIdentity(t *testing.T) {
	t.Parallel()
	now := time.Date(2026, time.August, 24, 0, 0, 0, 0, time.UTC)
	identity := managementapi.Me{Namespaces: []managementapi.MeNamespaceScope{
		{
			User: &managementapi.MeUser{UserID: "router-user-b", Status: "active"},
			Teams: []managementapi.MeTeamMembership{
				{TeamID: "team-b", Status: "active"},
				{TeamID: "disabled-team", Status: "disabled"},
			},
		},
		{
			User:  &managementapi.MeUser{UserID: "router-user-a", Status: "active"},
			Teams: []managementapi.MeTeamMembership{{TeamID: "team-a", Status: "active"}},
		},
	}}
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodGet || request.URL.Path != managementBasePath+"/me" ||
			request.Header.Get("Authorization") != "Bearer scoped-token" {
			http.Error(response, "unexpected request", http.StatusBadRequest)
			return
		}
		response.Header().Set("Content-Type", managementMediaType)
		_ = json.NewEncoder(response).Encode(identity)
	}))
	defer server.Close()

	provider := &managementSessionProvider{
		routerURL: server.URL,
		client:    server.Client(),
		now:       func() time.Time { return now },
		cache: map[string]cachedManagementToken{
			"dashboard-session": {accessToken: "scoped-token", expiresAt: now.Add(time.Minute)},
		},
		inflight: make(map[string]*managementTokenExchange),
	}
	users, teams, err := provider.ResolveEvaluationScope(t.Context(), dashboardauth.AuthContext{
		UserID: "dashboard-local-user", SessionID: "dashboard-session", ExpiresAt: now.Add(time.Hour),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(users, []string{"router-user-a", "router-user-b"}) {
		t.Fatalf("users = %v", users)
	}
	wantTeams := map[string]string{"team-a": "router-user-a", "team-b": "router-user-b"}
	if !reflect.DeepEqual(teams, wantTeams) {
		t.Fatalf("teams = %v, want %v", teams, wantTeams)
	}
}

func TestResolveEvaluationScopeRequiresLinkedActiveUser(t *testing.T) {
	t.Parallel()
	now := time.Date(2026, time.August, 24, 0, 0, 0, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", managementMediaType)
		_ = json.NewEncoder(response).Encode(managementapi.Me{})
	}))
	defer server.Close()
	provider := &managementSessionProvider{
		routerURL: server.URL, client: server.Client(), now: func() time.Time { return now },
		cache: map[string]cachedManagementToken{
			"dashboard-session": {accessToken: "scoped-token", expiresAt: now.Add(time.Minute)},
		},
		inflight: make(map[string]*managementTokenExchange),
	}
	_, _, err := provider.ResolveEvaluationScope(t.Context(), dashboardauth.AuthContext{
		UserID: "dashboard-local-user", SessionID: "dashboard-session", ExpiresAt: now.Add(time.Hour),
	})
	if !errors.Is(err, errEvaluationScopeUnavailable) {
		t.Fatalf("error = %v", err)
	}
}
