package routerauth

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

type firstAdminProvisioningFixture struct {
	*testing.T
	sync.Mutex
	now               time.Time
	issuerID          string
	principalID       string
	namespaceID       string
	quotaPartitionID  string
	userID            string
	roleID            string
	identity          dashboardauth.FirstAdminIdentity
	tokenPath         string
	bootstrapped      bool
	namespace         bool
	user              bool
	linked            bool
	binding           bool
	challenge         int
	provisioningCalls []string
}

type firstAdminProvisioningSetup struct {
	fixture  *firstAdminProvisioningFixture
	provider ManagementIdentityProvider
}

func newFirstAdminProvisioningFixture(t *testing.T) *firstAdminProvisioningFixture {
	now := time.Date(2026, time.August, 23, 16, 0, 0, 0, time.UTC)
	tokenPath := filepath.Join(t.TempDir(), "router-token")
	if writeErr := os.WriteFile(tokenPath, []byte("router-bootstrap-token-which-is-at-least-32-bytes"), 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	return &firstAdminProvisioningFixture{
		T:                t,
		now:              now,
		issuerID:         "10000000-0000-4000-8000-000000000011",
		principalID:      "10000000-0000-4000-8000-000000000012",
		namespaceID:      "10000000-0000-4000-8000-000000000013",
		quotaPartitionID: "10000000-0000-4000-8000-000000000014",
		userID:           "10000000-0000-4000-8000-000000000015",
		roleID:           "10000000-0000-5000-8000-000000000002",
		identity: dashboardauth.FirstAdminIdentity{
			UserID: "10000000-0000-4000-8000-000000000016", SessionID: "10000000-0000-4000-8000-000000000017",
			Email: "admin@example.test", DisplayName: "Admin", AuthenticatedAt: now, ExpiresAt: now.Add(12 * time.Hour),
		},
		tokenPath: tokenPath,
	}
}

func newFirstAdminProvisioningSetup(t *testing.T) firstAdminProvisioningSetup {
	fixture := newFirstAdminProvisioningFixture(t)
	server := httptest.NewServer(fixture)
	t.Cleanup(server.Close)
	provider, err := NewManagementSessionProvider(ManagementSessionOptions{
		RouterURL: server.URL, IssuerURL: "https://dashboard.example.test", IssuerID: fixture.issuerID,
		Signer: &recordingAssertionSigner{}, Client: server.Client(), Now: func() time.Time { return fixture.now },
		BootstrapTokenFile: fixture.tokenPath,
	})
	if err != nil {
		t.Fatal(err)
	}
	return firstAdminProvisioningSetup{fixture: fixture, provider: provider}
}

func (fixture *firstAdminProvisioningFixture) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	fixture.Lock()
	defer fixture.Unlock()
	if request.Header.Get("Accept") != managementMediaType {
		fixture.Errorf("Accept = %q", request.Header.Get("Accept"))
	}
	switch request.URL.Path {
	case managementBasePath + "/auth/exchange-challenges":
		fixture.serveExchangeChallenge(response)
	case managementBasePath + "/auth/bootstrap":
		fixture.serveBootstrap(response, request)
	case managementBasePath + "/auth/token-exchange":
		fixture.serveTokenExchange(response)
	case managementBasePath + "/me":
		fixture.serveMe(response, request)
	case managementBasePath + "/namespaces":
		fixture.serveNamespaces(response, request)
	case managementBasePath + "/users":
		fixture.serveUsers(response, request)
	case managementBasePath + "/management-roles":
		fixture.serveManagementRoles(response, request)
	case managementBasePath + "/namespaces/" + fixture.namespaceID + "/principal-user-links/" + fixture.principalID:
		fixture.servePrincipalUserLink(response, request)
	case managementBasePath + "/role-bindings":
		fixture.serveRoleBindings(response, request)
	default:
		http.NotFound(response, request)
	}
}

func writeFirstAdminManagementResponse(response http.ResponseWriter, status int, payload any) {
	response.Header().Set("Content-Type", managementMediaType)
	response.WriteHeader(status)
	if payload != nil {
		_ = json.NewEncoder(response).Encode(payload)
	}
}

func (fixture *firstAdminProvisioningFixture) recordProvisioningCall(request *http.Request) {
	fixture.provisioningCalls = append(fixture.provisioningCalls, request.Method+" "+request.URL.Path)
}

func (fixture *firstAdminProvisioningFixture) serveExchangeChallenge(response http.ResponseWriter) {
	if !fixture.bootstrapped {
		http.Error(response, "not bootstrapped", http.StatusServiceUnavailable)
		return
	}
	fixture.challenge++
	writeFirstAdminManagementResponse(response, http.StatusCreated, exchangeChallenge{
		ExchangeChallengeID: "10000000-0000-4000-8000-000000000020",
		Nonce:               "nonce", ExpiresAt: fixture.now.Add(time.Minute),
	})
}

func (fixture *firstAdminProvisioningFixture) serveBootstrap(response http.ResponseWriter, request *http.Request) {
	if request.Header.Get("Authorization") != "VSR-Bootstrap router-bootstrap-token-which-is-at-least-32-bytes" ||
		request.Header.Get(managementapi.HeaderIdempotencyKey) != installationKey("bootstrap", fixture.identity.UserID) {
		fixture.Errorf("bootstrap headers = %#v", request.Header)
	}
	var body managementapi.BootstrapRequest
	decodeErr := json.NewDecoder(request.Body).Decode(&body)
	if decodeErr != nil || body.External == nil || body.Kind != "external_principal" ||
		body.External.Subject != fixture.identity.UserID || body.External.IssuerID != fixture.issuerID {
		fixture.Errorf("bootstrap body = %#v error=%v", body, decodeErr)
	}
	fixture.bootstrapped = true
	writeFirstAdminManagementResponse(response, http.StatusCreated, managementapi.BootstrapResponse{
		PrincipalID: fixture.principalID, RoleBindingID: "10000000-0000-4000-8000-000000000021",
		FinalizationRequired: true,
	})
}

func (fixture *firstAdminProvisioningFixture) serveTokenExchange(response http.ResponseWriter) {
	writeFirstAdminManagementResponse(response, http.StatusOK, managementTokenEnvelope{
		AccessToken: "cluster-admin-token", TokenType: "Bearer", ExpiresIn: 300,
		ManagementSessionID: "10000000-0000-4000-8000-000000000022",
	})
}

func (fixture *firstAdminProvisioningFixture) serveMe(response http.ResponseWriter, request *http.Request) {
	fixture.recordProvisioningCall(request)
	payload := managementapi.Me{Principal: managementapi.MePrincipal{PrincipalID: fixture.principalID}}
	if fixture.namespace && fixture.user && fixture.linked && fixture.binding {
		payload.Namespaces = []managementapi.MeNamespaceScope{{
			Namespace: managementapi.MeNamespace{NamespaceID: fixture.namespaceID, Name: defaultNamespaceName, Status: "active"},
			User: &managementapi.MeUser{
				UserID: fixture.userID, Email: fixture.identity.Email,
				DisplayName: fixture.identity.DisplayName, Status: "active",
			},
			RoleBindings: []managementapi.ManagementRoleBinding{{
				RoleID: fixture.roleID, PrincipalID: fixture.principalID, Status: "active",
				Scope: managementapi.ManagementScope{Kind: "namespace", NamespaceID: fixture.namespaceID},
			}},
		}}
	}
	writeFirstAdminManagementResponse(response, http.StatusOK, payload)
}

func (fixture *firstAdminProvisioningFixture) serveNamespaces(response http.ResponseWriter, request *http.Request) {
	fixture.recordProvisioningCall(request)
	if request.Method == http.MethodGet {
		items := []namespaceView{}
		if fixture.namespace {
			items = append(items, namespaceView{
				NamespaceID: fixture.namespaceID, Name: defaultNamespaceName,
				QuotaPartitionID: fixture.quotaPartitionID, BillingCurrency: defaultBillingCurrency,
				Status: "active", Revision: 1, RuntimeEpoch: 1, CreatedAt: fixture.now, UpdatedAt: fixture.now,
			})
		}
		writeFirstAdminManagementResponse(response, http.StatusOK, namespacePage{Data: items, Page: managementapi.PageInfo{PageSize: 200}})
		return
	}
	fixture.namespace = true
	writeFirstAdminManagementResponse(response, http.StatusCreated,
		managementapi.NewResourceMutationReceipt("namespace", fixture.namespaceID, 1, nil))
}

func (fixture *firstAdminProvisioningFixture) serveUsers(response http.ResponseWriter, request *http.Request) {
	fixture.recordProvisioningCall(request)
	if request.Header.Get(managementapi.HeaderNamespaceID) != fixture.namespaceID {
		fixture.Errorf("user namespace header = %q", request.Header.Get(managementapi.HeaderNamespaceID))
	}
	if request.Method == http.MethodPost && !fixture.binding {
		http.Error(response, "platform admin binding required", http.StatusForbidden)
		return
	}
	if request.Method == http.MethodGet {
		items := []managementapi.UserView{}
		if fixture.user {
			items = append(items, managementapi.UserView{
				UserID: fixture.userID, Email: fixture.identity.Email,
				DisplayName: fixture.identity.DisplayName, Status: "active", Revision: 1,
				CreatedAt: fixture.now, UpdatedAt: fixture.now,
			})
		}
		writeFirstAdminManagementResponse(response, http.StatusOK,
			managementapi.UserPage{Data: items, Page: managementapi.PageInfo{PageSize: 200}})
		return
	}
	fixture.user = true
	writeFirstAdminManagementResponse(response, http.StatusCreated,
		managementapi.NewResourceMutationReceipt("user", fixture.userID, 1, nil))
}

func (fixture *firstAdminProvisioningFixture) serveManagementRoles(response http.ResponseWriter, request *http.Request) {
	fixture.recordProvisioningCall(request)
	writeFirstAdminManagementResponse(response, http.StatusOK, managementapi.Page[managementapi.ManagementRole]{
		Data: []managementapi.ManagementRole{{
			RoleID: fixture.roleID, Name: platformAdminRoleName,
			DisplayName: "Platform admin", Permissions: []string{"namespace.read"}, BuiltIn: true,
			Status: "active", Revision: 1, CreatedAt: fixture.now, UpdatedAt: fixture.now,
		}},
		Page: managementapi.PageInfo{PageSize: 200},
	})
}

func (fixture *firstAdminProvisioningFixture) servePrincipalUserLink(response http.ResponseWriter, request *http.Request) {
	fixture.recordProvisioningCall(request)
	if request.Method == http.MethodGet && !fixture.linked {
		http.NotFound(response, request)
		return
	}
	fixture.linked = true
	status := http.StatusCreated
	if request.Method == http.MethodGet {
		status = http.StatusOK
	}
	writeFirstAdminManagementResponse(response, status, managementapi.PrincipalUserLinkDetail{Data: managementapi.PrincipalUserLink{
		PrincipalID: fixture.principalID, NamespaceID: fixture.namespaceID, UserID: fixture.userID, Revision: 1,
		CreatedAt: fixture.now, UpdatedAt: fixture.now,
	}})
}

func (fixture *firstAdminProvisioningFixture) serveRoleBindings(response http.ResponseWriter, request *http.Request) {
	fixture.recordProvisioningCall(request)
	if request.Method == http.MethodGet {
		items := []managementapi.ManagementRoleBinding{}
		if fixture.binding {
			items = append(items, managementapi.ManagementRoleBinding{
				BindingID: "10000000-0000-4000-8000-000000000023", PrincipalID: fixture.principalID,
				RoleID: fixture.roleID, Scope: managementapi.ManagementScope{Kind: "namespace", NamespaceID: fixture.namespaceID},
				Status: "active", Revision: 1, CreatedAt: fixture.now, UpdatedAt: fixture.now,
			})
		}
		writeFirstAdminManagementResponse(response, http.StatusOK,
			managementapi.Page[managementapi.ManagementRoleBinding]{Data: items, Page: managementapi.PageInfo{PageSize: 200}})
		return
	}
	fixture.binding = true
	writeFirstAdminManagementResponse(response, http.StatusCreated,
		managementapi.NewResourceMutationReceipt("management_role_binding", "10000000-0000-4000-8000-000000000023", 1, nil))
}

func assertFirstAdminProvisioningCalls(t *testing.T, fixture *firstAdminProvisioningFixture) {
	t.Helper()
	want := []string{
		http.MethodGet + " " + managementBasePath + "/me",
		http.MethodGet + " " + managementBasePath + "/namespaces",
		http.MethodPost + " " + managementBasePath + "/namespaces",
		http.MethodGet + " " + managementBasePath + "/management-roles",
		http.MethodGet + " " + managementBasePath + "/role-bindings",
		http.MethodPost + " " + managementBasePath + "/role-bindings",
		http.MethodGet + " " + managementBasePath + "/users",
		http.MethodPost + " " + managementBasePath + "/users",
		http.MethodGet + " " + managementBasePath + "/namespaces/" + fixture.namespaceID + "/principal-user-links/" + fixture.principalID,
		http.MethodPut + " " + managementBasePath + "/namespaces/" + fixture.namespaceID + "/principal-user-links/" + fixture.principalID,
		http.MethodGet + " " + managementBasePath + "/me",
	}
	fixture.Lock()
	got := append([]string(nil), fixture.provisioningCalls...)
	fixture.Unlock()
	if len(got) != len(want) {
		t.Fatalf("provisioning calls = %#v, want %#v", got, want)
	}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("provisioning calls = %#v, want %#v", got, want)
		}
	}
}

func assertBootstrapTokenFinalized(t *testing.T, tokenPath string) {
	t.Helper()
	if _, statErr := os.Stat(tokenPath); !os.IsNotExist(statErr) {
		t.Fatalf("bootstrap token still exists: %v", statErr)
	}
}
