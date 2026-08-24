package routerauth

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

func TestFirstAdminProvisionerCompletesRouterAuthorityAndFinalizesToken(t *testing.T) {
	t.Parallel()
	now := time.Date(2026, time.August, 23, 16, 0, 0, 0, time.UTC)
	issuerID := "10000000-0000-4000-8000-000000000011"
	principalID := "10000000-0000-4000-8000-000000000012"
	namespaceID := "10000000-0000-4000-8000-000000000013"
	quotaPartitionID := "10000000-0000-4000-8000-000000000014"
	userID := "10000000-0000-4000-8000-000000000015"
	roleID := "10000000-0000-5000-8000-000000000002"
	identity := dashboardauth.FirstAdminIdentity{
		UserID:    "10000000-0000-4000-8000-000000000016",
		SessionID: "10000000-0000-4000-8000-000000000017",
		Email:     "admin@example.test", DisplayName: "Admin",
		AuthenticatedAt: now,
		ExpiresAt:       now.Add(12 * time.Hour),
	}
	secretDir := t.TempDir()
	tokenPath := filepath.Join(secretDir, "router-token")
	if err := os.WriteFile(tokenPath, []byte("router-bootstrap-token-which-is-at-least-32-bytes"), 0o600); err != nil {
		t.Fatal(err)
	}

	type state struct {
		sync.Mutex
		bootstrapped, namespace, user, linked, binding bool
		challenge                                      int
		provisioningCalls                              []string
	}
	installation := &state{}
	write := func(response http.ResponseWriter, status int, payload any) {
		response.Header().Set("Content-Type", managementMediaType)
		response.WriteHeader(status)
		if payload != nil {
			_ = json.NewEncoder(response).Encode(payload)
		}
	}
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		installation.Lock()
		defer installation.Unlock()
		if request.Header.Get("Accept") != managementMediaType {
			t.Errorf("Accept = %q", request.Header.Get("Accept"))
		}
		switch request.URL.Path {
		case managementBasePath + "/auth/exchange-challenges":
			if !installation.bootstrapped {
				http.Error(response, "not bootstrapped", http.StatusServiceUnavailable)
				return
			}
			installation.challenge++
			write(response, http.StatusCreated, exchangeChallenge{
				ExchangeChallengeID: "10000000-0000-4000-8000-000000000020",
				Nonce:               "nonce", ExpiresAt: now.Add(time.Minute),
			})
		case managementBasePath + "/auth/bootstrap":
			if request.Header.Get("Authorization") != "VSR-Bootstrap router-bootstrap-token-which-is-at-least-32-bytes" ||
				request.Header.Get(managementapi.HeaderIdempotencyKey) != installationKey("bootstrap", identity.UserID) {
				t.Errorf("bootstrap headers = %#v", request.Header)
			}
			var body managementapi.BootstrapRequest
			if err := json.NewDecoder(request.Body).Decode(&body); err != nil || body.External == nil ||
				body.Kind != "external_principal" || body.External.Subject != identity.UserID ||
				body.External.IssuerID != issuerID {
				t.Errorf("bootstrap body = %#v error=%v", body, err)
			}
			installation.bootstrapped = true
			write(response, http.StatusCreated, managementapi.BootstrapResponse{
				PrincipalID: principalID, RoleBindingID: "10000000-0000-4000-8000-000000000021",
				FinalizationRequired: true,
			})
		case managementBasePath + "/auth/token-exchange":
			write(response, http.StatusOK, managementTokenEnvelope{
				AccessToken: "cluster-admin-token", TokenType: "Bearer", ExpiresIn: 300,
				ManagementSessionID: "10000000-0000-4000-8000-000000000022",
			})
		case managementBasePath + "/me":
			installation.provisioningCalls = append(installation.provisioningCalls, request.Method+" "+request.URL.Path)
			payload := managementapi.Me{Principal: managementapi.MePrincipal{PrincipalID: principalID}}
			if installation.namespace && installation.user && installation.linked && installation.binding {
				payload.Namespaces = []managementapi.MeNamespaceScope{{
					Namespace: managementapi.MeNamespace{NamespaceID: namespaceID, Name: defaultNamespaceName, Status: "active"},
					User:      &managementapi.MeUser{UserID: userID, Email: identity.Email, DisplayName: identity.DisplayName, Status: "active"},
					RoleBindings: []managementapi.ManagementRoleBinding{{
						RoleID: roleID, PrincipalID: principalID, Status: "active",
						Scope: managementapi.ManagementScope{Kind: "namespace", NamespaceID: namespaceID},
					}},
				}}
			}
			write(response, http.StatusOK, payload)
		case managementBasePath + "/namespaces":
			installation.provisioningCalls = append(installation.provisioningCalls, request.Method+" "+request.URL.Path)
			if request.Method == http.MethodGet {
				items := []namespaceView{}
				if installation.namespace {
					items = append(items, namespaceView{
						NamespaceID: namespaceID, Name: defaultNamespaceName,
						QuotaPartitionID: quotaPartitionID, BillingCurrency: defaultBillingCurrency,
						Status: "active", Revision: 1, RuntimeEpoch: 1, CreatedAt: now, UpdatedAt: now,
					})
				}
				write(response, http.StatusOK, namespacePage{Data: items, Page: managementapi.PageInfo{PageSize: 200}})
				return
			}
			installation.namespace = true
			write(response, http.StatusCreated, managementapi.NewResourceMutationReceipt("namespace", namespaceID, 1, nil))
		case managementBasePath + "/users":
			installation.provisioningCalls = append(installation.provisioningCalls, request.Method+" "+request.URL.Path)
			if request.Header.Get(managementapi.HeaderNamespaceID) != namespaceID {
				t.Errorf("user namespace header = %q", request.Header.Get(managementapi.HeaderNamespaceID))
			}
			if request.Method == http.MethodPost && !installation.binding {
				http.Error(response, "platform admin binding required", http.StatusForbidden)
				return
			}
			if request.Method == http.MethodGet {
				items := []managementapi.UserView{}
				if installation.user {
					items = append(items, managementapi.UserView{
						UserID: userID, Email: identity.Email,
						DisplayName: identity.DisplayName, Status: "active", Revision: 1, CreatedAt: now, UpdatedAt: now,
					})
				}
				write(response, http.StatusOK, managementapi.UserPage{Data: items, Page: managementapi.PageInfo{PageSize: 200}})
				return
			}
			installation.user = true
			write(response, http.StatusCreated, managementapi.NewResourceMutationReceipt("user", userID, 1, nil))
		case managementBasePath + "/management-roles":
			installation.provisioningCalls = append(installation.provisioningCalls, request.Method+" "+request.URL.Path)
			write(response, http.StatusOK, managementapi.Page[managementapi.ManagementRole]{
				Data: []managementapi.ManagementRole{{
					RoleID: roleID, Name: platformAdminRoleName,
					DisplayName: "Platform admin", Permissions: []string{"namespace.read"}, BuiltIn: true,
					Status: "active", Revision: 1, CreatedAt: now, UpdatedAt: now,
				}},
				Page: managementapi.PageInfo{PageSize: 200},
			})
		case managementBasePath + "/namespaces/" + namespaceID + "/principal-user-links/" + principalID:
			installation.provisioningCalls = append(installation.provisioningCalls, request.Method+" "+request.URL.Path)
			if request.Method == http.MethodGet && !installation.linked {
				http.NotFound(response, request)
				return
			}
			installation.linked = true
			write(response, map[bool]int{true: http.StatusOK, false: http.StatusCreated}[request.Method == http.MethodGet],
				managementapi.PrincipalUserLinkDetail{Data: managementapi.PrincipalUserLink{
					PrincipalID: principalID, NamespaceID: namespaceID, UserID: userID, Revision: 1,
					CreatedAt: now, UpdatedAt: now,
				}})
		case managementBasePath + "/role-bindings":
			installation.provisioningCalls = append(installation.provisioningCalls, request.Method+" "+request.URL.Path)
			if request.Method == http.MethodGet {
				items := []managementapi.ManagementRoleBinding{}
				if installation.binding {
					items = append(items, managementapi.ManagementRoleBinding{
						BindingID:   "10000000-0000-4000-8000-000000000023",
						PrincipalID: principalID, RoleID: roleID, Scope: managementapi.ManagementScope{Kind: "namespace", NamespaceID: namespaceID},
						Status: "active", Revision: 1, CreatedAt: now, UpdatedAt: now,
					})
				}
				write(response, http.StatusOK, managementapi.Page[managementapi.ManagementRoleBinding]{Data: items, Page: managementapi.PageInfo{PageSize: 200}})
				return
			}
			installation.binding = true
			write(response, http.StatusCreated, managementapi.NewResourceMutationReceipt("management_role_binding",
				"10000000-0000-4000-8000-000000000023", 1, nil))
		default:
			http.NotFound(response, request)
		}
	}))
	defer server.Close()

	provider, err := NewManagementSessionProvider(ManagementSessionOptions{
		RouterURL: server.URL, IssuerURL: "https://dashboard.example.test", IssuerID: issuerID,
		Signer: &recordingAssertionSigner{}, Client: server.Client(), Now: func() time.Time { return now },
		BootstrapTokenFile: tokenPath,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := provider.ProvisionFirstAdmin(context.Background(), identity); err != nil {
		t.Fatalf("ProvisionFirstAdmin() error = %v", err)
	}
	wantProvisioningCalls := []string{
		http.MethodGet + " " + managementBasePath + "/me",
		http.MethodGet + " " + managementBasePath + "/namespaces",
		http.MethodPost + " " + managementBasePath + "/namespaces",
		http.MethodGet + " " + managementBasePath + "/management-roles",
		http.MethodGet + " " + managementBasePath + "/role-bindings",
		http.MethodPost + " " + managementBasePath + "/role-bindings",
		http.MethodGet + " " + managementBasePath + "/users",
		http.MethodPost + " " + managementBasePath + "/users",
		http.MethodGet + " " + managementBasePath + "/namespaces/" + namespaceID + "/principal-user-links/" + principalID,
		http.MethodPut + " " + managementBasePath + "/namespaces/" + namespaceID + "/principal-user-links/" + principalID,
		http.MethodGet + " " + managementBasePath + "/me",
	}
	installation.Lock()
	gotProvisioningCalls := append([]string(nil), installation.provisioningCalls...)
	installation.Unlock()
	if len(gotProvisioningCalls) != len(wantProvisioningCalls) {
		t.Fatalf("provisioning calls = %#v, want %#v", gotProvisioningCalls, wantProvisioningCalls)
	}
	for index := range wantProvisioningCalls {
		if gotProvisioningCalls[index] != wantProvisioningCalls[index] {
			t.Fatalf("provisioning calls = %#v, want %#v", gotProvisioningCalls, wantProvisioningCalls)
		}
	}
	if _, err := os.Stat(tokenPath); !os.IsNotExist(err) {
		t.Fatalf("bootstrap token still exists: %v", err)
	}
	if err := provider.ProvisionFirstAdmin(context.Background(), identity); err != nil {
		t.Fatalf("idempotent ProvisionFirstAdmin() error = %v", err)
	}
}

func TestBootstrapTokenRequiresOwnerOnlyFileAndRefusesReplacement(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	path := filepath.Join(directory, "router-token")
	if err := os.WriteFile(path, []byte("router-bootstrap-token-which-is-at-least-32-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := observeBootstrapToken(path); err == nil {
		t.Fatal("observeBootstrapToken() accepted group/world-readable secret")
	}
	if err := os.Chmod(path, 0o600); err != nil {
		t.Fatal(err)
	}
	observed, err := observeBootstrapToken(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("replacement-bootstrap-token-at-least-32-bytes"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := finalizeBootstrapToken(path, observed); err == nil {
		t.Fatal("finalizeBootstrapToken() removed a replacement file")
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("replacement token was removed: %v", err)
	}
}

func TestFinalizeBootstrapTokenRejectsReplacementAfterFileIdentityReuse(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	path := filepath.Join(directory, "router-token")
	original := []byte("router-bootstrap-token-which-is-at-least-32-bytes")
	if err := os.WriteFile(path, original, 0o600); err != nil {
		t.Fatal(err)
	}
	observed, err := observeBootstrapToken(path)
	if err != nil {
		t.Fatal(err)
	}
	if removeErr := os.Remove(path); removeErr != nil {
		t.Fatal(removeErr)
	}
	replacement := []byte("replacement-bootstrap-token-at-least-32-bytes")
	if writeErr := os.WriteFile(path, replacement, 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	replacementInfo, err := os.Lstat(path)
	if err != nil {
		t.Fatal(err)
	}
	// Model inode reuse explicitly: a FileInfo comparison alone now reports
	// the replacement as the observed file, while the stable content identity
	// still describes the credential that was actually consumed.
	observed.fileInfo = replacementInfo
	if err := finalizeBootstrapToken(path, observed); err == nil {
		t.Fatal("finalizeBootstrapToken() accepted different content after file identity reuse")
	}
	if payload, err := os.ReadFile(path); err != nil || string(payload) != string(replacement) {
		t.Fatalf("replacement token = %q, %v", payload, err)
	}
}

func TestFinalizeBootstrapTokenRejectsRepeatedReplacement(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	path := filepath.Join(directory, "router-token")
	for iteration := 0; iteration < 128; iteration++ {
		original := []byte(fmt.Sprintf("router-bootstrap-token-original-%032d", iteration))
		if err := os.WriteFile(path, original, 0o600); err != nil {
			t.Fatal(err)
		}
		observed, err := observeBootstrapToken(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		replacement := []byte(fmt.Sprintf("router-bootstrap-token-replaced-%032d", iteration))
		if err := os.WriteFile(path, replacement, 0o600); err != nil {
			t.Fatal(err)
		}
		if err := finalizeBootstrapToken(path, observed); err == nil {
			t.Fatalf("iteration %d accepted replacement token", iteration)
		}
		if payload, err := os.ReadFile(path); err != nil || string(payload) != string(replacement) {
			t.Fatalf("iteration %d replacement token = %q, %v", iteration, payload, err)
		}
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
	}
}

func TestFinalizeBootstrapTokenAcceptsConcurrentRemoval(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	path := filepath.Join(directory, "router-token")
	if err := os.WriteFile(path, []byte("router-bootstrap-token-which-is-at-least-32-bytes"), 0o600); err != nil {
		t.Fatal(err)
	}
	observed, err := observeBootstrapToken(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := finalizeBootstrapToken(path, observed); err != nil {
		t.Fatalf("finalizeBootstrapToken() after concurrent removal = %v", err)
	}
}

func TestFinalizeVerifiedBootstrapTokenClaimRejectsReplacement(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	path := filepath.Join(directory, "router-token")
	claimDirectory := filepath.Join(directory, ".vllm-sr-bootstrap-finalize-test")
	claimPath := filepath.Join(claimDirectory, "token")
	if err := os.Mkdir(claimDirectory, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(claimPath, []byte("router-bootstrap-token-which-is-at-least-32-bytes"), 0o600); err != nil {
		t.Fatal(err)
	}
	replacement := []byte("replacement-bootstrap-token-at-least-32-bytes")
	if err := os.WriteFile(path, replacement, 0o600); err != nil {
		t.Fatal(err)
	}

	if err := finalizeVerifiedBootstrapTokenClaim(path, claimPath, claimDirectory); err == nil {
		t.Fatal("finalizeVerifiedBootstrapTokenClaim() accepted a replacement token")
	}
	if payload, err := os.ReadFile(path); err != nil || string(payload) != string(replacement) {
		t.Fatalf("replacement token = %q, %v", payload, err)
	}
	if _, err := os.Stat(claimPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("verified token claim still exists: %v", err)
	}
	if _, err := os.Stat(claimDirectory); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("token claim directory still exists: %v", err)
	}
}
