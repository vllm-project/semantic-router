package auth

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestPolicyMuxBindsHandlerAndContractsAsOneGroup(t *testing.T) {
	t.Parallel()

	routes := NewPolicyMux()
	handler := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})
	routes.HandleGroup([]RouteContract{
		ProtectedRoute("/api/proxy/items", PermConfigRead, SensitivitySensitive, ResourceOwnerConfig, http.MethodGet),
		ProtectedMutationRoute("/api/proxy/items/", PermConfigWrite, "proxy.item.update", SensitivitySensitive, ResourceOwnerConfig, 1<<20, http.MethodPost),
	}, handler)

	for _, request := range []struct {
		method string
		path   string
	}{
		{method: http.MethodGet, path: "/api/proxy/items"},
		{method: http.MethodPost, path: "/api/proxy/items/one"},
	} {
		policy, lookup := routes.LookupRoutePolicy(request.method, request.path)
		if lookup != RouteFound || policy.Permission == "" {
			t.Fatalf("%s %s lookup = (%+v, %v)", request.method, request.path, policy, lookup)
		}
		recorder := httptest.NewRecorder()
		routes.ServeHTTP(recorder, httptest.NewRequest(request.method, request.path, nil))
		if recorder.Code != http.StatusNoContent {
			t.Fatalf("%s %s handler status = %d", request.method, request.path, recorder.Code)
		}
	}
}

func TestPolicyMuxUsesServeMuxPathMatching(t *testing.T) {
	t.Parallel()

	routes := NewPolicyMux()
	routes.HandleFunc(
		ProtectedRoute("/api/CaseSensitive/", PermConfigRead, SensitivitySensitive, ResourceOwnerConfig, http.MethodGet),
		func(http.ResponseWriter, *http.Request) {},
	)

	if _, lookup := routes.LookupRoutePolicy(http.MethodGet, "/api/CaseSensitive/item"); lookup != RouteFound {
		t.Fatalf("matching-case lookup = %v, want %v", lookup, RouteFound)
	}
	if _, lookup := routes.LookupRoutePolicy(http.MethodGet, "/api/casesensitive/item"); lookup != RouteNotFound {
		t.Fatalf("different-case lookup = %v, want %v", lookup, RouteNotFound)
	}
}

func TestPolicyMuxRejectsIncompleteGroupBeforeRegistration(t *testing.T) {
	t.Parallel()

	routes := NewPolicyMux()
	defer func() {
		if recover() == nil {
			t.Fatal("expected incomplete route group to panic")
		}
		if contracts := routes.Contracts(); len(contracts) != 0 {
			t.Fatalf("contracts registered before validation completed: %+v", contracts)
		}
	}()
	routes.HandleGroup([]RouteContract{
		ProtectedRoute("/api/valid", PermConfigRead, SensitivitySensitive, ResourceOwnerConfig, http.MethodGet),
		{Pattern: "/api/incomplete"},
	}, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
}

func TestPolicyMuxSealRejectsLateRegistration(t *testing.T) {
	t.Parallel()

	routes := NewPolicyMux()
	routes.HandleFunc(PublicRoute("/health", http.MethodGet), func(http.ResponseWriter, *http.Request) {})
	routes.Seal()

	defer func() {
		if recover() == nil {
			t.Fatal("expected late route registration to panic")
		}
	}()
	routes.HandleFunc(PublicRoute("/ready", http.MethodGet), func(http.ResponseWriter, *http.Request) {})
}

func TestValidateRouteContractRejectsIncompletePolicies(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		contract RouteContract
		want     string
	}{
		{
			name: "missing permission",
			contract: Route("/api/test", RoutePolicy{
				Method:        http.MethodPost,
				AuditMode:     AuditRequired,
				AuditAction:   "test.write",
				Sensitivity:   SensitivitySecret,
				ResourceOwner: ResourceOwnerConfig,
			}),
			want: "has no permission",
		},
		{
			name: "missing audit declaration",
			contract: Route("/api/test", RoutePolicy{
				Method:        http.MethodPost,
				Permission:    PermConfigWrite,
				Sensitivity:   SensitivitySecret,
				ResourceOwner: ResourceOwnerConfig,
			}),
			want: "invalid audit mode",
		},
		{
			name: "duplicate method",
			contract: Route(
				"/api/test",
				ReadPolicy(http.MethodGet, PermConfigRead, SensitivitySensitive, ResourceOwnerConfig),
				ReadPolicy(http.MethodGet, PermConfigRead, SensitivitySensitive, ResourceOwnerConfig),
			),
			want: "duplicate policy",
		},
		{
			name: "mixed resource ownership",
			contract: ProtectedMutationRoute(
				"/api/test",
				PermConfigWrite,
				"test.write",
				SensitivitySecret,
				ResourceOwner("config,tenant_grants"),
				64<<10,
				http.MethodPost,
			),
			want: "invalid resource owner",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			err := ValidateRouteContract(test.contract)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("ValidateRouteContract() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestGenericConfigWriteCannotOwnEnterpriseSecurityResources(t *testing.T) {
	t.Parallel()

	tests := []struct {
		owner      ResourceOwner
		permission string
	}{
		{owner: ResourceOwnerTenantGrants, permission: PermGrantPublish},
		{owner: ResourceOwnerTenantQuotas, permission: PermQuotaPublish},
		{owner: ResourceOwnerVirtualKeys, permission: PermVirtualKeys},
		{owner: ResourceOwnerAuditPolicy, permission: PermAuditPolicy},
		{owner: ResourceOwnerBreakGlass, permission: PermBreakGlass},
	}

	for _, test := range tests {
		t.Run(string(test.owner), func(t *testing.T) {
			t.Parallel()
			generic := ProtectedMutationRoute(
				"/api/enterprise/"+string(test.owner),
				PermConfigWrite,
				"enterprise.update",
				SensitivitySecret,
				test.owner,
				64<<10,
				http.MethodPost,
			)
			if err := ValidateRouteContract(generic); err == nil {
				t.Fatal("generic config.write unexpectedly owns an isolated resource")
			}

			isolated := ProtectedMutationRoute(
				"/api/enterprise/"+string(test.owner),
				test.permission,
				"enterprise.update",
				SensitivitySecret,
				test.owner,
				64<<10,
				http.MethodPost,
			)
			if test.owner == ResourceOwnerBreakGlass {
				isolated = BreakGlassMutationRoute(
					"/api/enterprise/"+string(test.owner),
					"enterprise.breakglass",
					64<<10,
					10*time.Minute,
					http.MethodPost,
				)
			}
			if err := ValidateRouteContract(isolated); err != nil {
				t.Fatalf("isolated permission rejected: %v", err)
			}
		})
	}
}
