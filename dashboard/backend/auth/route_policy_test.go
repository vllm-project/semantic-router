package auth

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestValidateRouteContractRejectsIncompletePolicies(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		contract RouteContract
		want     string
	}{
		{name: "relative pattern", contract: PublicRoute("healthz", http.MethodGet), want: "absolute path"},
		{name: "no methods", contract: Route("/api/test"), want: "no method policies"},
		{
			name: "missing permission",
			contract: Route("/api/test", RoutePolicy{
				Method: http.MethodPost, AuditMode: AuditRequired, AuditAction: "test.write",
				Sensitivity: SensitivitySecret, ResourceOwner: ResourceOwnerConfig,
			}),
			want: "has no permission",
		},
		{
			name: "missing audit action",
			contract: Route("/api/test", RoutePolicy{
				Method: http.MethodPost, Permissions: []string{PermConfigWrite}, AuditMode: AuditRequired,
				Sensitivity: SensitivitySecret, ResourceOwner: ResourceOwnerConfig,
			}),
			want: "requires an audit action",
		},
		{
			name: "invalid audit mode",
			contract: Route("/api/test", RoutePolicy{
				Method: http.MethodPost, Permissions: []string{PermConfigWrite}, AuditMode: AuditMode("sometimes"),
				AuditAction: "test.write", Sensitivity: SensitivitySecret, ResourceOwner: ResourceOwnerConfig,
			}),
			want: "invalid audit mode",
		},
		{
			name:     "invalid sensitivity",
			contract: Route("/api/test", ReadPolicy(http.MethodGet, PermConfigRead, Sensitivity("mild"), ResourceOwnerConfig)),
			want:     "invalid sensitivity",
		},
		{
			name:     "invalid owner",
			contract: Route("/api/test", ReadPolicy(http.MethodGet, PermConfigRead, SensitivitySensitive, ResourceOwner("tenant"))),
			want:     "invalid resource owner",
		},
		{
			name: "duplicate method",
			contract: Route("/api/test",
				ReadPolicy(http.MethodGet, PermConfigRead, SensitivitySensitive, ResourceOwnerConfig),
				ReadPolicy(http.MethodGet, PermConfigRead, SensitivitySensitive, ResourceOwnerConfig),
			),
			want: "duplicate policy",
		},
		{
			name:     "public route with permission",
			contract: Route("/api/test", PublicPolicy(http.MethodGet).AlsoRequiring(PermConfigRead)),
			want:     "must not require a permission",
		},
		{
			name: "public route demanding audit",
			contract: Route("/api/test", RoutePolicy{
				Method: http.MethodPost, Public: true, AuditMode: AuditRequired, AuditAction: "x",
				Sensitivity: SensitivityPublic, ResourceOwner: ResourceOwnerPublic,
			}),
			want: "without a permission",
		},
		{
			name:     "negative body limit",
			contract: ProtectedBoundedRoute("/api/test", PermConfigRead, SensitivitySensitive, ResourceOwnerConfig, -1, http.MethodPost),
			want:     "negative body limit",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			err := ValidateRouteContract(test.contract)
			if test.want == "" {
				if err != nil {
					t.Fatalf("ValidateRouteContract() error = %v, want nil", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("ValidateRouteContract() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestPolicyMuxBindsHandlerAndContractsAsOneGroup(t *testing.T) {
	t.Parallel()

	routes := NewPolicyMux()
	handler := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) })
	routes.HandleGroup([]RouteContract{
		ProtectedRoute("/api/items", PermConfigRead, SensitivitySensitive, ResourceOwnerConfig, http.MethodGet),
		ProtectedMutationRoute("/api/items/{id}", PermConfigWrite, "item.update", SensitivitySensitive, ResourceOwnerConfig, 1<<20, http.MethodPost),
	}, handler)

	for _, request := range []struct {
		method, path, permission string
	}{
		{http.MethodGet, "/api/items", PermConfigRead},
		{http.MethodPost, "/api/items/one", PermConfigWrite},
	} {
		policy, lookup := routes.LookupRoutePolicy(request.method, request.path)
		if lookup != RouteFound || len(policy.Permissions) != 1 || policy.Permissions[0] != request.permission {
			t.Fatalf("%s %s lookup = (%+v, %v)", request.method, request.path, policy, lookup)
		}
		recorder := httptest.NewRecorder()
		routes.ServeHTTP(recorder, httptest.NewRequest(request.method, request.path, nil))
		if recorder.Code != http.StatusNoContent {
			t.Fatalf("%s %s handler status = %d", request.method, request.path, recorder.Code)
		}
	}
	if _, lookup := routes.LookupRoutePolicy(http.MethodDelete, "/api/items/one"); lookup != RouteMethodNotAllowed {
		t.Fatalf("undeclared method lookup = %v, want %v", lookup, RouteMethodNotAllowed)
	}
	if policy, lookup := routes.LookupRoutePolicy(http.MethodOptions, "/api/items/one"); lookup != RouteFound || !policy.Public {
		t.Fatalf("OPTIONS lookup = (%+v, %v), want public preflight", policy, lookup)
	}
	if _, lookup := routes.LookupRoutePolicy(http.MethodGet, "/api/items/one/extra"); lookup != RouteNotFound {
		t.Fatalf("deeper path lookup = %v, want %v", lookup, RouteNotFound)
	}
	if _, lookup := routes.LookupRoutePolicy(http.MethodGet, "/api/Items"); lookup != RouteNotFound {
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

func TestPolicyMuxRejectsProtectedFallbackAndLateRegistration(t *testing.T) {
	t.Parallel()

	routes := NewPolicyMux()
	noop := http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	expectPanic := func(name string, register func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatalf("%s did not panic", name)
			}
		}()
		register()
	}
	expectPanic("protected fallback", func() { routes.HandleFallback("/api/", noop) })
	expectPanic("embedded fallback", func() { routes.HandleFallback("/embedded/", noop) })
	routes.HandleFallback("/", noop)
	routes.HandleFunc(PublicRoute("/healthz", http.MethodGet), noop)
	expectPanic("duplicate pattern", func() { routes.HandleFunc(PublicRoute("/healthz", http.MethodGet), noop) })
	routes.Seal()
	expectPanic("late registration", func() { routes.HandleFunc(PublicRoute("/ready", http.MethodGet), noop) })
}
