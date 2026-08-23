package managementserver

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providerdiscovery"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
)

const (
	testNamespaceID  = "11111111-1111-4111-8111-111111111111"
	testPrincipalID  = "22222222-2222-4222-8222-222222222222"
	testCredentialID = "33333333-3333-4333-8333-333333333333"
	testRevision     = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	testAuthority    = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
)

type providerCatalogStub struct {
	listResult     providercatalog.ListResult
	detailResult   providercatalog.DetailResult
	discoveryPlan  providercatalog.DiscoveryPlan
	listErr        error
	detailErr      error
	discoveryErr   error
	lastList       providercatalog.ListRequest
	lastProviderID string
	lastDiscovery  providercatalog.DiscoverModelsRequest
	listCalls      int
	detailCalls    int
	discoveryCalls int
}

func (catalog *providerCatalogStub) List(_ context.Context, request providercatalog.ListRequest) (providercatalog.ListResult, error) {
	catalog.listCalls++
	catalog.lastList = request
	return catalog.listResult, catalog.listErr
}

func (catalog *providerCatalogStub) Get(_ context.Context, providerID string) (providercatalog.DetailResult, error) {
	catalog.detailCalls++
	catalog.lastProviderID = providerID
	return catalog.detailResult, catalog.detailErr
}

func (catalog *providerCatalogStub) PrepareDiscovery(
	_ context.Context,
	providerID string,
	request providercatalog.DiscoverModelsRequest,
) (providercatalog.DiscoveryPlan, error) {
	catalog.discoveryCalls++
	catalog.lastProviderID = providerID
	catalog.lastDiscovery = request
	return catalog.discoveryPlan, catalog.discoveryErr
}

type providerDiscoveryStub struct {
	result providerdiscovery.Result
	err    error
	last   providerdiscovery.ExecuteRequest
	calls  int
}

func (discovery *providerDiscoveryStub) Execute(
	_ context.Context,
	request providerdiscovery.ExecuteRequest,
) (providerdiscovery.Result, error) {
	discovery.calls++
	discovery.last = request
	return discovery.result, discovery.err
}

type sessionStub struct{ err error }

func (stub sessionStub) Authenticate(
	_ context.Context,
	_ string,
	namespaceID string,
	_ time.Time,
) (managementauth.AuthenticatedSession, error) {
	if stub.err != nil {
		return managementauth.AuthenticatedSession{}, stub.err
	}
	return managementauth.AuthenticatedSession{
		NamespaceID: namespaceID,
		Session: managementauth.LiveSession{Session: managementauth.Session{
			PrincipalID: testPrincipalID,
		}},
	}, nil
}

type authorizerStub struct {
	decision AuthorizationDecision
	err      error
	last     AuthorizationRequest
	calls    int
}

func (stub *authorizerStub) Authorize(_ context.Context, request AuthorizationRequest) (AuthorizationDecision, error) {
	stub.calls++
	stub.last = request
	return stub.decision, stub.err
}

func TestProviderRoutesListAndDetailExposeOnlyPublicCatalogFields(t *testing.T) {
	provider := catalogDefinition()
	catalog := &providerCatalogStub{
		listResult: providercatalog.ListResult{
			CatalogRevision: testRevision, Providers: []providercatalog.Definition{provider},
			Categories: []string{"Model APIs"}, PageSize: 1,
		},
		detailResult: providercatalog.DetailResult{CatalogRevision: testRevision, Provider: provider},
	}
	routes, authorizer := newTestProviderRoutes(t, catalog, &providerDiscoveryStub{}, sessionStub{})
	mux := http.NewServeMux()
	routes.Register(mux)

	for _, target := range []string{
		providerPath + "?pageSize=1&search=Provider&category=Model+APIs&capability=tools",
		providerPath + "/provider-a",
	} {
		request := authorizedRequest(t, http.MethodGet, target, nil)
		request.Header.Set(managementapi.HeaderRequestID, "request-123")
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, request)
		if response.Code != http.StatusOK {
			t.Fatalf("GET %s status = %d, body = %s", target, response.Code, response.Body.String())
		}
		if got := response.Header().Get("Content-Type"); got != managementapi.JSONMediaType {
			t.Errorf("Content-Type = %q", got)
		}
		if response.Header().Get("Cache-Control") != "no-store" ||
			response.Header().Get("Vary") != "Accept, Authorization" ||
			response.Header().Get(managementapi.HeaderRequestID) != "request-123" {
			t.Errorf("response safety headers = %#v", response.Header())
		}
		wire := response.Body.String()
		for _, forbidden := range []string{
			"protocol-adapter.v1", providercatalog.StaticBackendCompilerID,
			"credential-adapter.v1", "discovery-adapter.v1",
			"X-Internal", "/internal/invoke", "/internal/models",
		} {
			if strings.Contains(wire, forbidden) {
				t.Errorf("GET %s exposed %q: %s", target, forbidden, wire)
			}
		}
	}
	if catalog.listCalls != 1 || catalog.detailCalls != 1 || authorizer.calls != 2 {
		t.Fatalf("calls: list=%d detail=%d authz=%d", catalog.listCalls, catalog.detailCalls, authorizer.calls)
	}
	if catalog.lastList.PageSize != 1 || catalog.lastList.Search != "Provider" ||
		catalog.lastList.Category != "Model APIs" || catalog.lastList.Capability != "tools" {
		t.Fatalf("list request = %#v", catalog.lastList)
	}
}

func TestProviderRoutesDiscoveryBindsAuthorizationAndPlan(t *testing.T) {
	plan := providercatalog.DiscoveryPlan{
		CatalogRevision: testRevision, NamespaceID: testNamespaceID, ProviderID: "provider-a",
		DiscoveryAdapterID: "discovery-adapter.v1", CredentialAdapterID: "credential-adapter.v1",
		CredentialID: testCredentialID, NormalizedOrigin: "https://api.example.com",
		Path: "/internal/models", PageSize: 25,
	}
	catalog := &providerCatalogStub{discoveryPlan: plan}
	discovery := &providerDiscoveryStub{result: providerdiscovery.Result{
		Models: []providerdiscovery.Model{{
			CatalogItemID: "pmi_safe", ProviderModelID: "model-a", DisplayName: "Model A",
			Capabilities: []string{"tools"},
		}},
		CatalogRevision: testRevision, DiscoveryRevision: "signed.discovery.claim",
		ExpiresAt: time.Date(2026, 8, 22, 12, 5, 0, 0, time.UTC),
	}}
	routes, authorizer := newTestProviderRoutes(t, catalog, discovery, sessionStub{})
	mux := http.NewServeMux()
	routes.Register(mux)
	body := `{"credentialId":"` + testCredentialID + `","search":"model","pageSize":25}`
	request := authorizedRequest(t, http.MethodPost, providerPath+"/provider-a:discover-models", strings.NewReader(body))
	request.Header.Set("Content-Type", managementapi.JSONMediaType+"; charset=utf-8")
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if catalog.discoveryCalls != 1 || discovery.calls != 1 || authorizer.calls != 1 {
		t.Fatalf("calls: catalog=%d discovery=%d authz=%d", catalog.discoveryCalls, discovery.calls, authorizer.calls)
	}
	credentialTargets := authorizer.last.Targets["credential"]
	if len(credentialTargets) != 1 || credentialTargets[0].Scope != accesscontrol.ResourceScope(
		testNamespaceID, accesscontrol.ScopeResourceProviderCredential, testCredentialID,
	) || !authorizer.last.Conditions["provider_credential_supplied"] ||
		authorizer.last.Conditions["no_provider_credential_supplied"] {
		t.Fatalf("authorization request = %#v", authorizer.last)
	}
	if authorizer.last.Operation.Path != providerPath+"/{providerId}:discover-models" {
		t.Fatalf("authorization operation = %#v", authorizer.last.Operation)
	}
	if catalog.lastDiscovery.NamespaceID != testNamespaceID || catalog.lastDiscovery.CredentialID != testCredentialID ||
		catalog.lastDiscovery.Search != "model" || catalog.lastDiscovery.PageSize != 25 {
		t.Fatalf("catalog discovery request = %#v", catalog.lastDiscovery)
	}
	if discovery.last.Plan.CredentialID != testCredentialID || discovery.last.AuthorityDigest != testAuthority {
		t.Fatalf("executor request = %#v", discovery.last)
	}
	if strings.Contains(response.Body.String(), "discovery-adapter.v1") ||
		strings.Contains(response.Body.String(), "/internal/models") {
		t.Fatalf("response exposed execution internals: %s", response.Body.String())
	}
}

func TestProviderRoutesFailClosedBeforeDomainExecution(t *testing.T) {
	tests := []struct {
		name        string
		method      string
		target      string
		body        string
		contentType string
		sessionErr  error
		authzErr    error
		wantStatus  int
	}{
		{name: "unknown query", method: http.MethodGet, target: providerPath + "?offset=1", wantStatus: http.StatusBadRequest},
		{name: "duplicate query", method: http.MethodGet, target: providerPath + "?pageSize=1&pageSize=2", wantStatus: http.StatusBadRequest},
		{name: "invalid provider path", method: http.MethodGet, target: providerPath + "/Provider-A", wantStatus: http.StatusNotFound},
		{name: "unknown JSON field", method: http.MethodPost, target: providerPath + "/provider-a:discover-models", body: `{"unknown":true}`, contentType: managementapi.JSONMediaType, wantStatus: http.StatusBadRequest},
		{name: "trailing JSON", method: http.MethodPost, target: providerPath + "/provider-a:discover-models", body: `{}` + `{}`, contentType: managementapi.JSONMediaType, wantStatus: http.StatusBadRequest},
		{name: "authentication denied", method: http.MethodGet, target: providerPath, sessionErr: managementauth.ErrAuthenticationDenied, wantStatus: http.StatusUnauthorized},
		{name: "authorization denied", method: http.MethodGet, target: providerPath, authzErr: managementauthorization.ErrDenied, wantStatus: http.StatusForbidden},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			catalog := &providerCatalogStub{}
			discovery := &providerDiscoveryStub{}
			routes, authorizer := newTestProviderRoutes(t, catalog, discovery, sessionStub{err: test.sessionErr})
			authorizer.err = test.authzErr
			request := authorizedRequest(t, test.method, test.target, strings.NewReader(test.body))
			if test.contentType != "" {
				request.Header.Set("Content-Type", test.contentType)
			}
			response := httptest.NewRecorder()
			routes.ServeHTTP(response, request)
			if response.Code != test.wantStatus {
				t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
			}
			if catalog.listCalls+catalog.detailCalls+catalog.discoveryCalls != 0 || discovery.calls != 0 {
				t.Fatalf("domain executed after rejected transport/auth request: %#v %#v", catalog, discovery)
			}
		})
	}
}

func TestProviderRoutesLimitBodyAndNeverExposeInternalErrors(t *testing.T) {
	catalog := &providerCatalogStub{listErr: errors.New("database password=secret-value")}
	routes, _ := newTestProviderRoutes(t, catalog, &providerDiscoveryStub{}, sessionStub{})
	request := authorizedRequest(t, http.MethodGet, providerPath, nil)
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusServiceUnavailable || strings.Contains(response.Body.String(), "secret-value") {
		t.Fatalf("unsafe catalog error response: status=%d body=%s", response.Code, response.Body.String())
	}

	large := bytes.Repeat([]byte{'x'}, maximumDiscoveryBodyBytes+1)
	request = authorizedRequest(t, http.MethodPost, providerPath+"/provider-a:discover-models", bytes.NewReader(large))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	response = httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("oversized body status = %d, body=%s", response.Code, response.Body.String())
	}
}

func newTestProviderRoutes(
	t *testing.T,
	catalog ProviderCatalog,
	discovery ProviderDiscovery,
	sessions SessionAuthenticator,
) (*ProviderRoutes, *authorizerStub) {
	t.Helper()
	authorizer := &authorizerStub{decision: AuthorizationDecision{AuthorityDigest: testAuthority}}
	routes, err := NewProviderRoutes(ProviderRoutesOptions{
		Catalog: catalog, Discovery: discovery,
		Namespaces: NamespaceResolverFunc(func(context.Context, *http.Request) (string, error) {
			return testNamespaceID, nil
		}),
		Sessions: sessions, Authorization: authorizer,
		Now: func() time.Time { return time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatal(err)
	}
	return routes, authorizer
}

func authorizedRequest(t *testing.T, method, target string, body io.Reader) *http.Request {
	t.Helper()
	var request *http.Request
	if body == nil {
		request = httptest.NewRequest(method, target, nil)
	} else {
		request = httptest.NewRequest(method, target, body)
	}
	request.Header.Set("Authorization", "Bearer management-token")
	return request
}

func catalogDefinition() providercatalog.Definition {
	return providercatalog.Definition{
		ID: "provider-a", Revision: "sha256:" + strings.Repeat("a", 64),
		Display: providercatalog.Display{
			Name: "Provider A", Description: "A model API.", Category: "Model APIs",
			Icon: providercatalog.Icon{Source: "lobe", Value: "provider-a", Color: false},
		},
		Interfaces: []providercatalog.Interface{{
			ID: "default", Label: "Default", Default: true, WireFormat: "private.wire.v1",
			Compiler: providercatalog.Compiler{
				AdapterID: providercatalog.StaticBackendCompilerID,
				Config: map[string]any{
					"path": "/internal/invoke", "headers": map[string]any{"X-Internal": "hidden"},
				},
			},
		}},
		Credential: providercatalog.Credential{
			Mode: providercatalog.CredentialRequired, AdapterID: "credential-adapter.v1", Label: "API key",
		},
		Origin: providercatalog.Origin{Mode: providercatalog.OriginFixed, DefaultURL: "https://api.example.com"},
		Discovery: &providercatalog.Discovery{
			AdapterID: "discovery-adapter.v1", Path: "/internal/models",
			Headers: map[string]string{"X-Internal": "hidden"},
		},
		Capabilities: []string{"tools"},
	}
}
