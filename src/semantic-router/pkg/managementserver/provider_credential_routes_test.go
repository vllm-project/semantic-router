package managementserver

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	accesspostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	credentialmanagement "github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential/management"
)

func TestProviderCredentialRoutesPushExactScopeBeforePagination(t *testing.T) {
	service := &providerCredentialServiceStub{listResult: credentialmanagement.ListResult{
		Credentials: []credentialmanagement.Metadata{testCredentialMetadata()},
		NextCursor:  "opaque-credential-cursor", HasMore: true, PageSize: 1,
	}}
	routes, authorizer := newTestProviderCredentialRoutes(t, service)
	routes.scopes = resultScopeResolverFunc(func(_ context.Context, _ accesscontrol.ManagementPrincipalID,
		namespaceID accesscontrol.NamespaceID, permission accesscontrol.Permission,
	) (managementauthorization.ResultScope, error) {
		if permission != accesscontrol.PermissionProviderCredentialRead {
			t.Fatalf("permission = %q", permission)
		}
		return managementauthorization.ResultScope{
			NamespaceID: namespaceID,
			ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
				accesscontrol.ScopeResourceProviderCredential: {testCredentialID},
			},
		}, nil
	})
	request := authorizedRequest(t, http.MethodGet, providerCredentialPath+"?pageSize=1", nil)
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	ids := service.lastList.Scope.IDs(accesscontrol.ScopeResourceProviderCredential)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), testCredentialID) ||
		!strings.Contains(response.Body.String(), "opaque-credential-cursor") || authorizer.calls != 0 ||
		len(ids) != 1 || ids[0] != testCredentialID {
		t.Fatalf("status=%d body=%s authCalls=%d request=%#v", response.Code,
			response.Body.String(), authorizer.calls, service.lastList)
	}
}

func TestProviderCredentialRoutesUseSafeReceiptsAndStrictTransport(t *testing.T) {
	service := &providerCredentialServiceStub{
		createResult: credentialmanagement.MutationResult{
			CredentialID: testCredentialID, Revision: 1, Replayed: true,
		},
		rotateResult: credentialmanagement.MutationResult{CredentialID: testCredentialID, Revision: 2},
		getResult:    testCredentialMetadata(),
		listResult: credentialmanagement.ListResult{
			Credentials: []credentialmanagement.Metadata{testCredentialMetadata()}, PageSize: 50,
		},
	}
	routes, _ := newTestProviderCredentialRoutes(t, service)

	createBody := `{"name":"Primary","providerId":"provider-a","secret":"backend-secret"}`
	create := authorizedRequest(t, http.MethodPost, providerCredentialPath, strings.NewReader(createBody))
	create.Header.Set("Content-Type", managementapi.JSONMediaType)
	create.Header.Set(managementapi.HeaderIdempotencyKey, "create-key-0123456789")
	created := httptest.NewRecorder()
	routes.ServeHTTP(created, create)
	if created.Code != http.StatusCreated {
		t.Fatalf("create status=%d body=%s", created.Code, created.Body.String())
	}
	if created.Header().Get(managementapi.HeaderIdempotencyReplayed) != "true" ||
		created.Header().Get(managementapi.HeaderETag) != `"pc:1"` ||
		created.Header().Get("Location") != providerCredentialPath+"/"+testCredentialID {
		t.Fatalf("create headers = %#v", created.Header())
	}
	for _, forbidden := range []string{"backend-secret", "secret", "ciphertext", "adapter", "activeVersion", "desiredRevision"} {
		if strings.Contains(strings.ToLower(created.Body.String()), strings.ToLower(forbidden)) {
			t.Fatalf("create response exposed %q: %s", forbidden, created.Body.String())
		}
	}
	for _, expected := range []string{`"kind":"provider_credential"`, `"id":"` + testCredentialID + `"`, `"revision":1`, `"replayed":true`} {
		if !strings.Contains(created.Body.String(), expected) {
			t.Fatalf("create response omitted %s: %s", expected, created.Body.String())
		}
	}
	if string(service.lastCreate.Secret) != "backend-secret" || service.lastCreate.ProviderID != "provider-a" {
		t.Fatalf("create request = %#v", service.lastCreate)
	}

	detail := authorizedRequest(t, http.MethodGet, providerCredentialPath+"/"+testCredentialID, nil)
	detailed := httptest.NewRecorder()
	routes.ServeHTTP(detailed, detail)
	if detailed.Code != http.StatusOK || strings.Contains(detailed.Body.String(), "activeVersion") ||
		strings.Contains(detailed.Body.String(), "adapter") {
		t.Fatalf("detail status=%d body=%s", detailed.Code, detailed.Body.String())
	}

	rotate := authorizedRequest(t, http.MethodPost, providerCredentialPath+"/"+testCredentialID+":rotate", strings.NewReader(`{"secret":"next-secret"}`))
	rotate.Header.Set("Content-Type", managementapi.JSONMediaType+"; charset=utf-8")
	rotate.Header.Set(managementapi.HeaderIdempotencyKey, "rotate-key-0123456789")
	rotate.Header.Set(managementapi.HeaderIfMatch, `"pc:1"`)
	rotated := httptest.NewRecorder()
	routes.ServeHTTP(rotated, rotate)
	if rotated.Code != http.StatusOK || rotated.Header().Get(managementapi.HeaderETag) != `"pc:2"` ||
		strings.Contains(rotated.Body.String(), "next-secret") {
		t.Fatalf("rotate status=%d headers=%#v body=%s", rotated.Code, rotated.Header(), rotated.Body.String())
	}
	if service.lastRotate.ExpectedRevision != 1 || string(service.lastRotate.Secret) != "next-secret" {
		t.Fatalf("rotate request = %#v", service.lastRotate)
	}
}

func TestProviderCredentialRoutesFailClosedBeforeService(t *testing.T) {
	tests := []struct {
		name        string
		method      string
		target      string
		body        string
		contentType string
		idempotency string
		ifMatch     string
		wantStatus  int
	}{
		{name: "generic JSON rejected", method: http.MethodPost, target: providerCredentialPath, body: `{}`, contentType: "application/json", idempotency: "create-key-0123456789", wantStatus: http.StatusUnsupportedMediaType},
		{name: "missing idempotency", method: http.MethodPost, target: providerCredentialPath, body: `{}`, contentType: managementapi.JSONMediaType, wantStatus: http.StatusBadRequest},
		{name: "unknown create field", method: http.MethodPost, target: providerCredentialPath, body: `{"name":"x","providerId":"p","secret":"s","rawSecret":true}`, contentType: managementapi.JSONMediaType, idempotency: "create-key-0123456789", wantStatus: http.StatusBadRequest},
		{name: "missing CAS", method: http.MethodPatch, target: providerCredentialPath + "/" + testCredentialID, body: `{"name":"next"}`, contentType: managementapi.JSONMediaType, wantStatus: http.StatusPreconditionRequired},
		{name: "invalid CAS", method: http.MethodDelete, target: providerCredentialPath + "/" + testCredentialID, ifMatch: `"1"`, wantStatus: http.StatusBadRequest},
		{name: "unknown action", method: http.MethodPost, target: providerCredentialPath + "/" + testCredentialID + ":reveal", body: `{}`, contentType: managementapi.JSONMediaType, idempotency: "action-key-0123456789", ifMatch: `"pc:1"`, wantStatus: http.StatusNotFound},
		{name: "offset forbidden", method: http.MethodGet, target: providerCredentialPath + "?offset=1", wantStatus: http.StatusBadRequest},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			service := &providerCredentialServiceStub{}
			routes, _ := newTestProviderCredentialRoutes(t, service)
			request := authorizedRequest(t, test.method, test.target, strings.NewReader(test.body))
			if test.contentType != "" {
				request.Header.Set("Content-Type", test.contentType)
			}
			if test.idempotency != "" {
				request.Header.Set(managementapi.HeaderIdempotencyKey, test.idempotency)
			}
			if test.ifMatch != "" {
				request.Header.Set(managementapi.HeaderIfMatch, test.ifMatch)
			}
			response := httptest.NewRecorder()
			routes.ServeHTTP(response, request)
			if response.Code != test.wantStatus {
				t.Fatalf("status=%d want=%d body=%s", response.Code, test.wantStatus, response.Body.String())
			}
			if service.calls != 0 {
				t.Fatalf("service called %d times after rejected request", service.calls)
			}
		})
	}
}

func TestProviderCredentialRoutesMapStableDomainErrors(t *testing.T) {
	for name, test := range map[string]struct {
		err        error
		wantStatus int
		wantCode   string
	}{
		"idempotency":    {managementcommand.ErrConflict, http.StatusConflict, "idempotency_conflict"},
		"unsafe origin":  {credentialmanagement.ErrUnsafeOrigin, http.StatusBadRequest, "origin_denied"},
		"revision":       {accesspostgres.ErrRevisionConflict, http.StatusPreconditionFailed, "revision_conflict"},
		"provider drift": {credentialmanagement.ErrProviderMismatch, http.StatusConflict, "credential_unavailable"},
	} {
		t.Run(name, func(t *testing.T) {
			service := &providerCredentialServiceStub{createErr: test.err}
			routes, _ := newTestProviderCredentialRoutes(t, service)
			request := authorizedRequest(t, http.MethodPost, providerCredentialPath, strings.NewReader(`{"name":"Primary","providerId":"provider-a","secret":"secret"}`))
			request.Header.Set("Content-Type", managementapi.JSONMediaType)
			request.Header.Set(managementapi.HeaderIdempotencyKey, "create-key-0123456789")
			response := httptest.NewRecorder()
			routes.ServeHTTP(response, request)
			if response.Code != test.wantStatus || !strings.Contains(response.Body.String(), `"code":"`+test.wantCode+`"`) {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
		})
	}
}

type providerCredentialServiceStub struct {
	createResult credentialmanagement.MutationResult
	rotateResult credentialmanagement.MutationResult
	getResult    credentialmanagement.Metadata
	listResult   credentialmanagement.ListResult
	createErr    error
	lastCreate   credentialmanagement.CreateRequest
	lastRotate   credentialmanagement.RotateRequest
	lastList     credentialmanagement.ListRequest
	calls        int
	readyErr     error
}

func (service *providerCredentialServiceStub) Ready(context.Context) error { return service.readyErr }

func (service *providerCredentialServiceStub) Create(_ context.Context, request credentialmanagement.CreateRequest) (credentialmanagement.MutationResult, error) {
	service.calls++
	request.Secret = append([]byte(nil), request.Secret...)
	service.lastCreate = request
	return service.createResult, service.createErr
}

func (service *providerCredentialServiceStub) Get(_ context.Context, _, _ string) (credentialmanagement.Metadata, error) {
	service.calls++
	return service.getResult, nil
}

func (service *providerCredentialServiceStub) List(_ context.Context, request credentialmanagement.ListRequest) (credentialmanagement.ListResult, error) {
	service.calls++
	service.lastList = request
	return service.listResult, nil
}

func (service *providerCredentialServiceStub) Rename(_ context.Context, request credentialmanagement.RenameRequest) (credentialmanagement.MutationResult, error) {
	service.calls++
	return credentialmanagement.MutationResult{CredentialID: request.CredentialID, Revision: request.ExpectedRevision + 1}, nil
}

func (service *providerCredentialServiceStub) Rotate(_ context.Context, request credentialmanagement.RotateRequest) (credentialmanagement.MutationResult, error) {
	service.calls++
	request.Secret = append([]byte(nil), request.Secret...)
	service.lastRotate = request
	return service.rotateResult, nil
}

func (service *providerCredentialServiceStub) Disable(_ context.Context, request credentialmanagement.LifecycleRequest) (credentialmanagement.MutationResult, error) {
	service.calls++
	return credentialmanagement.MutationResult{CredentialID: request.CredentialID, Revision: request.ExpectedRevision + 1}, nil
}

func (service *providerCredentialServiceStub) Reactivate(_ context.Context, request credentialmanagement.LifecycleRequest) (credentialmanagement.MutationResult, error) {
	service.calls++
	return credentialmanagement.MutationResult{CredentialID: request.CredentialID, Revision: request.ExpectedRevision + 1}, nil
}

func (service *providerCredentialServiceStub) Delete(_ context.Context, request credentialmanagement.LifecycleRequest) (credentialmanagement.MutationResult, error) {
	service.calls++
	return credentialmanagement.MutationResult{CredentialID: request.CredentialID, Revision: request.ExpectedRevision + 1}, nil
}

func newTestProviderCredentialRoutes(t *testing.T, service ProviderCredentialService) (*ProviderCredentialRoutes, *authorizerStub) {
	t.Helper()
	authorizer := &authorizerStub{decision: AuthorizationDecision{AuthorityDigest: testAuthority}}
	routes, err := NewProviderCredentialRoutes(ProviderCredentialRoutesOptions{
		Service: service,
		Namespaces: NamespaceResolverFunc(func(context.Context, *http.Request) (string, error) {
			return testNamespaceID, nil
		}),
		Sessions: sessionStub{}, Authorization: authorizer, Scopes: allowAllResultScopes(),
		Now: func() time.Time { return time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatal(err)
	}
	return routes, authorizer
}

func testCredentialMetadata() credentialmanagement.Metadata {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	return credentialmanagement.Metadata{
		CredentialID: testCredentialID, NamespaceID: testNamespaceID, Name: "Primary",
		ProviderID: "provider-a", CatalogRevision: testRevision,
		NormalizedOrigin: "https://api.example.com/v1", Status: providercredential.StatusActive,
		Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
}

var _ ProviderCredentialService = (*providerCredentialServiceStub)(nil)
