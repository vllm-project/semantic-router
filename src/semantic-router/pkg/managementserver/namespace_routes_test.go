package managementserver

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/namespacemanagement"
)

const (
	namespaceRouteID        = "a1111111-1111-4111-8111-111111111111"
	namespaceRouteOther     = "a2222222-2222-4222-8222-222222222222"
	namespaceRoutePrincipal = "a3333333-3333-4333-8333-333333333333"
)

type namespaceRouteServiceStub struct {
	NamespaceManagementService
	page             namespacemanagement.Page[namespacemanagement.Namespace]
	createResult     namespacemanagement.MutationResult
	selfService      namespacemanagement.SelfServicePolicy
	selfPatchResult  namespacemanagement.MutationResult
	lastList         namespacemanagement.ListRequest
	lastCreate       namespacemanagement.CreateNamespaceRequest
	lastSelfPatch    namespacemanagement.PatchSelfServicePolicyRequest
	getNamespaceCall int
}

func (service *namespaceRouteServiceStub) ListNamespaces(
	_ context.Context,
	request namespacemanagement.ListRequest,
) (namespacemanagement.Page[namespacemanagement.Namespace], error) {
	service.lastList = request
	return service.page, nil
}

func (service *namespaceRouteServiceStub) CreateNamespace(
	_ context.Context,
	request namespacemanagement.CreateNamespaceRequest,
) (namespacemanagement.MutationResult, error) {
	service.lastCreate = request
	return service.createResult, nil
}

func (service *namespaceRouteServiceStub) GetNamespace(
	context.Context,
	string,
) (namespacemanagement.Namespace, error) {
	service.getNamespaceCall++
	return namespacemanagement.Namespace{}, namespacemanagement.ErrNotFound
}

func (service *namespaceRouteServiceStub) GetSelfServicePolicy(
	context.Context,
	string,
) (namespacemanagement.SelfServicePolicy, error) {
	return service.selfService, nil
}

func (service *namespaceRouteServiceStub) PatchSelfServicePolicy(
	_ context.Context,
	request namespacemanagement.PatchSelfServicePolicyRequest,
) (namespacemanagement.MutationResult, error) {
	service.lastSelfPatch = request
	return service.selfPatchResult, nil
}

type namespaceScopeResolverFunc func(context.Context, string) (namespacemanagement.ResultScope, error)

func (function namespaceScopeResolverFunc) ResolveNamespaceResultScope(
	ctx context.Context,
	principalID string,
) (namespacemanagement.ResultScope, error) {
	return function(ctx, principalID)
}

type namespaceRouteAuthorizerFunc func(context.Context, AuthorizationRequest) (AuthorizationDecision, error)

func (function namespaceRouteAuthorizerFunc) Authorize(
	ctx context.Context,
	request AuthorizationRequest,
) (AuthorizationDecision, error) {
	return function(ctx, request)
}

type namespaceSessionStub struct{}

func (namespaceSessionStub) Authenticate(
	_ context.Context,
	_ string,
	namespaceID string,
	_ time.Time,
) (managementauth.AuthenticatedSession, error) {
	return managementauth.AuthenticatedSession{
		NamespaceID: namespaceID,
		Session: managementauth.LiveSession{Session: managementauth.Session{
			PrincipalID: namespaceRoutePrincipal,
		}},
	}, nil
}

func TestNamespaceListPushesPrincipalScopeBeforePagination(t *testing.T) {
	now := time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC)
	service := &namespaceRouteServiceStub{
		page: namespacemanagement.Page[namespacemanagement.Namespace]{
			Items: []namespacemanagement.Namespace{{
				ID:               namespaceRouteID,
				Name:             "Scoped",
				QuotaPartitionID: namespaceRouteID,
				BillingCurrency:  "USD",
				Status:           accesscontrol.NamespaceStatusActive,
				Revision:         1,
				RuntimeEpoch:     1,
				CreatedAt:        now,
				UpdatedAt:        now,
			}},
			PageSize: 2,
		},
	}
	authorizationCalls := 0
	routes := newNamespaceTestRoutes(
		t,
		service,
		namespaceRouteAuthorizerFunc(func(context.Context, AuthorizationRequest) (AuthorizationDecision, error) {
			authorizationCalls++
			return AuthorizationDecision{}, errors.New("list must use compiled result scope")
		}),
		namespaceScopeResolverFunc(func(_ context.Context, principalID string) (namespacemanagement.ResultScope, error) {
			if principalID != namespaceRoutePrincipal {
				t.Fatalf("scope principal = %q", principalID)
			}
			return namespacemanagement.ResultScope{NamespaceIDs: []string{namespaceRouteID}}, nil
		}),
	)

	response := serveNamespaceRequest(routes, authorizedRequest(
		t,
		http.MethodGet,
		namespacesPath+"?pageSize=2&status=active",
		nil,
	))
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), namespaceRouteID) {
		t.Fatalf("Namespace list status=%d body=%s", response.Code, response.Body.String())
	}
	if authorizationCalls != 0 || service.lastList.Scope.All ||
		len(service.lastList.Scope.NamespaceIDs) != 1 ||
		service.lastList.Scope.NamespaceIDs[0] != namespaceRouteID ||
		service.lastList.PageSize != 2 || service.lastList.Status != "active" {
		t.Fatalf("Namespace list request = %#v, authorization calls = %d", service.lastList, authorizationCalls)
	}
}

func TestNamespaceDetailDenialIsNondisclosingAndDoesNotRead(t *testing.T) {
	service := &namespaceRouteServiceStub{}
	routes := newNamespaceTestRoutes(
		t,
		service,
		namespaceRouteAuthorizerFunc(func(context.Context, AuthorizationRequest) (AuthorizationDecision, error) {
			return AuthorizationDecision{}, managementauthorization.ErrDenied
		}),
		namespaceScopeResolverFunc(func(context.Context, string) (namespacemanagement.ResultScope, error) {
			return namespacemanagement.ResultScope{All: true}, nil
		}),
	)

	response := serveNamespaceRequest(routes, authorizedRequest(
		t,
		http.MethodGet,
		namespacesPath+"/"+namespaceRouteOther,
		nil,
	))
	if response.Code != http.StatusNotFound || service.getNamespaceCall != 0 ||
		strings.Contains(response.Body.String(), "forbidden") {
		t.Fatalf("denied detail status=%d calls=%d body=%s", response.Code, service.getNamespaceCall, response.Body.String())
	}
}

func TestNamespaceCreateUsesClusterSessionAndCanonicalReceipt(t *testing.T) {
	service := &namespaceRouteServiceStub{
		createResult: namespacemanagement.MutationResult{
			Kind:       "namespace",
			ID:         namespaceRouteID,
			Revision:   1,
			HTTPStatus: http.StatusCreated,
		},
	}
	var authorized AuthorizationRequest
	routes := newNamespaceTestRoutes(
		t,
		service,
		namespaceRouteAuthorizerFunc(func(_ context.Context, request AuthorizationRequest) (AuthorizationDecision, error) {
			authorized = request
			return AuthorizationDecision{}, nil
		}),
		namespaceScopeResolverFunc(func(context.Context, string) (namespacemanagement.ResultScope, error) {
			return namespacemanagement.ResultScope{All: true}, nil
		}),
	)
	request := authorizedRequest(
		t,
		http.MethodPost,
		namespacesPath,
		strings.NewReader(`{"name":"Default","billingCurrency":"USD","reason":"Initial tenant"}`),
	)
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set(managementapi.HeaderIdempotencyKey, "namespace-create-route-0001")

	response := serveNamespaceRequest(routes, request)
	if response.Code != http.StatusCreated ||
		response.Header().Get("Location") != namespacesPath+"/"+namespaceRouteID ||
		response.Header().Get(managementapi.HeaderETag) != `"namespace:1"` {
		t.Fatalf("Namespace create status=%d headers=%#v body=%s", response.Code, response.Header(), response.Body.String())
	}
	if authorized.NamespaceID != "" || authorized.Operation.Scope != managementapi.ScopeCluster ||
		service.lastCreate.IdempotencyKey != "namespace-create-route-0001" ||
		service.lastCreate.Actor.Reason != "Initial tenant" {
		t.Fatalf("Namespace create authorization=%#v request=%#v", authorized, service.lastCreate)
	}
}

func TestSelfServicePatchRequiresCASBeforeMutation(t *testing.T) {
	service := &namespaceRouteServiceStub{
		selfService: namespacemanagement.SelfServicePolicy{
			NamespaceID:         namespaceRouteID,
			DelegatedSessionTTL: 15 * time.Minute,
			Revision:            2,
		},
		selfPatchResult: namespacemanagement.MutationResult{
			Kind:       "self_service_policy",
			ID:         namespaceRouteID,
			Revision:   3,
			HTTPStatus: http.StatusOK,
		},
	}
	routes := newNamespaceTestRoutes(
		t,
		service,
		namespaceRouteAuthorizerFunc(func(context.Context, AuthorizationRequest) (AuthorizationDecision, error) {
			return AuthorizationDecision{}, nil
		}),
		namespaceScopeResolverFunc(func(context.Context, string) (namespacemanagement.ResultScope, error) {
			return namespacemanagement.ResultScope{All: true}, nil
		}),
	)

	withoutCAS := authorizedRequest(
		t,
		http.MethodPatch,
		namespacesPath+"/"+namespaceRouteID+"/self-service-policy",
		strings.NewReader(`{"maxKeysPerUser":1,"reason":"Enable one key"}`),
	)
	withoutCAS.Header.Set("Content-Type", managementapi.JSONMediaType)
	response := serveNamespaceRequest(routes, withoutCAS)
	if response.Code != http.StatusPreconditionRequired {
		t.Fatalf("self-service PATCH without CAS status=%d body=%s", response.Code, response.Body.String())
	}

	withCAS := authorizedRequest(
		t,
		http.MethodPatch,
		namespacesPath+"/"+namespaceRouteID+"/self-service-policy",
		strings.NewReader(`{"maxKeysPerUser":1,"reason":"Enable one key"}`),
	)
	withCAS.Header.Set("Content-Type", managementapi.JSONMediaType)
	withCAS.Header.Set(managementapi.HeaderIfMatch, `"self-service-policy:2"`)
	response = serveNamespaceRequest(routes, withCAS)
	if response.Code != http.StatusOK ||
		service.lastSelfPatch.ExpectedRevision != 2 ||
		service.lastSelfPatch.MaxKeysPerUser == nil ||
		*service.lastSelfPatch.MaxKeysPerUser != 1 {
		t.Fatalf("self-service PATCH status=%d request=%#v body=%s", response.Code, service.lastSelfPatch, response.Body.String())
	}
}

func newNamespaceTestRoutes(
	t *testing.T,
	service NamespaceManagementService,
	authorizer Authorizer,
	scopes NamespaceResultScopeResolver,
) *NamespaceRoutes {
	t.Helper()
	routes, err := NewNamespaceRoutes(NamespaceRoutesOptions{
		Service:       service,
		Sessions:      namespaceSessionStub{},
		Authorization: authorizer,
		Scopes:        scopes,
		Now: func() time.Time {
			return time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC)
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return routes
}

func serveNamespaceRequest(routes *NamespaceRoutes, request *http.Request) *httptest.ResponseRecorder {
	mux := http.NewServeMux()
	routes.Register(mux)
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, request)
	return response
}
