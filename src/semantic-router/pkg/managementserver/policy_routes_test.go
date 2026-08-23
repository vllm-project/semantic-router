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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policybulk"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

const (
	policyOneID   = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
	policyTwoID   = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
	bindingOneID  = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
	policyUserOne = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
	policyUserTwo = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
)

func TestPolicyRoutesPushExactPolicyScopeBeforePagination(t *testing.T) {
	now := policyTestNow()
	service := &policyServiceStub{accessPage: policymanagement.Page[policymanagement.AccessPolicy]{
		Items: []policymanagement.AccessPolicy{{
			ID: policyOneID, NamespaceID: testNamespaceID,
			Name: "One", Status: accesscontrol.PolicyStatusActive, Revision: 1, CreatedAt: now, UpdatedAt: now,
		}},
		NextCursor: "opaque-policy-cursor", HasMore: true, PageSize: 2,
	}}
	authorizationCalls := 0
	authorizer := policyAuthorizerFunc(func(context.Context, AuthorizationRequest) (AuthorizationDecision, error) {
		authorizationCalls++
		return AuthorizationDecision{}, managementauthorization.ErrDenied
	})
	routes := newTestPolicyRoutes(t, service, &policyBulkStub{}, authorizer)
	routes.scopes = resultScopeResolverFunc(func(_ context.Context, _ accesscontrol.ManagementPrincipalID,
		namespaceID accesscontrol.NamespaceID, permission accesscontrol.Permission,
	) (managementauthorization.ResultScope, error) {
		if permission != accesscontrol.PermissionAccessPolicyRead {
			t.Fatalf("permission = %q", permission)
		}
		return managementauthorization.ResultScope{
			NamespaceID: namespaceID,
			ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
				accesscontrol.ScopeResourceAccessPolicy: {policyOneID},
			},
		}, nil
	})
	request := authorizedRequest(t, http.MethodGet, accessPoliciesPath+"?pageSize=2&status=active&search=One", nil)
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), policyOneID) ||
		strings.Contains(response.Body.String(), policyTwoID) || !strings.Contains(response.Body.String(), "opaque-policy-cursor") {
		t.Fatalf("scoped policy list status=%d body=%s", response.Code, response.Body.String())
	}
	ids := service.lastAccessList.Scope.IDs(accesscontrol.ScopeResourceAccessPolicy)
	if authorizationCalls != 0 || service.lastAccessList.Scope.All || len(ids) != 1 || ids[0] != policyOneID ||
		service.lastAccessList.Search != "One" {
		t.Fatalf("list authorization calls=%d request=%#v", authorizationCalls, service.lastAccessList)
	}
}

func TestPolicyRoutesHideDeniedResourceBeforeCASAndEnforceDeleteCAS(t *testing.T) {
	now := policyTestNow()
	policy := policymanagement.AccessPolicy{
		ID: policyOneID, NamespaceID: testNamespaceID,
		Name: "One", Status: accesscontrol.PolicyStatusActive, Revision: 3, CreatedAt: now, UpdatedAt: now,
	}
	service := &policyServiceStub{accessPolicy: policy, deleteAccessResult: policymanagement.MutationResult{
		Kind: "access_policy", ID: policyOneID, Revision: 4, HTTPStatus: http.StatusNoContent,
	}}
	deniedRoutes := newTestPolicyRoutes(t, service, &policyBulkStub{}, policyAuthorizerFunc(
		func(context.Context, AuthorizationRequest) (AuthorizationDecision, error) {
			return AuthorizationDecision{}, managementauthorization.ErrDenied
		}))
	denied := authorizedRequest(t, http.MethodDelete, accessPoliciesPath+"/"+policyOneID, nil)
	denied.Header.Set(managementapi.HeaderIfMatch, `"access-policy:3"`)
	deniedResponse := httptest.NewRecorder()
	deniedRoutes.ServeHTTP(deniedResponse, denied)
	if deniedResponse.Code != http.StatusNotFound || service.deleteAccessCalls != 0 {
		t.Fatalf("denied delete status=%d calls=%d", deniedResponse.Code, service.deleteAccessCalls)
	}

	allowedRoutes := newTestPolicyRoutes(t, service, &policyBulkStub{}, &authorizerStub{})
	missing := authorizedRequest(t, http.MethodDelete, accessPoliciesPath+"/"+policyOneID, nil)
	missingResponse := httptest.NewRecorder()
	allowedRoutes.ServeHTTP(missingResponse, missing)
	if missingResponse.Code != http.StatusPreconditionRequired || service.deleteAccessCalls != 0 {
		t.Fatalf("missing CAS status=%d calls=%d", missingResponse.Code, service.deleteAccessCalls)
	}
	deleted := authorizedRequest(t, http.MethodDelete, accessPoliciesPath+"/"+policyOneID, nil)
	deleted.Header.Set(managementapi.HeaderIfMatch, `"access-policy:3"`)
	deletedResponse := httptest.NewRecorder()
	allowedRoutes.ServeHTTP(deletedResponse, deleted)
	if deletedResponse.Code != http.StatusNoContent || service.lastDeleteAccess.ExpectedRevision != 3 {
		t.Fatalf("delete status=%d request=%#v", deletedResponse.Code, service.lastDeleteAccess)
	}
}

func TestPolicyRoutesUseStrictMediaAndCanonicalISODuration(t *testing.T) {
	service := &policyServiceStub{createRateResult: policymanagement.MutationResult{
		Kind: "rate_limit_policy", ID: policyOneID, Revision: 1, HTTPStatus: http.StatusCreated,
	}}
	routes := newTestPolicyRoutes(t, service, &policyBulkStub{}, &authorizerStub{})
	body := `{"name":"Minute quota","rules":[{"metric":"requests","algorithm":"sliding_log","limit":"12","window":"PT1M","accounting":"request","enforcement":"enforce"}]}`
	request := authorizedRequest(t, http.MethodPost, ratePoliciesPath, strings.NewReader(body))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set(managementapi.HeaderIdempotencyKey, "create-rate-policy-0001")
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusCreated || len(service.lastCreateRate.Rules) != 1 ||
		service.lastCreateRate.Rules[0].Window.Duration() != time.Minute {
		t.Fatalf("rate create status=%d request=%#v body=%s", response.Code, service.lastCreateRate, response.Body.String())
	}

	invalid := authorizedRequest(t, http.MethodPost, ratePoliciesPath, strings.NewReader(body))
	invalid.Header.Set("Content-Type", "application/json")
	invalid.Header.Set(managementapi.HeaderIdempotencyKey, "create-rate-policy-0002")
	invalidResponse := httptest.NewRecorder()
	routes.ServeHTTP(invalidResponse, invalid)
	if invalidResponse.Code != http.StatusUnsupportedMediaType || service.createRateCalls != 1 {
		t.Fatalf("wrong media status=%d calls=%d", invalidResponse.Code, service.createRateCalls)
	}
}

func TestPolicyBulkRouteAuthorizesWholeRequestBeforeEnqueue(t *testing.T) {
	bulk := &policyBulkStub{accessResult: policybulk.EnqueueResult{Operation: policybulk.Operation{
		ID: bindingOneID, NamespaceID: testNamespaceID, Kind: policybulk.AccessBindingOperationKind,
		OriginPrincipalID: testPrincipalID, State: policybulk.OperationPending, Total: 2,
		TargetIDs: []string{policyOneID, policyTwoID}, CreatedAt: policyTestNow(), UpdatedAt: policyTestNow(),
	}}}
	authorizer := policyAuthorizerFunc(func(_ context.Context, request AuthorizationRequest) (AuthorizationDecision, error) {
		for _, target := range request.Targets["subject"] {
			if string(target.Scope.UserID) == policyUserTwo {
				return AuthorizationDecision{}, managementauthorization.ErrDenied
			}
		}
		return AuthorizationDecision{}, nil
	})
	routes := newTestPolicyRoutes(t, &policyServiceStub{}, bulk, authorizer)
	body := `{"items":[` +
		`{"itemId":"` + policyOneID + `","policyId":"` + policyOneID + `","subject":{"type":"user","id":"` + policyUserOne + `"}},` +
		`{"itemId":"` + policyTwoID + `","policyId":"` + policyOneID + `","subject":{"type":"user","id":"` + policyUserTwo + `"}}]}`
	request := authorizedRequest(t, http.MethodPost, accessBindingBulkPath, strings.NewReader(body))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set(managementapi.HeaderIdempotencyKey, "bulk-access-bindings-0001")
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusForbidden || bulk.accessCalls != 0 {
		t.Fatalf("denied bulk status=%d enqueue calls=%d body=%s", response.Code, bulk.accessCalls, response.Body.String())
	}

	allowed := newTestPolicyRoutes(t, &policyServiceStub{}, bulk, &authorizerStub{})
	request = authorizedRequest(t, http.MethodPost, accessBindingBulkPath, strings.NewReader(body))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set(managementapi.HeaderIdempotencyKey, "bulk-access-bindings-0002")
	response = httptest.NewRecorder()
	allowed.ServeHTTP(response, request)
	if response.Code != http.StatusAccepted || bulk.accessCalls != 1 ||
		response.Header().Get("Location") != managementapi.BasePath+"/operations/"+bindingOneID {
		t.Fatalf("accepted bulk status=%d calls=%d headers=%#v body=%s",
			response.Code, bulk.accessCalls, response.Header(), response.Body.String())
	}
}

type policyAuthorizerFunc func(context.Context, AuthorizationRequest) (AuthorizationDecision, error)

func (function policyAuthorizerFunc) Authorize(ctx context.Context, request AuthorizationRequest) (AuthorizationDecision, error) {
	return function(ctx, request)
}

type policyBulkStub struct {
	accessResult policybulk.EnqueueResult
	rateResult   policybulk.EnqueueResult
	accessCalls  int
	rateCalls    int
}

func (service *policyBulkStub) Ready(context.Context) error { return nil }
func (service *policyBulkStub) EnqueueAccessBindings(_ context.Context, _ policybulk.EnqueueAccessRequest) (policybulk.EnqueueResult, error) {
	service.accessCalls++
	return service.accessResult, nil
}

func (service *policyBulkStub) EnqueueRateBindings(_ context.Context, _ policybulk.EnqueueRateRequest) (policybulk.EnqueueResult, error) {
	service.rateCalls++
	return service.rateResult, nil
}

type policyServiceStub struct {
	accessPolicy       policymanagement.AccessPolicy
	accessPage         policymanagement.Page[policymanagement.AccessPolicy]
	lastAccessList     policymanagement.ListPoliciesRequest
	createRateResult   policymanagement.MutationResult
	deleteAccessResult policymanagement.MutationResult
	lastCreateRate     policymanagement.CreateRateLimitPolicyRequest
	lastDeleteAccess   policymanagement.DeletePolicyRequest
	createRateCalls    int
	deleteAccessCalls  int
}

func (service *policyServiceStub) Ready(context.Context) error { return nil }
func (service *policyServiceStub) GetAccessPolicy(context.Context, string, string) (policymanagement.AccessPolicy, error) {
	if service.accessPolicy.ID == "" {
		return policymanagement.AccessPolicy{}, policymanagement.ErrNotFound
	}
	return service.accessPolicy, nil
}

func (service *policyServiceStub) ListAccessPolicies(_ context.Context, request policymanagement.ListPoliciesRequest) (policymanagement.Page[policymanagement.AccessPolicy], error) {
	service.lastAccessList = request
	return service.accessPage, nil
}

func (service *policyServiceStub) CreateAccessPolicy(context.Context, policymanagement.CreateAccessPolicyRequest) (policymanagement.MutationResult, error) {
	return policymanagement.MutationResult{}, errors.New("unexpected CreateAccessPolicy")
}

func (service *policyServiceStub) UpdateAccessPolicy(context.Context, policymanagement.UpdateAccessPolicyRequest) (policymanagement.MutationResult, error) {
	return policymanagement.MutationResult{}, errors.New("unexpected UpdateAccessPolicy")
}

func (service *policyServiceStub) DeleteAccessPolicy(_ context.Context, request policymanagement.DeletePolicyRequest) (policymanagement.MutationResult, error) {
	service.deleteAccessCalls++
	service.lastDeleteAccess = request
	return service.deleteAccessResult, nil
}

func (service *policyServiceStub) GetRateLimitPolicy(context.Context, string, string) (policymanagement.RateLimitPolicy, error) {
	return policymanagement.RateLimitPolicy{}, policymanagement.ErrNotFound
}

func (service *policyServiceStub) ListRateLimitPolicies(context.Context, policymanagement.ListPoliciesRequest) (policymanagement.Page[policymanagement.RateLimitPolicy], error) {
	return policymanagement.Page[policymanagement.RateLimitPolicy]{}, nil
}

func (service *policyServiceStub) CreateRateLimitPolicy(_ context.Context, request policymanagement.CreateRateLimitPolicyRequest) (policymanagement.MutationResult, error) {
	service.createRateCalls++
	service.lastCreateRate = request
	return service.createRateResult, nil
}

func (service *policyServiceStub) UpdateRateLimitPolicy(context.Context, policymanagement.UpdateRateLimitPolicyRequest) (policymanagement.MutationResult, error) {
	return policymanagement.MutationResult{}, errors.New("unexpected UpdateRateLimitPolicy")
}

func (service *policyServiceStub) DeleteRateLimitPolicy(context.Context, policymanagement.DeletePolicyRequest) (policymanagement.MutationResult, error) {
	return policymanagement.MutationResult{}, errors.New("unexpected DeleteRateLimitPolicy")
}

func (service *policyServiceStub) GetAccessBinding(context.Context, string, string) (policymanagement.AccessPolicyBinding, error) {
	return policymanagement.AccessPolicyBinding{}, policymanagement.ErrNotFound
}

func (service *policyServiceStub) ListAccessBindings(context.Context, policymanagement.ListBindingsRequest) (policymanagement.Page[policymanagement.AccessPolicyBinding], error) {
	return policymanagement.Page[policymanagement.AccessPolicyBinding]{}, nil
}

func (service *policyServiceStub) CreateAccessBinding(context.Context, policymanagement.CreateAccessBindingRequest) (policymanagement.MutationResult, error) {
	return policymanagement.MutationResult{}, errors.New("unexpected CreateAccessBinding")
}

func (service *policyServiceStub) UpdateAccessBinding(context.Context, policymanagement.UpdateBindingRequest) (policymanagement.MutationResult, error) {
	return policymanagement.MutationResult{}, errors.New("unexpected UpdateAccessBinding")
}

func (service *policyServiceStub) DeleteAccessBinding(context.Context, policymanagement.DeleteBindingRequest) (policymanagement.MutationResult, error) {
	return policymanagement.MutationResult{}, errors.New("unexpected DeleteAccessBinding")
}

func (service *policyServiceStub) GetRateBinding(context.Context, string, string) (policymanagement.RateLimitBinding, error) {
	return policymanagement.RateLimitBinding{}, policymanagement.ErrNotFound
}

func (service *policyServiceStub) ListRateBindings(context.Context, policymanagement.ListBindingsRequest) (policymanagement.Page[policymanagement.RateLimitBinding], error) {
	return policymanagement.Page[policymanagement.RateLimitBinding]{}, nil
}

func (service *policyServiceStub) CreateRateBinding(context.Context, policymanagement.CreateRateBindingRequest) (policymanagement.MutationResult, error) {
	return policymanagement.MutationResult{}, errors.New("unexpected CreateRateBinding")
}

func (service *policyServiceStub) CreateInlineRateBinding(context.Context, policymanagement.CreateInlineRateBindingRequest) (policymanagement.InlineRateBindingResult, error) {
	return policymanagement.InlineRateBindingResult{}, errors.New("unexpected CreateInlineRateBinding")
}

func (service *policyServiceStub) UpdateRateBinding(context.Context, policymanagement.UpdateBindingRequest) (policymanagement.MutationResult, error) {
	return policymanagement.MutationResult{}, errors.New("unexpected UpdateRateBinding")
}

func (service *policyServiceStub) DeleteRateBinding(context.Context, policymanagement.DeleteBindingRequest) (policymanagement.MutationResult, error) {
	return policymanagement.MutationResult{}, errors.New("unexpected DeleteRateBinding")
}

func newTestPolicyRoutes(t *testing.T, service PolicyManagementService, bulk PolicyBulkService,
	authorizer Authorizer,
) *PolicyRoutes {
	t.Helper()
	routes, err := NewPolicyRoutes(PolicyRoutesOptions{
		Service: service, Bulk: bulk,
		Namespaces: NamespaceResolverFunc(func(context.Context, *http.Request) (string, error) {
			return testNamespaceID, nil
		}),
		Sessions: sessionStub{}, Authorization: authorizer, Scopes: allowAllResultScopes(), Now: policyTestNow,
	})
	if err != nil {
		t.Fatal(err)
	}
	return routes
}

func policyTestNow() time.Time { return time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC) }

var (
	_ PolicyManagementService = (*policyServiceStub)(nil)
	_ PolicyBulkService       = (*policyBulkStub)(nil)
)
