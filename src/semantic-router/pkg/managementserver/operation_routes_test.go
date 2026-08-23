package managementserver

import (
	"context"
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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotareconciliation"
)

const (
	operationOneID       = "a1111111-1111-4111-8111-111111111111"
	operationTwoID       = "a2222222-2222-4222-8222-222222222222"
	operationItemOneID   = "a3333333-3333-4333-8333-333333333333"
	operationItemTwoID   = "a4444444-4444-4444-8444-444444444444"
	operationPolicyOneID = "a5555555-5555-4555-8555-555555555555"
	operationPolicyTwoID = "a6666666-6666-4666-8666-666666666666"
	operationUserOneID   = "a7777777-7777-4777-8777-777777777777"
	operationUserTwoID   = "a8888888-8888-4888-8888-888888888888"
	operationOtherActor  = "a9999999-9999-4999-8999-999999999999"
)

func TestOperationRoutesPushCompleteVisibilityBeforePagination(t *testing.T) {
	first := testPolicyBulkOperation(operationOneID, operationItemOneID, operationPolicyOneID, operationUserOneID, testPrincipalID)
	service := &operationServiceStub{page: policybulk.Page{
		Items: []policybulk.Operation{first}, NextCursor: "opaque-operation-cursor",
		HasMore: true, PageSize: 2,
	}}
	authorizer := &operationAuthorizerRecorder{}
	routes := newTestOperationRoutes(t, service, authorizer)
	routes.scopes = resultScopeResolverFunc(func(_ context.Context, _ accesscontrol.ManagementPrincipalID,
		namespaceID accesscontrol.NamespaceID, permission accesscontrol.Permission,
	) (managementauthorization.ResultScope, error) {
		scope := managementauthorization.ResultScope{NamespaceID: namespaceID}
		switch permission {
		case accesscontrol.PermissionAccessPolicyRead:
			scope.ResourceIDs = map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
				accesscontrol.ScopeResourceAccessPolicy: {operationPolicyOneID},
			}
		case accesscontrol.PermissionRatePolicyRead:
		case accesscontrol.PermissionOperationRead:
			scope.UserIDs = []accesscontrol.UserID{operationUserOneID}
			scope.ResourceIDs = map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
				accesscontrol.ScopeResourceAccessPolicy: {operationPolicyOneID},
			}
		default:
			t.Fatalf("permission = %q", permission)
		}
		return scope, nil
	})
	request := authorizedRequest(t, http.MethodGet, operationsPath+"?pageSize=2&kind="+
		policybulk.AccessBindingOperationKind+"&state=pending&originPrincipalId="+testPrincipalID, nil)
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), operationOneID) ||
		strings.Contains(response.Body.String(), operationTwoID) ||
		!strings.Contains(response.Body.String(), "opaque-operation-cursor") {
		t.Fatalf("operation list status=%d body=%s", response.Code, response.Body.String())
	}
	if service.lastList.NamespaceID != testNamespaceID || service.lastList.PageSize != 2 ||
		service.lastList.Kind != policybulk.AccessBindingOperationKind ||
		service.lastList.State != policybulk.OperationPending || service.lastList.OriginPrincipalID != testPrincipalID {
		t.Fatalf("operation list request = %#v", service.lastList)
	}
	accessIDs := service.lastList.Visibility.Access.IDs(accesscontrol.ScopeResourceAccessPolicy)
	operationIDs := service.lastList.Visibility.Operation.IDs(accesscontrol.ScopeResourceAccessPolicy)
	if len(authorizer.requests) != 0 || len(accessIDs) != 1 || accessIDs[0] != operationPolicyOneID ||
		len(operationIDs) != 1 || operationIDs[0] != operationPolicyOneID ||
		len(service.lastList.Visibility.Operation.UserIDs) != 1 ||
		service.lastList.Visibility.Operation.UserIDs[0] != operationUserOneID ||
		!service.lastList.Visibility.Rate.Empty() {
		t.Fatalf("operation visibility = %#v, authorization calls=%d", service.lastList.Visibility, len(authorizer.requests))
	}
}

func TestOperationRoutesHideDeniedDetailWithoutETag(t *testing.T) {
	operation := testPolicyBulkOperation(operationOneID, operationItemOneID, operationPolicyOneID, operationUserOneID, testPrincipalID)
	service := &operationServiceStub{operation: operation}
	routes := newTestOperationRoutes(t, service, &operationAuthorizerRecorder{authorize: func(AuthorizationRequest) error {
		return managementauthorization.ErrDenied
	}})
	request := authorizedRequest(t, http.MethodGet, operationsPath+"/"+operationOneID, nil)
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusNotFound || response.Header().Get(managementapi.HeaderETag) != "" {
		t.Fatalf("denied operation detail status=%d headers=%#v body=%s", response.Code, response.Header(), response.Body.String())
	}
}

func TestOperationRoutesResolveRegisteredUnknownUsageOperationWithinNamespace(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	unknown := &unknownUsageServiceStub{fence: unknownUsageTestFence(), operation: quotareconciliation.Operation{
		ID: unknownOpID, NamespaceID: testNamespaceID, FenceID: unknownFenceID,
		Kind: quotareconciliation.OperationKind, OriginPrincipalID: testPrincipalID,
		ActorChain: []string{testPrincipalID}, Version: 2, State: quotareconciliation.OperationPending,
		Total: 1, CreatedAt: now, UpdatedAt: now,
	}}
	authorizer := &operationAuthorizerRecorder{}
	reader, err := NewUnknownUsageOperationDetailReader(unknown, authorizer)
	if err != nil {
		t.Fatal(err)
	}
	policy := &operationServiceStub{getErr: policybulk.ErrNotFound}
	routes := newTestOperationRoutesWithReaders(t, policy, authorizer, []OperationDetailReader{reader})
	request := authorizedRequest(t, http.MethodGet, operationsPath+"/"+unknownOpID, nil)
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusOK || response.Header().Get(managementapi.HeaderETag) != `"operation:2"` ||
		!strings.Contains(response.Body.String(), `"kind":"unknown_usage_fence.reconcile"`) ||
		unknown.operationNamespace != testNamespaceID || unknown.operationID != unknownOpID {
		t.Fatalf("unknown operation status=%d headers=%#v scope=%s/%s body=%s",
			response.Code, response.Header(), unknown.operationNamespace, unknown.operationID, response.Body.String())
	}
	if authorizer.requestForPath(unknownUsageFencesPath+"/{fenceId}").Operation.Path == "" ||
		authorizer.requestForPath(operationsPath+"/{operationId}").Operation.Path == "" {
		t.Fatalf("unknown Operation did not recheck domain and generic authorization: %#v", authorizer.requests)
	}
}

func TestOperationRoutesHideDeniedRegisteredKindAndUnknownID(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	unknown := &unknownUsageServiceStub{fence: unknownUsageTestFence(), operation: quotareconciliation.Operation{
		ID: unknownOpID, NamespaceID: testNamespaceID, FenceID: unknownFenceID,
		Kind: quotareconciliation.OperationKind, OriginPrincipalID: operationOtherActor,
		ActorChain: []string{operationOtherActor}, Version: 1, State: quotareconciliation.OperationPending,
		Total: 1, CreatedAt: now, UpdatedAt: now,
	}}
	authorizer := &operationAuthorizerRecorder{authorize: func(request AuthorizationRequest) error {
		if request.Operation.Path == operationsPath+"/{operationId}" {
			return managementauthorization.ErrDenied
		}
		return nil
	}}
	reader, err := NewUnknownUsageOperationDetailReader(unknown, authorizer)
	if err != nil {
		t.Fatal(err)
	}
	policy := &operationServiceStub{getErr: policybulk.ErrNotFound}
	routes := newTestOperationRoutesWithReaders(t, policy, authorizer, []OperationDetailReader{reader})
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, authorizedRequest(t, http.MethodGet, operationsPath+"/"+unknownOpID, nil))
	if response.Code != http.StatusNotFound || response.Header().Get(managementapi.HeaderETag) != "" ||
		strings.Contains(response.Body.String(), quotareconciliation.OperationKind) {
		t.Fatalf("denied registered Operation status=%d headers=%#v body=%s",
			response.Code, response.Header(), response.Body.String())
	}

	unknown.operation = quotareconciliation.Operation{}
	unknown.operationErr = quotareconciliation.ErrNotFound
	unknownID := "b1111111-1111-4111-8111-111111111111"
	response = httptest.NewRecorder()
	routes.ServeHTTP(response, authorizedRequest(t, http.MethodGet, operationsPath+"/"+unknownID, nil))
	if response.Code != http.StatusNotFound || response.Header().Get(managementapi.HeaderETag) != "" {
		t.Fatalf("unknown Operation status=%d headers=%#v body=%s",
			response.Code, response.Header(), response.Body.String())
	}
}

func TestOperationCancelAuthorizesBeforeCASAndUsesDurableIdempotency(t *testing.T) {
	operation := testPolicyBulkOperation(operationOneID, operationItemOneID, operationPolicyOneID, operationUserOneID, testPrincipalID)
	operation.Version = 3
	service := &operationServiceStub{operation: operation}
	deniedRoutes := newTestOperationRoutes(t, service, &operationAuthorizerRecorder{authorize: func(AuthorizationRequest) error {
		return managementauthorization.ErrDenied
	}})
	denied := authorizedRequest(t, http.MethodPost, operationsPath+"/"+operationOneID+":cancel", nil)
	deniedResponse := httptest.NewRecorder()
	deniedRoutes.ServeHTTP(deniedResponse, denied)
	if deniedResponse.Code != http.StatusNotFound || service.cancelCalls != 0 {
		t.Fatalf("denied cancel status=%d calls=%d body=%s", deniedResponse.Code, service.cancelCalls, deniedResponse.Body.String())
	}

	authorizer := &operationAuthorizerRecorder{}
	routes := newTestOperationRoutes(t, service, authorizer)
	missing := authorizedRequest(t, http.MethodPost, operationsPath+"/"+operationOneID+":cancel", nil)
	missingResponse := httptest.NewRecorder()
	routes.ServeHTTP(missingResponse, missing)
	if missingResponse.Code != http.StatusPreconditionRequired || service.cancelCalls != 0 {
		t.Fatalf("missing cancel CAS status=%d calls=%d", missingResponse.Code, service.cancelCalls)
	}

	stale := authorizedRequest(t, http.MethodPost, operationsPath+"/"+operationOneID+":cancel", nil)
	stale.Header.Set(managementapi.HeaderIfMatch, `"operation:2"`)
	stale.Header.Set(managementapi.HeaderIdempotencyKey, "cancel-operation-stale-01")
	service.cancelErr = policybulk.ErrRevisionConflict
	staleResponse := httptest.NewRecorder()
	routes.ServeHTTP(staleResponse, stale)
	if staleResponse.Code != http.StatusPreconditionFailed || service.cancelCalls != 1 {
		t.Fatalf("stale cancel status=%d calls=%d body=%s", staleResponse.Code, service.cancelCalls, staleResponse.Body.String())
	}

	cancelled := operation
	cancelled.Version = 4
	cancelled.State = policybulk.OperationCancelled
	cancelled.Completed = 1
	completedAt := cancelled.UpdatedAt.Add(time.Second)
	cancelled.UpdatedAt, cancelled.CompletedAt = completedAt, &completedAt
	service.cancelErr = nil
	service.cancelResult = policybulk.CancelResult{Operation: cancelled, Replayed: true}
	success := authorizedRequest(t, http.MethodPost, operationsPath+"/"+operationOneID+":cancel", nil)
	success.Header.Set(managementapi.HeaderIfMatch, `"operation:3"`)
	success.Header.Set(managementapi.HeaderIdempotencyKey, "cancel-operation-success-01")
	successResponse := httptest.NewRecorder()
	routes.ServeHTTP(successResponse, success)
	if successResponse.Code != http.StatusOK || successResponse.Header().Get(managementapi.HeaderETag) != `"operation:4"` ||
		successResponse.Header().Get(managementapi.HeaderIdempotencyReplayed) != "true" ||
		!strings.Contains(successResponse.Body.String(), `"state":"cancelled"`) {
		t.Fatalf("successful cancel status=%d headers=%#v body=%s", successResponse.Code, successResponse.Header(), successResponse.Body.String())
	}
	if service.lastCancel.ExpectedVersion != 3 || service.lastCancel.IdempotencyKey != "cancel-operation-success-01" ||
		service.lastCancel.Actor.PrincipalID != testPrincipalID {
		t.Fatalf("cancel request = %#v", service.lastCancel)
	}
	mutation := authorizer.requestForPath(accessBindingBulkPath)
	generic := authorizer.requestForPath(operationsPath + "/{operationId}:cancel")
	if mutation.Operation.Path == "" || generic.Operation.Path == "" ||
		!generic.Recorded["original_domain_mutation"] || !generic.Conditions["operation_originator"] {
		t.Fatalf("cancel authorization mutation=%#v generic=%#v", mutation, generic)
	}
}

func TestOperationRoutesRejectUnboundedOrBodyBearingRequests(t *testing.T) {
	service := &operationServiceStub{}
	routes := newTestOperationRoutes(t, service, &operationAuthorizerRecorder{})
	for _, request := range []*http.Request{
		authorizedRequest(t, http.MethodGet, operationsPath+"?offset=1", nil),
		authorizedRequest(t, http.MethodGet, operationsPath+"?pageSize=201", nil),
		authorizedRequest(t, http.MethodGet, operationsPath, strings.NewReader(`{}`)),
	} {
		response := httptest.NewRecorder()
		routes.ServeHTTP(response, request)
		if response.Code != http.StatusBadRequest {
			t.Errorf("strict operation request %s status=%d body=%s", request.URL.String(), response.Code, response.Body.String())
		}
	}
	if service.listCalls != 0 {
		t.Fatalf("invalid lists reached service %d times", service.listCalls)
	}
}

type operationServiceStub struct {
	operation    policybulk.Operation
	page         policybulk.Page
	cancelResult policybulk.CancelResult
	getErr       error
	listErr      error
	cancelErr    error
	lastList     policybulk.ListRequest
	lastCancel   policybulk.CancelRequest
	listCalls    int
	cancelCalls  int
}

func (service *operationServiceStub) Ready(context.Context) error { return nil }

func (service *operationServiceStub) Get(context.Context, string, string) (policybulk.Operation, error) {
	return service.operation, service.getErr
}

func (service *operationServiceStub) List(_ context.Context, request policybulk.ListRequest) (policybulk.Page, error) {
	service.listCalls++
	service.lastList = request
	return service.page, service.listErr
}

func (service *operationServiceStub) Cancel(_ context.Context, request policybulk.CancelRequest) (policybulk.CancelResult, error) {
	service.cancelCalls++
	service.lastCancel = request
	return service.cancelResult, service.cancelErr
}

type operationAuthorizerRecorder struct {
	authorize func(AuthorizationRequest) error
	requests  []AuthorizationRequest
}

func (authorizer *operationAuthorizerRecorder) Authorize(_ context.Context, request AuthorizationRequest) (AuthorizationDecision, error) {
	authorizer.requests = append(authorizer.requests, request)
	if authorizer.authorize != nil {
		if err := authorizer.authorize(request); err != nil {
			return AuthorizationDecision{}, err
		}
	}
	return AuthorizationDecision{}, nil
}

func (authorizer *operationAuthorizerRecorder) requestForPath(path string) AuthorizationRequest {
	for index := len(authorizer.requests) - 1; index >= 0; index-- {
		if authorizer.requests[index].Operation.Path == path {
			return authorizer.requests[index]
		}
	}
	return AuthorizationRequest{}
}

func newTestOperationRoutes(t *testing.T, service OperationService, authorizer Authorizer) *OperationRoutes {
	return newTestOperationRoutesWithReaders(t, service, authorizer, nil)
}

func newTestOperationRoutesWithReaders(t *testing.T, service OperationService, authorizer Authorizer,
	readers []OperationDetailReader,
) *OperationRoutes {
	t.Helper()
	routes, err := NewOperationRoutes(OperationRoutesOptions{
		Service: service, DetailReaders: readers,
		Namespaces: NamespaceResolverFunc(func(context.Context, *http.Request) (string, error) {
			return testNamespaceID, nil
		}),
		Sessions: sessionStub{}, Authorization: authorizer, Scopes: allowAllResultScopes(),
		Now: func() time.Time { return time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatal(err)
	}
	return routes
}

func testPolicyBulkOperation(operationID, itemID, policyID, userID, originID string) policybulk.Operation {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	return policybulk.Operation{
		ID: operationID, NamespaceID: testNamespaceID, Kind: policybulk.AccessBindingOperationKind,
		OriginPrincipalID: originID, ActorChain: []string{originID}, Version: 1,
		State: policybulk.OperationPending, Total: 1, TargetIDs: []string{itemID},
		Targets: []policybulk.OperationTarget{{
			ItemID: itemID, Kind: policybulk.ItemKindAccessBinding,
			PolicyID: policyID, Subject: policymanagement.Subject{Type: accesscontrol.SubjectKindUser, ID: userID},
		}},
		CreatedAt: now, UpdatedAt: now,
	}
}
