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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotareconciliation"
)

const (
	unknownFenceID   = "f1111111-1111-4111-8111-111111111111"
	unknownBindingID = "f2222222-2222-4222-8222-222222222222"
	unknownRuleID    = "f3333333-3333-4333-8333-333333333333"
	unknownPolicyID  = "f4444444-4444-4444-8444-444444444444"
	unknownUserID    = "f5555555-5555-4555-8555-555555555555"
	unknownOpID      = "f6666666-6666-4666-8666-666666666666"
)

func TestUnknownUsageListPushesAuthorizedScopeBeforePagination(t *testing.T) {
	service := &unknownUsageServiceStub{page: quotareconciliation.Page{
		Items: []quotareconciliation.Fence{unknownUsageTestFence()}, PageSize: 25,
	}}
	authorizer := &authorizerStub{err: errors.New("list must not authorize rows after pagination")}
	routes := newTestUnknownUsageRoutes(t, service, authorizer)
	routes.scopes = resultScopeResolverFunc(func(_ context.Context, _ accesscontrol.ManagementPrincipalID,
		namespaceID accesscontrol.NamespaceID, permission accesscontrol.Permission,
	) (managementauthorization.ResultScope, error) {
		if permission != accesscontrol.PermissionQuotaRead {
			t.Fatalf("list permission = %q", permission)
		}
		return managementauthorization.ResultScope{
			NamespaceID: namespaceID,
			ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
				accesscontrol.ScopeResourceRateLimitBinding: {unknownBindingID},
			},
		}, nil
	})
	request := authorizedRequest(t, http.MethodGet, unknownUsageFencesPath+"?pageSize=25&state=open", nil)
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusOK || authorizer.calls != 0 || service.list.Scope.All {
		t.Fatalf("list status=%d authorization=%d request=%#v body=%s",
			response.Code, authorizer.calls, service.list, response.Body.String())
	}
	ids := service.list.Scope.IDs(accesscontrol.ScopeResourceRateLimitBinding)
	if len(ids) != 1 || ids[0] != unknownBindingID || service.list.State != quotareconciliation.FenceOpen {
		t.Fatalf("list scope = %#v", service.list)
	}
}

func TestUnknownUsageDetailIsNondisclosingAndFieldAuthorized(t *testing.T) {
	service := &unknownUsageServiceStub{fence: unknownUsageTestFence()}
	denied := newTestUnknownUsageRoutes(t, service, &authorizerStub{err: managementauthorization.ErrDenied})
	request := authorizedRequest(t, http.MethodGet, unknownUsageFencesPath+"/"+unknownFenceID, nil)
	response := httptest.NewRecorder()
	denied.ServeHTTP(response, request)
	if response.Code != http.StatusNotFound || strings.Contains(response.Body.String(), "admission-one") {
		t.Fatalf("denied detail status=%d body=%s", response.Code, response.Body.String())
	}

	authorizer := &authorizerStub{}
	allowed := newTestUnknownUsageRoutes(t, service, authorizer)
	request = authorizedRequest(t, http.MethodGet, unknownUsageFencesPath+"/"+unknownFenceID+
		"?includeInternalDimensions=true&includeEvidence=true&includeActor=true", nil)
	response = httptest.NewRecorder()
	allowed.ServeHTTP(response, request)
	body := response.Body.String()
	if response.Code != http.StatusOK || response.Header().Get(managementapi.HeaderETag) != `"unknown-usage-fence:3"` ||
		!strings.Contains(body, "provider-model-one") || !strings.Contains(body, strings.Repeat("a", 64)) ||
		!authorizer.last.Conditions["internal_usage_dimensions_requested"] ||
		!authorizer.last.Conditions["fence_payload_evidence_requested"] ||
		!authorizer.last.Conditions["fence_actor_or_audit_fields_requested"] {
		t.Fatalf("allowed detail status=%d authorization=%#v body=%s", response.Code, authorizer.last, body)
	}
	if len(authorizer.last.Targets["all_affected_bindings"]) != 1 {
		t.Fatalf("detail targets = %#v", authorizer.last.Targets)
	}
}

func TestUnknownUsageReconcileRequiresCASAndReturnsDurableOperation(t *testing.T) {
	service := &unknownUsageServiceStub{fence: unknownUsageTestFence(), enqueue: quotareconciliation.EnqueueResult{
		Operation: quotareconciliation.Operation{
			ID: unknownOpID, NamespaceID: testNamespaceID,
			FenceID: unknownFenceID, Kind: quotareconciliation.OperationKind,
			State: quotareconciliation.OperationPending, Total: 1,
			CreatedAt: policyTestNow(), UpdatedAt: policyTestNow(),
		},
	}}
	routes := newTestUnknownUsageRoutes(t, service, &authorizerStub{})
	body := `{"strategy":"conservative_debit","evidenceReferences":["usage-ledger:event-one"],"reason":"Use the admission bound."}`
	missingCAS := authorizedRequest(t, http.MethodPost, unknownUsageFencesPath+"/"+unknownFenceID+":reconcile", strings.NewReader(body))
	missingCAS.Header.Set("Content-Type", managementapi.JSONMediaType)
	missingCAS.Header.Set(managementapi.HeaderIdempotencyKey, "unknown-reconcile-0001")
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, missingCAS)
	if response.Code != http.StatusPreconditionRequired || service.reconcileCalls != 0 {
		t.Fatalf("missing CAS status=%d calls=%d", response.Code, service.reconcileCalls)
	}

	request := authorizedRequest(t, http.MethodPost, unknownUsageFencesPath+"/"+unknownFenceID+":reconcile", strings.NewReader(body))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set(managementapi.HeaderIdempotencyKey, "unknown-reconcile-0001")
	request.Header.Set(managementapi.HeaderIfMatch, `"unknown-usage-fence:3"`)
	response = httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusAccepted || service.reconcileCalls != 1 ||
		service.reconcile.ExpectedRevision != 3 || service.reconcile.Strategy != quotareconciliation.StrategyConservativeDebit ||
		response.Header().Get("Location") != managementapi.BasePath+"/operations/"+unknownOpID {
		t.Fatalf("reconcile status=%d request=%#v headers=%#v body=%s",
			response.Code, service.reconcile, response.Header(), response.Body.String())
	}
}

func newTestUnknownUsageRoutes(t *testing.T, service UnknownUsageService, authorizer Authorizer) *UnknownUsageRoutes {
	t.Helper()
	routes, err := NewUnknownUsageRoutes(UnknownUsageRoutesOptions{
		Service: service,
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

func unknownUsageTestFence() quotareconciliation.Fence {
	now := policyTestNow()
	return quotareconciliation.Fence{
		ID: unknownFenceID, NamespaceID: testNamespaceID, AdmissionID: "admission-one",
		State: quotareconciliation.FenceOpen, Revision: 3, Reason: "provider_usage_unavailable",
		Bindings: []quotareconciliation.Binding{{
			BindingID: unknownBindingID, RuleID: unknownRuleID, PolicyID: unknownPolicyID,
			Subject: quotareconciliation.Subject{Kind: accesscontrol.SubjectKindUser, ID: unknownUserID},
			Metric:  quota.MetricTotalTokens, Algorithm: quota.AlgorithmSlidingLog,
			Enforcement: quota.EnforcementEnforce, AdmissionLimit: "100", MaximumDebit: "80", Window: time.Minute,
		}},
		KnownCharge: quotareconciliation.Charge{InputTokens: "4", OutputTokens: "6", TotalTokens: "10", Costs: []quotareconciliation.Cost{}},
		Unknown: []quotareconciliation.UnknownDispatch{{
			DispatchID:     "dispatch-one",
			EvidenceDigest: strings.Repeat("a", 64), Reason: "provider_usage_unavailable",
			ModelID: "model_one", BackendID: "f7777777-7777-4777-8777-777777777777",
			ProviderID: "provider-one", ProviderModelID: "provider-model-one", PricingRevision: 2,
		}},
		CreatedAt: now, UpdatedAt: now,
	}
}

type unknownUsageServiceStub struct {
	fence              quotareconciliation.Fence
	operation          quotareconciliation.Operation
	operationErr       error
	operationNamespace string
	operationID        string
	page               quotareconciliation.Page
	list               quotareconciliation.ListRequest
	reconcile          quotareconciliation.ReconcileRequest
	enqueue            quotareconciliation.EnqueueResult
	reconcileCalls     int
}

func (*unknownUsageServiceStub) Ready(context.Context) error { return nil }
func (service *unknownUsageServiceStub) Get(context.Context, string, string) (quotareconciliation.Fence, error) {
	if service.fence.ID == "" {
		return quotareconciliation.Fence{}, quotareconciliation.ErrNotFound
	}
	return service.fence, nil
}

func (service *unknownUsageServiceStub) GetOperation(_ context.Context, namespaceID, operationID string) (quotareconciliation.Operation, error) {
	service.operationNamespace = namespaceID
	service.operationID = operationID
	if service.operationErr != nil {
		return quotareconciliation.Operation{}, service.operationErr
	}
	if service.operation.ID == "" {
		return quotareconciliation.Operation{}, quotareconciliation.ErrNotFound
	}
	return service.operation, nil
}

func (service *unknownUsageServiceStub) List(_ context.Context, request quotareconciliation.ListRequest) (quotareconciliation.Page, error) {
	service.list = request
	return service.page, nil
}

func (service *unknownUsageServiceStub) Reconcile(_ context.Context, request quotareconciliation.ReconcileRequest) (quotareconciliation.EnqueueResult, error) {
	service.reconcileCalls++
	service.reconcile = request
	return service.enqueue, nil
}
func (*unknownUsageServiceStub) Run(context.Context) error { return nil }
func (*unknownUsageServiceStub) Close()                    {}

var _ UnknownUsageService = (*unknownUsageServiceStub)(nil)
