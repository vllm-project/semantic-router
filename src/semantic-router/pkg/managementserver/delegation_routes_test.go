package managementserver

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/delegationmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

const testDelegatedSessionID = "99999999-9999-4999-8999-999999999999"

func TestDelegationCreateUsesNamespaceKeyAndIdempotencyContracts(t *testing.T) {
	canonical := `{"resourceId":"` + testDelegatedSessionID + `","kind":"delegated_inference_credential","secret":"vsd_secret"}`
	service := &delegationServiceStub{
		key:    testManagementAPIKey(testAPIKeyID, testAPIKeyOwnerID, time.Now().UTC()),
		secret: delegationmanagement.SecretResult{Session: delegationmanagement.Session{ID: testDelegatedSessionID}, CanonicalJSON: []byte(canonical)},
	}
	routes := newTestDelegationRoutes(t, service)
	request := authorizedRequest(t, http.MethodPost, selfInferenceSessionsPath,
		strings.NewReader(`{"keyId":"`+testAPIKeyID+`"}`))
	request.Header.Set(managementapi.HeaderNamespaceID, testNamespaceID)
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set(managementapi.HeaderIdempotencyKey, "delegation-create-012345")
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusCreated || response.Body.String() != canonical+"\n" {
		t.Fatalf("create status=%d body=%s", response.Code, response.Body.String())
	}
	if service.lastCreate.NamespaceID != testNamespaceID || service.lastCreate.KeyID != testAPIKeyID ||
		service.lastCreate.IdempotencyKey != "delegation-create-012345" ||
		service.lastCreate.Actor.ManagementSessionID != testCredentialID {
		t.Fatalf("delegation create request = %#v", service.lastCreate)
	}
	if response.Header().Get("Location") != selfInferenceSessionsPath+"/"+testDelegatedSessionID ||
		response.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("delegation create headers = %#v", response.Header())
	}
}

func TestDelegationCreateRejectsUnknownBodyAndMissingIdempotency(t *testing.T) {
	for _, test := range []struct {
		name        string
		body        string
		idempotency string
	}{
		{name: "unknown field", body: `{"keyId":"` + testAPIKeyID + `","audience":"other"}`, idempotency: "delegation-create-012345"},
		{name: "missing idempotency", body: `{"keyId":"` + testAPIKeyID + `"}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			service := &delegationServiceStub{}
			routes := newTestDelegationRoutes(t, service)
			request := authorizedRequest(t, http.MethodPost, selfInferenceSessionsPath, strings.NewReader(test.body))
			request.Header.Set(managementapi.HeaderNamespaceID, testNamespaceID)
			request.Header.Set("Content-Type", managementapi.JSONMediaType)
			if test.idempotency != "" {
				request.Header.Set(managementapi.HeaderIdempotencyKey, test.idempotency)
			}
			response := httptest.NewRecorder()
			routes.ServeHTTP(response, request)
			if response.Code != http.StatusBadRequest || service.createCalls != 0 {
				t.Fatalf("status=%d creates=%d body=%s", response.Code, service.createCalls, response.Body.String())
			}
		})
	}
}

func TestSelfInferenceKeyListForwardsBoundedSearch(t *testing.T) {
	service := &delegationServiceStub{eligible: delegationmanagement.EligibleKey{
		KeyID: testAPIKeyID, Name: "Developer key", OwnerKind: accesscontrol.SubjectKindUser,
		OwnerID: testAPIKeyOwnerID,
	}}
	routes := newTestDelegationRoutes(t, service)
	request := authorizedRequest(t, http.MethodGet,
		selfInferenceKeysPath+"?search=Developer&pageSize=7", nil)
	request.Header.Set(managementapi.HeaderNamespaceID, testNamespaceID)
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusOK || service.lastList.Search != "Developer" ||
		service.lastList.PageSize != 7 || !strings.Contains(response.Body.String(), testAPIKeyID) {
		t.Fatalf("status=%d request=%#v body=%s", response.Code, service.lastList, response.Body.String())
	}
}

func TestSelfInferenceKeyDetailUsesScopedEligibility(t *testing.T) {
	service := &delegationServiceStub{eligible: delegationmanagement.EligibleKey{
		KeyID: testAPIKeyID, Name: "Developer key", OwnerKind: accesscontrol.SubjectKindUser,
		OwnerID: testAPIKeyOwnerID,
	}}
	routes := newTestDelegationRoutes(t, service)
	request := authorizedRequest(t, http.MethodGet, selfInferenceKeysPath+"/"+testAPIKeyID, nil)
	request.SetPathValue("keyId", testAPIKeyID)
	request.Header.Set(managementapi.HeaderNamespaceID, testNamespaceID)
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusOK || service.lastGet.KeyID != testAPIKeyID ||
		service.lastGet.PrincipalID != testPrincipalID ||
		!strings.Contains(response.Body.String(), `"data"`) {
		t.Fatalf("status=%d request=%#v body=%s", response.Code, service.lastGet, response.Body.String())
	}

	service.eligibleErr = delegationmanagement.ErrNotEligible
	notOwned := authorizedRequest(t, http.MethodGet, selfInferenceKeysPath+"/"+testAPIKeyID, nil)
	notOwned.SetPathValue("keyId", testAPIKeyID)
	notOwned.Header.Set(managementapi.HeaderNamespaceID, testNamespaceID)
	notOwnedResponse := httptest.NewRecorder()
	routes.ServeHTTP(notOwnedResponse, notOwned)
	if notOwnedResponse.Code != http.StatusNotFound {
		t.Fatalf("non-eligible key status=%d body=%s", notOwnedResponse.Code, notOwnedResponse.Body.String())
	}
}

type delegationServiceStub struct {
	key         accesscontrol.APIKey
	eligible    delegationmanagement.EligibleKey
	eligibleErr error
	lastList    delegationmanagement.ListRequest
	lastGet     delegationmanagement.EligibleKeyRequest
	secret      delegationmanagement.SecretResult
	lastCreate  delegationmanagement.CreateRequest
	createCalls int
}

func (*delegationServiceStub) Ready(context.Context) error { return nil }
func (*delegationServiceStub) ResolveSelf(context.Context, string, string, string) (delegationmanagement.SelfContext, error) {
	return delegationmanagement.SelfContext{UserID: testAPIKeyOwnerID}, nil
}

func (service *delegationServiceStub) GetKey(context.Context, string, string) (accesscontrol.APIKey, error) {
	return service.key, nil
}

func (*delegationServiceStub) GetSession(context.Context, string, string) (delegationmanagement.Session, error) {
	return delegationmanagement.Session{}, delegationmanagement.ErrNotFound
}

func (service *delegationServiceStub) ListEligibleKeys(_ context.Context, request delegationmanagement.ListRequest) (delegationmanagement.ResultPage[delegationmanagement.EligibleKey], error) {
	service.lastList = request
	return delegationmanagement.ResultPage[delegationmanagement.EligibleKey]{Items: []delegationmanagement.EligibleKey{service.eligible}}, service.eligibleErr
}

func (service *delegationServiceStub) GetEligibleKey(_ context.Context, request delegationmanagement.EligibleKeyRequest) (delegationmanagement.EligibleKey, error) {
	service.lastGet = request
	return service.eligible, service.eligibleErr
}

func (*delegationServiceStub) ListSessions(context.Context, delegationmanagement.ListRequest) (delegationmanagement.ResultPage[delegationmanagement.Session], error) {
	return delegationmanagement.ResultPage[delegationmanagement.Session]{}, nil
}

func (service *delegationServiceStub) Create(_ context.Context, request delegationmanagement.CreateRequest) (delegationmanagement.SecretResult, error) {
	service.createCalls++
	service.lastCreate = request
	return service.secret, nil
}

func (*delegationServiceStub) Revoke(context.Context, delegationmanagement.RevokeRequest) (delegationmanagement.MutationResult, error) {
	return delegationmanagement.MutationResult{}, nil
}

func (*delegationServiceStub) RevokeAll(context.Context, delegationmanagement.RevokeAllRequest) (delegationmanagement.RevokeAllResult, error) {
	return delegationmanagement.RevokeAllResult{}, nil
}

type delegationSessionStub struct{}

func (delegationSessionStub) Authenticate(
	context.Context,
	string,
	string,
	time.Time,
) (managementauth.AuthenticatedSession, error) {
	return managementauth.AuthenticatedSession{NamespaceID: testNamespaceID, Session: managementauth.LiveSession{
		Session: managementauth.Session{ID: testCredentialID, PrincipalID: testPrincipalID},
	}}, nil
}

func newTestDelegationRoutes(t *testing.T, service DelegationManagementService) *DelegationRoutes {
	t.Helper()
	routes, err := NewDelegationRoutes(DelegationRoutesOptions{
		Service: service, Namespaces: ExplicitNamespaceResolver{}, Sessions: delegationSessionStub{},
		Authorization: &authorizerStub{}, Now: func() time.Time { return time.Date(2026, 8, 23, 8, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatal(err)
	}
	return routes
}

var _ DelegationManagementService = (*delegationServiceStub)(nil)
