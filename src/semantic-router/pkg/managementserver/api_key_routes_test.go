package managementserver

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apikeymanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
)

const (
	testAPIKeyID       = "44444444-4444-4444-8444-444444444444"
	testAPIKeyOwnerID  = "55555555-5555-4555-8555-555555555555"
	testAPIKeySecondID = "66666666-6666-4666-8666-666666666666"
	testAccessPolicyID = "77777777-7777-4777-8777-777777777777"
	testRatePolicyID   = "88888888-8888-4888-8888-888888888888"
)

func TestAPIKeyRoutesDeliverCanonicalSecretAndAuthorizeOwner(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	key := testManagementAPIKey(testAPIKeyID, testAPIKeyOwnerID, now)
	body := `{"data":{"keyId":"` + testAPIKeyID + `","name":"user@example.test","owner":{"type":"user","id":"` + testAPIKeyOwnerID + `"},"status":"active","revision":1,"createdAt":"2026-08-22T12:00:00Z","updatedAt":"2026-08-22T12:00:00Z"},"credential":{"credentialId":"` + testCredentialID + `","keyId":"` + testAPIKeyID + `","kid":"credential-kid","status":"active","revealable":true,"notBefore":"2026-08-22T12:00:00Z","createdAt":"2026-08-22T12:00:00Z"},"secret":"vsr_credential-kid_secret","deliveryExpiresAt":"2026-08-22T12:05:00Z"}`
	service := &apiKeyServiceStub{get: key, create: apikeymanagement.SecretMutationResult{
		Key: key, ResponseRevision: 1, CanonicalJSON: []byte(body), Replayed: true,
	}}
	authorizer := &authorizerStub{decision: AuthorizationDecision{AuthorityDigest: testAuthority}}
	routes := newTestAPIKeyRoutes(t, service, authorizer)
	request := authorizedRequest(t, http.MethodPost, apiKeysPath, strings.NewReader(
		`{"name":"user@example.test","owner":{"type":"user","id":"`+testAPIKeyOwnerID+`"},"revealable":true}`,
	))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set(managementapi.HeaderIdempotencyKey, "create-api-key-0123456789")
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusCreated || response.Body.String() != body+"\n" {
		t.Fatalf("create status=%d body=%s", response.Code, response.Body.String())
	}
	if response.Header().Get(managementapi.HeaderETag) != `"key:1"` ||
		response.Header().Get(managementapi.HeaderIdempotencyReplayed) != "true" ||
		response.Header().Get("Location") != apiKeysPath+"/"+testAPIKeyID {
		t.Fatalf("create headers=%#v", response.Header())
	}
	ownerTargets := authorizer.last.Targets["owner"]
	if len(ownerTargets) != 1 || ownerTargets[0].Scope.Kind != accesscontrol.ScopeKindUser ||
		string(ownerTargets[0].Scope.UserID) != testAPIKeyOwnerID {
		t.Fatalf("owner authorization targets=%#v", authorizer.last.Targets)
	}
	if service.lastCreate.Owner.Kind != accesscontrol.SubjectKindUser ||
		service.lastCreate.IdempotencyKey != "create-api-key-0123456789" {
		t.Fatalf("create request=%#v", service.lastCreate)
	}
}

func TestAPIKeyRoutesPushExactKeyScopeBeforePagination(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	totalCount := uint64(71)
	first := testManagementAPIKey(testAPIKeyID, testAPIKeyOwnerID, now)
	service := &apiKeyServiceStub{list: apikeymanagement.KeyPage{
		Items: []accesscontrol.APIKey{first}, NextCursor: "opaque-next", HasMore: true,
		PageSize: 2, TotalCount: &totalCount,
	}}
	authorizationCalls := 0
	authorizer := apiKeyAuthorizerFunc(func(context.Context, AuthorizationRequest) (AuthorizationDecision, error) {
		authorizationCalls++
		return AuthorizationDecision{}, managementauthorization.ErrDenied
	})
	routes := newTestAPIKeyRoutes(t, service, authorizer)
	routes.scopes = resultScopeResolverFunc(func(_ context.Context, _ accesscontrol.ManagementPrincipalID,
		namespaceID accesscontrol.NamespaceID, permission accesscontrol.Permission,
	) (managementauthorization.ResultScope, error) {
		if permission != accesscontrol.PermissionKeyRead {
			t.Fatalf("permission = %q", permission)
		}
		return managementauthorization.ResultScope{
			NamespaceID: namespaceID,
			APIKeyIDs:   []accesscontrol.APIKeyID{testAPIKeyID},
		}, nil
	})
	request := authorizedRequest(t, http.MethodGet,
		apiKeysPath+"?pageSize=2&search=user%40example.test&includeTotal=true", nil)
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), testAPIKeyID) ||
		strings.Contains(response.Body.String(), testAPIKeySecondID) || !strings.Contains(response.Body.String(), "opaque-next") ||
		!strings.Contains(response.Body.String(), `"totalCount":"71"`) {
		t.Fatalf("scoped list status=%d body=%s", response.Code, response.Body.String())
	}
	if authorizationCalls != 0 || service.lastList.Scope.All ||
		len(service.lastList.Scope.APIKeyIDs) != 1 || service.lastList.Scope.APIKeyIDs[0] != testAPIKeyID ||
		service.lastList.Search != "user@example.test" || !service.lastList.IncludeTotal {
		t.Fatalf("list authorization calls=%d request=%#v", authorizationCalls, service.lastList)
	}
}

func TestAPIKeyCreateAuthorizesAndForwardsExplicitPolicyBindings(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	key := testManagementAPIKey(testAPIKeyID, testAPIKeyOwnerID, now)
	service := &apiKeyServiceStub{create: apikeymanagement.SecretMutationResult{
		Key: key, ResponseRevision: 1, CanonicalJSON: []byte(`{"data":{}}`),
	}}
	authorizer := &authorizerStub{}
	routes := newTestAPIKeyRoutes(t, service, authorizer)
	request := authorizedRequest(t, http.MethodPost, apiKeysPath, strings.NewReader(
		`{"name":"user@example.test","owner":{"type":"user","id":"`+testAPIKeyOwnerID+`"},`+
			`"accessPolicyIds":["`+testAccessPolicyID+`"],`+
			`"rateLimitOverride":{"policyId":"`+testRatePolicyID+`"}}`,
	))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set(managementapi.HeaderIdempotencyKey, "create-explicit-policy-key-01")
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusCreated {
		t.Fatalf("create status=%d body=%s", response.Code, response.Body.String())
	}
	conditions := authorizer.last.Conditions
	if !conditions["access_policy_binding_requested"] || !conditions["rate_policy_binding_requested"] ||
		conditions["inline_rate_policy_requested"] {
		t.Fatalf("policy conditions = %#v", conditions)
	}
	accessTargets := authorizer.last.Targets["access_policy"]
	rateTargets := authorizer.last.Targets["rate_policy"]
	if len(accessTargets) != 1 || accessTargets[0].Scope.ResourceType != accesscontrol.ScopeResourceAccessPolicy ||
		string(accessTargets[0].Scope.ResourceID) != testAccessPolicyID || len(rateTargets) != 1 ||
		rateTargets[0].Scope.ResourceType != accesscontrol.ScopeResourceRateLimitPolicy ||
		string(rateTargets[0].Scope.ResourceID) != testRatePolicyID {
		t.Fatalf("policy authorization targets = %#v", authorizer.last.Targets)
	}
	if len(service.lastCreate.AccessPolicyIDs) != 1 || service.lastCreate.AccessPolicyIDs[0] != testAccessPolicyID ||
		service.lastCreate.RateLimitOverride == nil || service.lastCreate.RateLimitOverride.PolicyID != testRatePolicyID {
		t.Fatalf("policy create request = %#v", service.lastCreate)
	}
}

func TestAPIKeyActionsHidePreconditionStateFromOutOfScopeCaller(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	service := &apiKeyServiceStub{get: testManagementAPIKey(testAPIKeyID, testAPIKeyOwnerID, now)}
	routes := newTestAPIKeyRoutes(t, service, apiKeyAuthorizerFunc(func(context.Context, AuthorizationRequest) (AuthorizationDecision, error) {
		return AuthorizationDecision{}, managementauthorization.ErrDenied
	}))
	request := authorizedRequest(t, http.MethodPost, apiKeysPath+"/"+testAPIKeyID+":disable", strings.NewReader(`{}`))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusNotFound {
		t.Fatalf("denied action status=%d body=%s", response.Code, response.Body.String())
	}
	if service.mutations != 0 {
		t.Fatalf("mutation executed after denied action: %d", service.mutations)
	}
}

func TestAPIKeyRoutesEnforceCASIdempotencyAndStrictBodies(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		name        string
		method      string
		target      string
		body        string
		contentType string
		idempotency string
		ifMatch     string
		want        int
	}{
		{"wrong media", http.MethodPost, apiKeysPath, `{}`, "application/json", "create-api-key-0123456789", "", http.StatusUnsupportedMediaType},
		{"unknown field", http.MethodPost, apiKeysPath, `{"name":"x","owner":{"type":"user","id":"` + testAPIKeyOwnerID + `"},"rawSecret":true}`, managementapi.JSONMediaType, "create-api-key-0123456789", "", http.StatusBadRequest},
		{"rename missing cas", http.MethodPatch, apiKeysPath + "/" + testAPIKeyID, `{"name":"next"}`, managementapi.JSONMediaType, "", "", http.StatusPreconditionRequired},
		{"action missing idempotency", http.MethodPost, apiKeysPath + "/" + testAPIKeyID + ":disable", `{}`, managementapi.JSONMediaType, "", `"key:1"`, http.StatusBadRequest},
		{"rotate bad cas", http.MethodPost, apiKeysPath + "/" + testAPIKeyID + "/credentials:rotate", `{"overlapSeconds":0}`, managementapi.JSONMediaType, "rotate-api-key-012345", `"1"`, http.StatusBadRequest},
		{"unknown action", http.MethodPost, apiKeysPath + "/" + testAPIKeyID + ":pause", `{}`, managementapi.JSONMediaType, "action-api-key-012345", `"key:1"`, http.StatusNotFound},
		{"offset forbidden", http.MethodGet, apiKeysPath + "?offset=1", "", "", "", "", http.StatusBadRequest},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			service := &apiKeyServiceStub{get: testManagementAPIKey(testAPIKeyID, testAPIKeyOwnerID, now)}
			routes := newTestAPIKeyRoutes(t, service, &authorizerStub{})
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
			if response.Code != test.want {
				t.Fatalf("status=%d want=%d body=%s", response.Code, test.want, response.Body.String())
			}
			if service.mutations != 0 {
				t.Fatalf("mutation executed after invalid request: %d", service.mutations)
			}
		})
	}
}

type apiKeyAuthorizerFunc func(context.Context, AuthorizationRequest) (AuthorizationDecision, error)

func (function apiKeyAuthorizerFunc) Authorize(ctx context.Context, request AuthorizationRequest) (AuthorizationDecision, error) {
	return function(ctx, request)
}

type apiKeyServiceStub struct {
	get        accesscontrol.APIKey
	list       apikeymanagement.KeyPage
	create     apikeymanagement.SecretMutationResult
	lastCreate apikeymanagement.CreateRequest
	lastList   apikeymanagement.ListKeysRequest
	mutations  int
}

func (service *apiKeyServiceStub) Ready(context.Context) error { return nil }
func (service *apiKeyServiceStub) Get(context.Context, string, string) (accesscontrol.APIKey, error) {
	return service.get, nil
}

func (service *apiKeyServiceStub) List(_ context.Context, request apikeymanagement.ListKeysRequest) (apikeymanagement.KeyPage, error) {
	service.lastList = request
	return service.list, nil
}

func (service *apiKeyServiceStub) Create(_ context.Context, request apikeymanagement.CreateRequest) (apikeymanagement.SecretMutationResult, error) {
	service.mutations++
	service.lastCreate = request
	return service.create, nil
}

func (service *apiKeyServiceStub) Rename(_ context.Context, request apikeymanagement.RenameRequest) (apikeymanagement.MutationResult, error) {
	service.mutations++
	key := service.get
	key.Name, key.Revision = request.Name, accesscontrol.Revision(request.ExpectedRevision+1)
	return apikeymanagement.MutationResult{Key: key, HTTPStatus: http.StatusOK}, nil
}

func (service *apiKeyServiceStub) Enable(context.Context, apikeymanagement.LifecycleRequest) (apikeymanagement.MutationResult, error) {
	service.mutations++
	return apikeymanagement.MutationResult{Key: service.get, HTTPStatus: http.StatusOK}, nil
}

func (service *apiKeyServiceStub) Disable(context.Context, apikeymanagement.LifecycleRequest) (apikeymanagement.MutationResult, error) {
	service.mutations++
	return apikeymanagement.MutationResult{Key: service.get, HTTPStatus: http.StatusOK}, nil
}

func (service *apiKeyServiceStub) Renew(context.Context, apikeymanagement.RenewRequest) (apikeymanagement.MutationResult, error) {
	service.mutations++
	return apikeymanagement.MutationResult{Key: service.get, HTTPStatus: http.StatusOK}, nil
}

func (service *apiKeyServiceStub) Reassign(context.Context, apikeymanagement.ReassignRequest) (apikeymanagement.MutationResult, error) {
	service.mutations++
	return apikeymanagement.MutationResult{Key: service.get, HTTPStatus: http.StatusOK}, nil
}

func (service *apiKeyServiceStub) Delete(context.Context, apikeymanagement.LifecycleRequest) (apikeymanagement.MutationResult, error) {
	service.mutations++
	return apikeymanagement.MutationResult{Key: service.get, HTTPStatus: http.StatusNoContent}, nil
}

func (service *apiKeyServiceStub) ListCredentials(context.Context, apikeymanagement.ListCredentialsRequest) (apikeymanagement.CredentialPage, error) {
	return apikeymanagement.CredentialPage{}, nil
}

func (service *apiKeyServiceStub) Rotate(context.Context, apikeymanagement.RotateRequest) (apikeymanagement.SecretMutationResult, error) {
	service.mutations++
	return service.create, nil
}

func (service *apiKeyServiceStub) RevokeCredential(context.Context, apikeymanagement.RevokeCredentialRequest) (apikeymanagement.MutationResult, error) {
	service.mutations++
	return apikeymanagement.MutationResult{Key: service.get, HTTPStatus: http.StatusNoContent}, nil
}

func (service *apiKeyServiceStub) Reveal(context.Context, apikeymanagement.RevealRequest) (string, error) {
	service.mutations++
	return "vsr_credential-kid_secret", nil
}

func newTestAPIKeyRoutes(t *testing.T, service APIKeyManagementService, authorizer Authorizer) *APIKeyRoutes {
	t.Helper()
	routes, err := NewAPIKeyRoutes(APIKeyRoutesOptions{
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
	return routes
}

func testManagementAPIKey(keyID, ownerID string, now time.Time) accesscontrol.APIKey {
	return accesscontrol.APIKey{
		NamespaceID: testNamespaceID, ID: accesscontrol.APIKeyID(keyID), Name: "user@example.test",
		Owner: accesscontrol.SubjectRef{
			NamespaceID: testNamespaceID,
			ID:          accesscontrol.SubjectID(ownerID), Kind: accesscontrol.SubjectKindUser,
		},
		Status: accesscontrol.APIKeyStatusActive, PolicyEpoch: 1, DelegationEpoch: 1,
		Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
}

var _ APIKeyManagementService = (*apiKeyServiceStub)(nil)
