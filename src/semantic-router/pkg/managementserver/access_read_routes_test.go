package managementserver

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

type accessReadServiceStub struct {
	inspection  accessmanagement.AuthorizationContext
	checkResult accessmanagement.AccessCheckResult
	context     accessmanagement.RoutingContext
	check       accessmanagement.AccessCheckRequest
	update      accessmanagement.UpdateRoutingContextRequest
	inspectCall int
	checkCall   int
}

func (stub *accessReadServiceStub) Ready(context.Context) error { return nil }

func (stub *accessReadServiceStub) Inspect(context.Context, string, accessmanagement.Subject) (accessmanagement.AuthorizationContext, error) {
	stub.inspectCall++
	return stub.inspection, nil
}

func (stub *accessReadServiceStub) GetEffectivePolicy(context.Context, string, accessmanagement.Subject) (accessmanagement.EffectivePolicy, error) {
	return accessmanagement.EffectivePolicy{}, nil
}

func (stub *accessReadServiceStub) GetQuota(context.Context, string, accessmanagement.Subject) (accessmanagement.EffectiveQuota, error) {
	return accessmanagement.EffectiveQuota{}, nil
}

func (stub *accessReadServiceStub) GetRoutingContext(context.Context, string, accessmanagement.Subject) (accessmanagement.RoutingContext, error) {
	return stub.context, nil
}

func (stub *accessReadServiceStub) UpdateRoutingContext(_ context.Context, request accessmanagement.UpdateRoutingContextRequest) (accessmanagement.RoutingContext, error) {
	stub.update = request
	return stub.context, nil
}

func (stub *accessReadServiceStub) Check(_ context.Context, request accessmanagement.AccessCheckRequest) (accessmanagement.AccessCheckResult, error) {
	stub.checkCall++
	stub.check = request
	return stub.checkResult, nil
}

func TestAccessCheckRejectsRawCredentialFields(t *testing.T) {
	service := &accessReadServiceStub{}
	routes := newTestAccessReadRoutes(t, service, AuthorizerFunc(func(context.Context, AuthorizationRequest) (AuthorizationDecision, error) {
		return AuthorizationDecision{}, nil
	}))
	request := authorizedRequest(t, http.MethodPost, managementapi.BasePath+"/access:check", strings.NewReader(
		`{"subject":{"type":"api_key","id":"44444444-4444-4444-8444-444444444444"},`+
			`"resource":{"type":"model","id":"model-1"},"permission":"invoke","credential":"secret"}`,
	))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest || service.inspectCall != 0 || service.checkCall != 0 {
		t.Fatalf("status=%d inspect=%d check=%d body=%s", response.Code, service.inspectCall, service.checkCall, response.Body.String())
	}
}

func TestAccessCheckForwardsOnlyStoredIdentityAndMarksOverrideSimulation(t *testing.T) {
	subject := accessmanagement.Subject{Kind: accesscontrol.SubjectKindAPIKey, ID: "44444444-4444-4444-8444-444444444444"}
	service := &accessReadServiceStub{
		inspection: accessmanagement.AuthorizationContext{Subject: subject},
		checkResult: accessmanagement.AccessCheckResult{
			Subject: subject, Resource: accesscontrol.GrantResource{Type: accesscontrol.GrantResourceModel, ID: "model-1"},
			Permission: accesscontrol.GrantPermissionInvoke, Decision: accesscontrol.AccessDecisionAllow,
			Simulation: true, DesiredRevision: 8, AppliedRevision: 8,
			RoutingContext: []accessmanagement.EffectiveClaim{{StoredClaim: accessmanagement.StoredClaim{
				Name: "priority", Value: routingsnapshot.ClaimValue{Kind: "integer", Integer: 7},
			}, Source: subject}},
		},
	}
	var authorized AuthorizationRequest
	routes := newTestAccessReadRoutes(t, service, AuthorizerFunc(func(_ context.Context, request AuthorizationRequest) (AuthorizationDecision, error) {
		authorized = request
		return AuthorizationDecision{}, nil
	}))
	request := authorizedRequest(t, http.MethodPost, managementapi.BasePath+"/access:check", strings.NewReader(
		`{"subject":{"type":"api_key","id":"44444444-4444-4444-8444-444444444444"},`+
			`"resource":{"type":"model","id":"model-1"},"permission":"invoke",`+
			`"routingContextOverride":{"priority":{"kind":"integer","integer":7}}}`,
	))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"simulation":true`) {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	if !authorized.Conditions["routing_context_override_requested"] || service.checkCall != 1 ||
		!service.check.OverridePresent || service.check.Override["priority"].Integer != 7 {
		t.Fatalf("authorization=%#v check=%#v", authorized, service.check)
	}
}

func TestRoutingContextPutForwardsCASAndReturnsFreshETag(t *testing.T) {
	subject := accessmanagement.Subject{Kind: accesscontrol.SubjectKindAPIKey, ID: "44444444-4444-4444-8444-444444444444"}
	service := &accessReadServiceStub{
		inspection: accessmanagement.AuthorizationContext{Subject: subject},
		context: accessmanagement.RoutingContext{
			Subject: subject, Revision: 12, SchemaRevision: 2,
			Stored: []accessmanagement.StoredClaim{{
				Name: "segment", Value: routingsnapshot.ClaimValue{Kind: "string", String: "research"},
				Revision: 12, UpdatedAt: time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC),
			}},
		},
	}
	routes := newTestAccessReadRoutes(t, service, AuthorizerFunc(func(context.Context, AuthorizationRequest) (AuthorizationDecision, error) {
		return AuthorizationDecision{}, nil
	}))
	mux := http.NewServeMux()
	routes.Register(mux)
	request := authorizedRequest(t, http.MethodPut,
		managementapi.BasePath+"/api-keys/44444444-4444-4444-8444-444444444444/routing-context",
		strings.NewReader(`{"values":{"segment":{"kind":"string","string":"research"}}}`))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set(managementapi.HeaderIfMatch, `"routing-context:11"`)
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, request)
	if response.Code != http.StatusOK || response.Header().Get(managementapi.HeaderETag) != `"routing-context:12"` {
		t.Fatalf("status=%d etag=%q body=%s", response.Code, response.Header().Get(managementapi.HeaderETag), response.Body.String())
	}
	if service.update.ExpectedRevision != 11 || service.update.Values["segment"].String != "research" || service.update.Actor.PrincipalID == "" {
		t.Fatalf("update request=%#v", service.update)
	}
}

func newTestAccessReadRoutes(t *testing.T, service AccessReadService, authorizer Authorizer) *AccessReadRoutes {
	t.Helper()
	routes, err := NewAccessReadRoutes(AccessReadRoutesOptions{
		Service: service,
		Namespaces: NamespaceResolverFunc(func(context.Context, *http.Request) (string, error) {
			return testNamespaceID, nil
		}),
		Sessions: sessionStub{}, Authorization: authorizer,
		Now: func() time.Time { return time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatal(err)
	}
	return routes
}
