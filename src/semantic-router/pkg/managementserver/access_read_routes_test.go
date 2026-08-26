package managementserver

import (
	"context"
	"encoding/json"
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
	catalog     accessmanagement.RoutingCatalog
	policy      accessmanagement.EffectivePolicy
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
	return stub.policy, nil
}

func (stub *accessReadServiceStub) GetQuota(context.Context, string, accessmanagement.Subject) (accessmanagement.EffectiveQuota, error) {
	return accessmanagement.EffectiveQuota{}, nil
}

func (stub *accessReadServiceStub) GetRoutingCatalog(context.Context, string, accessmanagement.Subject) (accessmanagement.RoutingCatalog, error) {
	return stub.catalog, nil
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

func TestKeyScopedRoutingCatalogUsesKeyAuthorizationWithoutGlobalRoutingPermission(t *testing.T) {
	subject := accessmanagement.Subject{
		Kind: accesscontrol.SubjectKindAPIKey,
		ID:   "44444444-4444-4444-8444-444444444444",
	}
	service := &accessReadServiceStub{
		inspection: accessmanagement.AuthorizationContext{
			Subject: subject,
			Ancestors: []accessmanagement.Subject{{
				Kind: accesscontrol.SubjectKindUser,
				ID:   "55555555-5555-4555-8555-555555555555",
			}},
		},
		catalog: accessmanagement.RoutingCatalog{
			Subject: subject, PolicyRevision: 7, PolicyDigest: strings.Repeat("d", 64),
			RoutingRevision: 9, RoutingDigest: strings.Repeat("e", 64),
			Models: []accessmanagement.RoutingCatalogModel{{ID: "model-visible", Revision: 1, Name: "Visible"}},
		},
	}
	var authorization AuthorizationRequest
	routes := newTestAccessReadRoutes(t, service, AuthorizerFunc(func(_ context.Context, request AuthorizationRequest) (AuthorizationDecision, error) {
		authorization = request
		return AuthorizationDecision{}, nil
	}))
	mux := http.NewServeMux()
	routes.Register(mux)
	request := authorizedRequest(t, http.MethodGet,
		managementapi.BasePath+"/api-keys/44444444-4444-4444-8444-444444444444/routing-catalog", nil)
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"model-visible"`) {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	if authorization.Operation.Permission.Canonical() !=
		"(key.read@key AND access_policy.read@key AND routing_context.read@key)" ||
		len(authorization.Targets["key"]) != 1 ||
		authorization.Targets["key"][0].Scope.ResourceID != accesscontrol.ResourceID(subject.ID) {
		t.Fatalf("routing catalog authorization = %#v", authorization)
	}
}

func TestKeyScopedRoutingCatalogEncodesRequiredEmptyCollectionsAsArrays(t *testing.T) {
	emptyWire, err := json.Marshal(routingCatalogDTO(accessmanagement.RoutingCatalog{}))
	if err != nil {
		t.Fatal(err)
	}
	var emptyPayload map[string]json.RawMessage
	if decodeErr := json.Unmarshal(emptyWire, &emptyPayload); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	for _, field := range []string{"models", "recipes", "entrypoints"} {
		if string(emptyPayload[field]) != "[]" {
			t.Errorf("RoutingCatalog.%s = %s, want []: %s", field, emptyPayload[field], emptyWire)
		}
	}

	mapped := routingCatalogDTO(accessmanagement.RoutingCatalog{
		Models:  []accessmanagement.RoutingCatalogModel{{}},
		Recipes: []accessmanagement.RoutingCatalogRecipe{{}},
		Entrypoints: []accessmanagement.RoutingCatalogEntrypoint{{
			Rules: []accessmanagement.RoutingCatalogRule{{
				Assignments: map[string]accessmanagement.RoutingCatalogAssignmentSet{"decision_empty": {}},
			}},
		}},
	})
	wire, err := json.Marshal(mapped)
	if err != nil {
		t.Fatal(err)
	}
	var payload struct {
		Models []struct {
			Aliases      json.RawMessage `json:"aliases"`
			Capabilities json.RawMessage `json:"capabilities"`
			LoRAs        json.RawMessage `json:"loras"`
			Tags         json.RawMessage `json:"tags"`
		} `json:"models"`
		Recipes []struct {
			Decisions   json.RawMessage `json:"decisions"`
			Signals     json.RawMessage `json:"signals"`
			Projections json.RawMessage `json:"projections"`
		} `json:"recipes"`
		Entrypoints []struct {
			Aliases json.RawMessage `json:"aliases"`
			Rules   []struct {
				Assignments map[string]struct {
					Models json.RawMessage `json:"models"`
				} `json:"assignments"`
			} `json:"rules"`
		} `json:"entrypoints"`
	}
	if err := json.Unmarshal(wire, &payload); err != nil {
		t.Fatal(err)
	}
	model := payload.Models[0]
	for field, value := range map[string]json.RawMessage{
		"aliases": model.Aliases, "capabilities": model.Capabilities, "loras": model.LoRAs, "tags": model.Tags,
	} {
		if string(value) != "[]" {
			t.Errorf("RoutingCatalogModel.%s = %s, want []: %s", field, value, wire)
		}
	}
	if string(payload.Recipes[0].Decisions) != "[]" || string(payload.Recipes[0].Signals) != "[]" ||
		string(payload.Recipes[0].Projections) != "[]" || string(payload.Entrypoints[0].Aliases) != "[]" ||
		string(payload.Entrypoints[0].Rules[0].Assignments["decision_empty"].Models) != "[]" {
		t.Fatalf("nested required collections must be arrays: %s", wire)
	}
}

func TestEffectivePolicyEncodesRequiredEmptyCollectionsAsArrays(t *testing.T) {
	subject := accessmanagement.Subject{
		Kind: accesscontrol.SubjectKindAPIKey,
		ID:   "44444444-4444-4444-8444-444444444444",
	}
	service := &accessReadServiceStub{
		inspection: accessmanagement.AuthorizationContext{Subject: subject},
		policy: accessmanagement.EffectivePolicy{
			Subject: subject, DesiredRevision: 2, AppliedRevision: 2,
			Quota: accessmanagement.EffectiveQuota{AsOf: time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC)},
		},
	}
	routes := newTestAccessReadRoutes(t, service, AuthorizerFunc(func(context.Context, AuthorizationRequest) (AuthorizationDecision, error) {
		return AuthorizationDecision{}, nil
	}))
	mux := http.NewServeMux()
	routes.Register(mux)
	request := authorizedRequest(t, http.MethodGet,
		managementapi.BasePath+"/api-keys/44444444-4444-4444-8444-444444444444/effective-policy", nil)
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var payload struct {
		Quota struct {
			Meters             json.RawMessage `json:"meters"`
			UnknownUsageFences json.RawMessage `json:"unknownUsageFences"`
		} `json:"quota"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if string(payload.Quota.Meters) != "[]" || string(payload.Quota.UnknownUsageFences) != "[]" {
		t.Fatalf("required collections must be arrays: %s", response.Body.String())
	}
	mapped := effectiveQuotaDTO(accessmanagement.EffectiveQuota{Meters: []accessmanagement.QuotaMeterView{{}}})
	activeFenceIDs, err := json.Marshal(mapped.Meters[0].ActiveFenceIDs)
	if err != nil {
		t.Fatal(err)
	}
	if string(activeFenceIDs) != "[]" {
		t.Fatalf("active fence IDs must encode as an array: %s", activeFenceIDs)
	}
	canonicalMeter := managementapi.QuotaMeter{
		PolicyID: "policy-1", RuleID: "rule-1", BindingID: "binding-1",
		Source: managementapi.GrantSource{
			SubjectType: "api_key", SubjectID: subject.ID, BindingID: "binding-1",
		},
		CounterOwner: subject.ID, Metric: "total_tokens", Algorithm: "sliding_log",
		Accounting: "response_actual", Enforcement: "enforce",
		Limit: "1", Used: "0", Remaining: nil,
		Completeness: "unknown", KnownDispatches: "0", IncompleteDispatches: "1",
		CapacityState: "fenced", ActiveFenceIDs: []string{"fence-1"},
		Freshness: managementapi.MeterFreshness{
			Source: "valkey", AsOf: time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC),
		},
	}
	meterWire, err := json.Marshal(canonicalMeter)
	if err != nil {
		t.Fatal(err)
	}
	var meterFields map[string]json.RawMessage
	if err := json.Unmarshal(meterWire, &meterFields); err != nil {
		t.Fatal(err)
	}
	for _, optional := range []string{"window", "currency", "overage", "resetAt"} {
		if _, present := meterFields[optional]; present {
			t.Errorf("empty optional QuotaMeter.%s must be omitted: %s", optional, meterWire)
		}
	}
	if string(meterFields["remaining"]) != "null" {
		t.Errorf("required nullable QuotaMeter.remaining = %s, want null: %s", meterFields["remaining"], meterWire)
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
