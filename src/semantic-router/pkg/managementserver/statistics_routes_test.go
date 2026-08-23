package managementserver

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementstatistics"
)

const (
	statisticsUserID = "22222222-2222-4222-8222-222222222222"
	statisticsTeamID = "33333333-3333-4333-8333-333333333333"
)

type statisticsServiceStub struct {
	snapshot managementstatistics.Snapshot
	err      error
	last     managementstatistics.Request
	calls    int
}

func (stub *statisticsServiceStub) Ready(context.Context) error { return stub.err }

func (stub *statisticsServiceStub) Snapshot(
	_ context.Context,
	request managementstatistics.Request,
) (managementstatistics.Snapshot, error) {
	stub.calls++
	stub.last = request
	return stub.snapshot, stub.err
}

type statisticsScopeStub struct {
	values map[accesscontrol.Permission]accesscontrol.ResultScope
	errors map[accesscontrol.Permission]error
	calls  []accesscontrol.Permission
}

func (stub *statisticsScopeStub) ResolveResultScope(
	_ context.Context,
	_ accesscontrol.ManagementPrincipalID,
	_ accesscontrol.NamespaceID,
	permission accesscontrol.Permission,
) (managementauthorization.ResultScope, error) {
	stub.calls = append(stub.calls, permission)
	if err := stub.errors[permission]; err != nil {
		return accesscontrol.ResultScope{}, err
	}
	value, found := stub.values[permission]
	if !found {
		return accesscontrol.ResultScope{}, managementauthorization.ErrDenied
	}
	return value, nil
}

func TestStatisticsRoutesProjectsEachFieldFromItsOwnReadScope(t *testing.T) {
	now := time.Date(2026, 8, 23, 10, 0, 0, 0, time.UTC)
	service := &statisticsServiceStub{snapshot: managementstatistics.Snapshot{
		AsOf: now, ExpiringBefore: now.Add(managementstatistics.DefaultExpiringWindow),
		Users: statisticsCount("2"), Teams: statisticsCount("1"),
		ActiveAPIKeys: statisticsCount("10000"), ExpiringAPIKeys: statisticsCount("3"),
		ActiveRatePolicies: statisticsCount("4"),
	}}
	scopes := &statisticsScopeStub{
		values: map[accesscontrol.Permission]accesscontrol.ResultScope{
			accesscontrol.PermissionUsageRead:      {NamespaceID: testNamespaceID, All: true},
			accesscontrol.PermissionUserRead:       {NamespaceID: testNamespaceID, UserIDs: []accesscontrol.UserID{accesscontrol.UserID(statisticsUserID)}},
			accesscontrol.PermissionTeamRead:       {NamespaceID: testNamespaceID, TeamIDs: []accesscontrol.TeamID{accesscontrol.TeamID(statisticsTeamID)}},
			accesscontrol.PermissionKeyRead:        {NamespaceID: testNamespaceID, All: true},
			accesscontrol.PermissionRatePolicyRead: {NamespaceID: testNamespaceID, All: true},
		},
		errors: map[accesscontrol.Permission]error{
			accesscontrol.PermissionAccessPolicyRead: managementauthorization.ErrDenied,
		},
	}
	routes := newTestStatisticsRoutes(t, service, scopes, now)
	response := serveStatisticsRequest(routes, statisticsPath)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["activeApiKeys"] != "10000" || body["users"] != "2" || body["activeRatePolicies"] != "4" {
		t.Fatalf("unexpected statistics response: %#v", body)
	}
	if _, leaked := body["accessPolicies"]; leaked {
		t.Fatalf("denied field was disclosed: %#v", body)
	}
	if service.calls != 1 || service.last.Scopes.AccessPolicies != nil || service.last.Scopes.APIKeys == nil ||
		!service.last.Scopes.APIKeys.All {
		t.Fatalf("unexpected projected request: %#v", service.last)
	}
	if len(scopes.calls) != 6 || scopes.calls[0] != accesscontrol.PermissionUsageRead {
		t.Fatalf("scope resolution order = %v", scopes.calls)
	}
}

func TestStatisticsRoutesRequireUsageReadBeforeResolvingFields(t *testing.T) {
	service := &statisticsServiceStub{}
	scopes := &statisticsScopeStub{values: map[accesscontrol.Permission]accesscontrol.ResultScope{}, errors: map[accesscontrol.Permission]error{
		accesscontrol.PermissionUsageRead: managementauthorization.ErrDenied,
	}}
	routes := newTestStatisticsRoutes(t, service, scopes, time.Now())
	response := serveStatisticsRequest(routes, statisticsPath)
	if response.Code != http.StatusForbidden || service.calls != 0 || len(scopes.calls) != 1 {
		t.Fatalf("status=%d calls=%d scopeCalls=%v body=%s", response.Code, service.calls, scopes.calls, response.Body.String())
	}
}

func TestStatisticsRoutesFailClosedWhenFieldScopeIsUnavailable(t *testing.T) {
	service := &statisticsServiceStub{}
	scopes := &statisticsScopeStub{
		values: map[accesscontrol.Permission]accesscontrol.ResultScope{
			accesscontrol.PermissionUsageRead: {NamespaceID: testNamespaceID, All: true},
		},
		errors: map[accesscontrol.Permission]error{
			accesscontrol.PermissionUserRead: errors.New("authorization store unavailable"),
		},
	}
	routes := newTestStatisticsRoutes(t, service, scopes, time.Now())
	response := serveStatisticsRequest(routes, statisticsPath)
	if response.Code != http.StatusServiceUnavailable || service.calls != 0 {
		t.Fatalf("status=%d calls=%d body=%s", response.Code, service.calls, response.Body.String())
	}
}

func TestStatisticsRoutesRejectUnboundedRequestShapes(t *testing.T) {
	routes := newTestStatisticsRoutes(t, &statisticsServiceStub{}, &statisticsScopeStub{}, time.Now())
	for _, target := range []string{statisticsPath + "?pageSize=200", statisticsPath + "/"} {
		response := serveStatisticsRequest(routes, target)
		if response.Code != http.StatusNotFound {
			t.Fatalf("%s status=%d body=%s", target, response.Code, response.Body.String())
		}
	}
}

func newTestStatisticsRoutes(
	t *testing.T,
	service StatisticsQueryService,
	scopes ResultScopeResolver,
	now time.Time,
) *StatisticsRoutes {
	t.Helper()
	routes, err := NewStatisticsRoutes(StatisticsRoutesOptions{
		Service: service, Scopes: scopes,
		Namespaces: NamespaceResolverFunc(func(context.Context, *http.Request) (string, error) {
			return testNamespaceID, nil
		}),
		Sessions: sessionStub{}, Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	return routes
}

func serveStatisticsRequest(routes *StatisticsRoutes, target string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(http.MethodGet, target, nil)
	request.Header.Set("Authorization", "Bearer management-token")
	request.Header.Set(managementapi.HeaderNamespaceID, testNamespaceID)
	response := httptest.NewRecorder()
	mux := http.NewServeMux()
	routes.Register(mux)
	mux.ServeHTTP(response, request)
	return response
}

func statisticsCount(value string) *managementstatistics.Count {
	count := managementstatistics.Count(value)
	return &count
}

var (
	_ StatisticsQueryService = (*statisticsServiceStub)(nil)
	_ ResultScopeResolver    = (*statisticsScopeStub)(nil)
)
