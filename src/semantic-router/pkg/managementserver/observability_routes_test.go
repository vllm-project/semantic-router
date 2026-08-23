package managementserver

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/auditlog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/subjectmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

const (
	observabilityTeamID = "33333333-3333-4333-8333-333333333333"
	observabilityUserID = "55555555-5555-4555-8555-555555555555"
)

func TestSubjectUsageRoutePinsPathResourceIntoAuthorizationAndSQLQuery(t *testing.T) {
	queries := &usageQueryServiceStub{summary: usageledger.UsageSummary{Totals: usageledger.UsageTotals{Costs: []usageledger.CostSummary{}}}}
	routes := newTestObservabilityRoutes(t, queries, &resultScopeStub{values: map[accesscontrol.Permission]managementauthorization.ResultScope{}})
	authorizer := &authorizerStub{}
	routes.authorization = authorizer

	response := serveObservabilityRequest(t, routes,
		managementapi.BasePath+"/users/"+observabilityUserID+"/usage?grain=hour", testNamespaceID)

	if response.Code != http.StatusOK || queries.summaryCalls != 1 {
		t.Fatalf("subject usage status=%d calls=%d body=%s", response.Code, queries.summaryCalls, response.Body.String())
	}
	if queries.lastUsage.Filters.UserID != observabilityUserID ||
		len(queries.lastUsage.Visibility.UserIDs) != 1 || queries.lastUsage.Visibility.UserIDs[0] != observabilityUserID ||
		queries.lastUsage.Visibility.All {
		t.Fatalf("subject usage query = %#v", queries.lastUsage)
	}
	targets := authorizer.last.Targets["user"]
	if len(targets) != 1 || targets[0].Scope != accesscontrol.UserScope(testNamespaceID, observabilityUserID) {
		t.Fatalf("subject usage authorization targets = %#v", authorizer.last.Targets)
	}
}

func TestSubjectUsageRouteIsNondisclosingForDeniedAndAbsentResources(t *testing.T) {
	for name, configure := range map[string]func(*ObservabilityRoutes){
		"denied": func(routes *ObservabilityRoutes) {
			routes.authorization = &authorizerStub{err: managementauthorization.ErrDenied}
		},
		"absent": func(routes *ObservabilityRoutes) {
			routes.resources = &usageResourceReaderStub{userErr: subjectmanagement.ErrNotFound}
		},
	} {
		t.Run(name, func(t *testing.T) {
			queries := &usageQueryServiceStub{}
			routes := newTestObservabilityRoutes(t, queries, &resultScopeStub{values: map[accesscontrol.Permission]managementauthorization.ResultScope{}})
			configure(routes)
			response := serveObservabilityRequest(t, routes,
				managementapi.BasePath+"/users/"+observabilityUserID+"/usage", testNamespaceID)
			if response.Code != http.StatusNotFound || queries.summaryCalls != 0 {
				t.Fatalf("status=%d calls=%d body=%s", response.Code, queries.summaryCalls, response.Body.String())
			}
		})
	}
}

func TestObservabilityUsagePushesAuthorizedScopeIntoQuery(t *testing.T) {
	queries := &usageQueryServiceStub{summary: usageledger.UsageSummary{
		Totals: usageledger.UsageTotals{Requests: "12", SuccessfulRequests: "10", InputTokens: "20", OutputTokens: "5", TotalTokens: "25", IncompleteDispatches: "0", Completeness: usageledger.CompletenessComplete, Costs: []usageledger.CostSummary{}},
		Grain:  usageledger.GrainHour, Final: false,
	}}
	routes := newTestObservabilityRoutes(t, queries, &resultScopeStub{values: map[accesscontrol.Permission]managementauthorization.ResultScope{
		accesscontrol.PermissionUsageRead: {NamespaceID: testNamespaceID, TeamIDs: []accesscontrol.TeamID{observabilityTeamID}},
	}})
	response := serveObservabilityRequest(t, routes, usagePath+"?grain=hour", testNamespaceID)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"requests":"12"`) {
		t.Fatalf("usage status=%d body=%s", response.Code, response.Body.String())
	}
	if queries.summaryCalls != 1 || queries.lastUsage.Visibility.All ||
		len(queries.lastUsage.Visibility.TeamIDs) != 1 || queries.lastUsage.Visibility.TeamIDs[0] != observabilityTeamID {
		t.Fatalf("authorized query=%#v", queries.lastUsage)
	}
	if queries.lastUsage.TimeZone != "UTC" {
		t.Fatalf("default query time zone = %q, want UTC", queries.lastUsage.TimeZone)
	}
}

func TestObservabilityInternalBreakdownRequiresMatchingInternalScope(t *testing.T) {
	queries := &usageQueryServiceStub{}
	scopes := &resultScopeStub{values: map[accesscontrol.Permission]managementauthorization.ResultScope{
		accesscontrol.PermissionUsageRead: {NamespaceID: testNamespaceID, TeamIDs: []accesscontrol.TeamID{observabilityTeamID}},
	}, errors: map[accesscontrol.Permission]error{
		accesscontrol.PermissionUsageInternalDimensionsRead: managementauthorization.ErrDenied,
	}}
	routes := newTestObservabilityRoutes(t, queries, scopes)
	response := serveObservabilityRequest(t, routes, usagePath+"/breakdowns?dimension=logical_model", testNamespaceID)
	if response.Code != http.StatusForbidden || queries.breakdownCalls != 0 {
		t.Fatalf("internal breakdown status=%d calls=%d body=%s", response.Code, queries.breakdownCalls, response.Body.String())
	}
}

func TestObservabilityRequestDetailRedactsDispatchesWithoutInternalPermission(t *testing.T) {
	queries := &usageQueryServiceStub{detail: usageledger.RequestDetail{
		Request:    usageledger.RequestLog{AdmissionID: "admission-1", EventID: "44444444-4444-4444-8444-444444444444", Costs: []usageledger.CostSummary{}},
		Dispatches: []usageledger.DispatchDetail{{DispatchID: "dispatch-1", ProviderID: "private-provider"}},
	}}
	scopes := &resultScopeStub{values: map[accesscontrol.Permission]managementauthorization.ResultScope{
		accesscontrol.PermissionLogRead: {NamespaceID: testNamespaceID, TeamIDs: []accesscontrol.TeamID{observabilityTeamID}},
	}, errors: map[accesscontrol.Permission]error{
		accesscontrol.PermissionUsageInternalDimensionsRead: managementauthorization.ErrDenied,
	}}
	routes := newTestObservabilityRoutes(t, queries, scopes)
	target := managementapi.BasePath + "/namespaces/" + testNamespaceID + "/request-logs/admission-1"
	response := serveObservabilityRequest(t, routes, target, testNamespaceID)
	if response.Code != http.StatusOK || strings.Contains(response.Body.String(), "private-provider") ||
		!strings.Contains(response.Body.String(), `"request"`) {
		t.Fatalf("request detail status=%d body=%s", response.Code, response.Body.String())
	}
	if len(queries.lastVisibility.TeamIDs) != 1 || queries.lastVisibility.TeamIDs[0] != observabilityTeamID {
		t.Fatalf("request detail visibility=%#v", queries.lastVisibility)
	}
}

func TestObservabilityRequestLogPageUsesCanonicalEnvelope(t *testing.T) {
	queries := &usageQueryServiceStub{logs: usageledger.LogPage{
		Items:      []usageledger.RequestLog{{AdmissionID: "admission-1", EventID: "44444444-4444-4444-8444-444444444444", Costs: []usageledger.CostSummary{}}},
		NextCursor: "signed-cursor",
	}}
	routes := newTestObservabilityRoutes(t, queries, &resultScopeStub{values: map[accesscontrol.Permission]managementauthorization.ResultScope{
		accesscontrol.PermissionLogRead: {NamespaceID: testNamespaceID, All: true},
	}})
	response := serveObservabilityRequest(t, routes, requestLogsPath+"?pageSize=1", testNamespaceID)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"data"`) ||
		!strings.Contains(response.Body.String(), `"nextCursor":"signed-cursor"`) ||
		!strings.Contains(response.Body.String(), `"hasMore":true`) {
		t.Fatalf("request-log page status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestObservabilityAuditRequiresNamespaceWideAuthority(t *testing.T) {
	queries := &usageQueryServiceStub{}
	audit := &auditQueryServiceStub{page: auditlog.Page{Items: []auditlog.Event{{ID: "event"}}}}
	routes := newTestObservabilityRoutesWithAudit(t, queries, audit, &resultScopeStub{values: map[accesscontrol.Permission]managementauthorization.ResultScope{
		accesscontrol.PermissionAuditRead: {NamespaceID: testNamespaceID, TeamIDs: []accesscontrol.TeamID{observabilityTeamID}},
	}})
	response := serveObservabilityRequest(t, routes, auditEventsPath, testNamespaceID)
	if response.Code != http.StatusForbidden || audit.calls != 0 {
		t.Fatalf("narrow audit status=%d calls=%d body=%s", response.Code, audit.calls, response.Body.String())
	}
}

func TestObservabilityAuditUsesCanonicalEnvelopeAndTypedFilters(t *testing.T) {
	queries := &usageQueryServiceStub{}
	audit := &auditQueryServiceStub{page: auditlog.Page{
		Items:      []auditlog.Event{{ID: "44444444-4444-4444-8444-444444444444", ActorChain: []string{}, Details: map[string]string{}}},
		NextCursor: "signed-audit-cursor",
	}}
	routes := newTestObservabilityRoutesWithAudit(t, queries, audit, &resultScopeStub{values: map[accesscontrol.Permission]managementauthorization.ResultScope{
		accesscontrol.PermissionAuditRead: {NamespaceID: testNamespaceID, All: true},
	}})
	response := serveObservabilityRequest(t, routes, auditEventsPath+"?action=key.disable&pageSize=1", testNamespaceID)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"nextCursor":"signed-audit-cursor"`) ||
		!strings.Contains(response.Body.String(), `"hasMore":true`) {
		t.Fatalf("audit page status=%d body=%s", response.Code, response.Body.String())
	}
	if audit.calls != 1 || audit.last.Filters.Action != "key.disable" || audit.last.PageSize != 1 {
		t.Fatalf("audit query=%#v calls=%d", audit.last, audit.calls)
	}
}

type usageQueryServiceStub struct {
	summary        usageledger.UsageSummary
	series         usageledger.UsageSeries
	breakdown      usageledger.UsageBreakdown
	logs           usageledger.LogPage
	detail         usageledger.RequestDetail
	err            error
	lastUsage      usageledger.UsageQuery
	lastVisibility usageledger.QueryVisibility
	summaryCalls   int
	breakdownCalls int
}

func (stub *usageQueryServiceStub) Summary(_ context.Context, query usageledger.UsageQuery) (usageledger.UsageSummary, error) {
	stub.summaryCalls++
	stub.lastUsage = query
	return stub.summary, stub.err
}

func (stub *usageQueryServiceStub) Series(_ context.Context, query usageledger.UsageQuery) (usageledger.UsageSeries, error) {
	stub.lastUsage = query
	return stub.series, stub.err
}

func (stub *usageQueryServiceStub) Breakdown(_ context.Context, query usageledger.BreakdownQuery) (usageledger.UsageBreakdown, error) {
	stub.breakdownCalls++
	stub.lastUsage = query.UsageQuery
	return stub.breakdown, stub.err
}

func (stub *usageQueryServiceStub) ListLogs(_ context.Context, query usageledger.LogQuery, _ *usageledger.LogCursorCodec) (usageledger.LogPage, error) {
	stub.lastVisibility = query.Visibility
	return stub.logs, stub.err
}

func (stub *usageQueryServiceStub) RequestDetail(_ context.Context, _, _ string, visibility usageledger.QueryVisibility) (usageledger.RequestDetail, error) {
	stub.lastVisibility = visibility
	return stub.detail, stub.err
}

type resultScopeStub struct {
	values map[accesscontrol.Permission]managementauthorization.ResultScope
	errors map[accesscontrol.Permission]error
}

type auditQueryServiceStub struct {
	page  auditlog.Page
	err   error
	last  auditlog.Query
	calls int
}

type usageResourceReaderStub struct {
	userErr error
	teamErr error
	keyErr  error
}

func (stub *usageResourceReaderStub) GetUser(_ context.Context, namespaceID, userID string) (subjectmanagement.User, error) {
	return subjectmanagement.User{ID: userID, NamespaceID: namespaceID}, stub.userErr
}

func (stub *usageResourceReaderStub) GetTeam(_ context.Context, namespaceID, teamID string) (subjectmanagement.Team, error) {
	return subjectmanagement.Team{ID: teamID, NamespaceID: namespaceID}, stub.teamErr
}

func (stub *usageResourceReaderStub) GetAPIKey(_ context.Context, namespaceID, keyID string) (accesscontrol.APIKey, error) {
	return accesscontrol.APIKey{ID: accesscontrol.APIKeyID(keyID), NamespaceID: accesscontrol.NamespaceID(namespaceID)}, stub.keyErr
}

func (stub *auditQueryServiceStub) List(_ context.Context, query auditlog.Query, _ *auditlog.CursorCodec) (auditlog.Page, error) {
	stub.calls++
	stub.last = query
	return stub.page, stub.err
}

func (stub *resultScopeStub) ResolveResultScope(
	_ context.Context,
	_ accesscontrol.ManagementPrincipalID,
	_ accesscontrol.NamespaceID,
	permission accesscontrol.Permission,
) (managementauthorization.ResultScope, error) {
	if err := stub.errors[permission]; err != nil {
		return managementauthorization.ResultScope{}, err
	}
	value, found := stub.values[permission]
	if !found {
		return managementauthorization.ResultScope{}, managementauthorization.ErrDenied
	}
	return value, nil
}

func newTestObservabilityRoutes(t *testing.T, queries UsageQueryService, scopes ResultScopeResolver) *ObservabilityRoutes {
	return newTestObservabilityRoutesWithAudit(t, queries, &auditQueryServiceStub{}, scopes)
}

func newTestObservabilityRoutesWithAudit(
	t *testing.T,
	queries UsageQueryService,
	audit AuditQueryService,
	scopes ResultScopeResolver,
) *ObservabilityRoutes {
	t.Helper()
	codec, err := usageledger.NewLogCursorCodec([]byte("0123456789abcdef0123456789abcdef"))
	if err != nil {
		t.Fatal(err)
	}
	auditCodec, err := auditlog.NewCursorCodec([]byte("abcdef0123456789abcdef0123456789"))
	if err != nil {
		t.Fatal(err)
	}
	routes, err := NewObservabilityRoutes(ObservabilityRoutesOptions{
		Queries: queries, LogCursors: codec, Audit: audit, AuditCursors: auditCodec,
		Resources: &usageResourceReaderStub{}, Authorization: &authorizerStub{}, Scopes: scopes,
		Namespaces: NamespaceResolverFunc(func(context.Context, *http.Request) (string, error) {
			return testNamespaceID, nil
		}),
		Sessions: sessionStub{}, Now: func() time.Time { return time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatal(err)
	}
	return routes
}

func serveObservabilityRequest(t *testing.T, routes *ObservabilityRoutes, target, namespaceID string) *httptest.ResponseRecorder {
	t.Helper()
	request := httptest.NewRequest(http.MethodGet, target, nil)
	request.Header.Set("Authorization", "Bearer management-token")
	request.Header.Set(managementapi.HeaderNamespaceID, namespaceID)
	response := httptest.NewRecorder()
	mux := http.NewServeMux()
	routes.Register(mux)
	mux.ServeHTTP(response, request)
	return response
}

var (
	_ UsageQueryService   = (*usageQueryServiceStub)(nil)
	_ ResultScopeResolver = (*resultScopeStub)(nil)
	_ AuditQueryService   = (*auditQueryServiceStub)(nil)
	_ UsageResourceReader = (*usageResourceReaderStub)(nil)
)
