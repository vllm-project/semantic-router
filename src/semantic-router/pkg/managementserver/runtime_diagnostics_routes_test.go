package managementserver

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/runtimediagnostics"
)

type runtimeDiagnosticsServiceStub struct {
	snapshot    runtimediagnostics.Snapshot
	err         error
	namespaceID string
	calls       int
}

func (stub *runtimeDiagnosticsServiceStub) Read(
	_ context.Context,
	namespaceID string,
) (runtimediagnostics.Snapshot, error) {
	stub.calls++
	stub.namespaceID = namespaceID
	return stub.snapshot, stub.err
}

func TestRuntimeDiagnosticsRequiresClusterAuthorityAndUsesExactSelector(t *testing.T) {
	service := &runtimeDiagnosticsServiceStub{snapshot: runtimediagnostics.Snapshot{
		Status: "ready", AsOf: time.Unix(100, 0).UTC(),
		PostgreSQL:           runtimediagnostics.StoreStatus{Status: "ready"},
		Valkey:               runtimediagnostics.StoreStatus{Status: "ready"},
		RegisteredNamespaces: 12,
		Namespace: &runtimediagnostics.NamespaceDiagnostics{
			NamespaceID: testNamespaceID, QuotaPartition: "partition-1",
			Publication: accesspublisher.RuntimeDiagnostics{
				NamespaceID: testNamespaceID, QuotaPartition: "partition-1",
				AsOf:      time.Unix(100, 0).UTC(),
				Readiness: accesspublisher.ReadinessDiagnostics{Ready: true},
			},
			Quota: quotaruntime.PartitionDiagnostics{
				Partition: "partition-1", AsOf: time.Unix(100, 0).UTC(), RecoveryState: "ready",
			},
			UsageStreamBacklogLimit: 100,
		},
	}}
	authorizer := &authorizerStub{}
	routes := newTestRuntimeDiagnosticsRoutes(t, service, sessionStub{}, authorizer)
	request := httptest.NewRequest(http.MethodGet, runtimeDiagnosticsPath+"?namespaceId="+testNamespaceID, nil)
	request.Header.Set("Authorization", "Bearer token")
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"registeredNamespaces":12`) ||
		!strings.Contains(response.Body.String(), `"namespaceId":"`+testNamespaceID+`"`) {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	if service.calls != 1 || service.namespaceID != testNamespaceID {
		t.Fatalf("service calls=%d namespace=%q", service.calls, service.namespaceID)
	}
	if authorizer.calls != 1 || authorizer.last.NamespaceID != "" || authorizer.last.Operation.Scope != "cluster" ||
		authorizer.last.Session.NamespaceID != "" {
		t.Fatalf("authorization=%#v", authorizer.last)
	}
}

func TestRuntimeDiagnosticsFailsClosedBeforeReadingService(t *testing.T) {
	service := &runtimeDiagnosticsServiceStub{}
	authorizer := &authorizerStub{err: managementauthorization.ErrDenied}
	routes := newTestRuntimeDiagnosticsRoutes(t, service, sessionStub{}, authorizer)
	request := httptest.NewRequest(http.MethodGet, runtimeDiagnosticsPath, nil)
	request.Header.Set("Authorization", "Bearer token")
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusForbidden || service.calls != 0 {
		t.Fatalf("status=%d calls=%d body=%s", response.Code, service.calls, response.Body.String())
	}
}

func TestRuntimeDiagnosticsValidatesSelectorAndNondisclosesMissingNamespace(t *testing.T) {
	service := &runtimeDiagnosticsServiceStub{err: runtimediagnostics.ErrNotFound}
	routes := newTestRuntimeDiagnosticsRoutes(t, service, sessionStub{}, &authorizerStub{})

	invalid := httptest.NewRequest(http.MethodGet, runtimeDiagnosticsPath+"?namespaceId=not-a-uuid", nil)
	invalid.Header.Set("Authorization", "Bearer token")
	invalidResponse := httptest.NewRecorder()
	routes.ServeHTTP(invalidResponse, invalid)
	if invalidResponse.Code != http.StatusBadRequest || service.calls != 0 {
		t.Fatalf("invalid status=%d calls=%d", invalidResponse.Code, service.calls)
	}

	missing := httptest.NewRequest(http.MethodGet, runtimeDiagnosticsPath+"?namespaceId="+testNamespaceID, nil)
	missing.Header.Set("Authorization", "Bearer token")
	missingResponse := httptest.NewRecorder()
	routes.ServeHTTP(missingResponse, missing)
	if missingResponse.Code != http.StatusNotFound || strings.Contains(missingResponse.Body.String(), testNamespaceID) {
		t.Fatalf("missing status=%d body=%s", missingResponse.Code, missingResponse.Body.String())
	}
}

func TestRuntimeDiagnosticsRejectsNamespacedSession(t *testing.T) {
	service := &runtimeDiagnosticsServiceStub{}
	routes := newTestRuntimeDiagnosticsRoutes(t, service, namespacedDiagnosticsSession{}, &authorizerStub{})
	request := httptest.NewRequest(http.MethodGet, runtimeDiagnosticsPath, nil)
	request.Header.Set("Authorization", "Bearer token")
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusServiceUnavailable || service.calls != 0 {
		t.Fatalf("status=%d calls=%d body=%s", response.Code, service.calls, response.Body.String())
	}
}

type namespacedDiagnosticsSession struct{}

func (namespacedDiagnosticsSession) Authenticate(
	context.Context,
	string,
	string,
	time.Time,
) (managementauth.AuthenticatedSession, error) {
	return managementauth.AuthenticatedSession{
		NamespaceID: testNamespaceID,
		Session:     managementauth.LiveSession{Session: managementauth.Session{PrincipalID: testPrincipalID}},
	}, nil
}

func newTestRuntimeDiagnosticsRoutes(
	t *testing.T,
	service RuntimeDiagnosticsService,
	sessions SessionAuthenticator,
	authorization Authorizer,
) *RuntimeDiagnosticsRoutes {
	t.Helper()
	routes, err := NewRuntimeDiagnosticsRoutes(RuntimeDiagnosticsRoutesOptions{
		Service: service, Sessions: sessions, Authorization: authorization,
		Now: func() time.Time { return time.Unix(100, 0).UTC() },
	})
	if err != nil {
		t.Fatal(err)
	}
	return routes
}
