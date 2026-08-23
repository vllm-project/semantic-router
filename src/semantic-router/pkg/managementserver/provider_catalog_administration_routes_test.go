package managementserver

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
)

type providerCatalogAdministrationStub struct {
	bootstrapState      providercatalog.PublicationState
	activateState       providercatalog.PublicationState
	bootstrapErr        error
	activateErr         error
	bootstrapGeneration uint64
	activateGeneration  uint64
	activateRevision    string
	bootstrapCalls      int
	activateCalls       int
}

func (stub *providerCatalogAdministrationStub) BootstrapRegistry(
	_ context.Context,
	expectedGeneration uint64,
) (providercatalog.PublicationState, error) {
	stub.bootstrapCalls++
	stub.bootstrapGeneration = expectedGeneration
	return stub.bootstrapState, stub.bootstrapErr
}

func (stub *providerCatalogAdministrationStub) Activate(
	_ context.Context,
	revision string,
	expectedGeneration uint64,
) (providercatalog.PublicationState, error) {
	stub.activateCalls++
	stub.activateRevision = revision
	stub.activateGeneration = expectedGeneration
	return stub.activateState, stub.activateErr
}

func TestProviderCatalogAdministrationStagesAndActivatesWithClusterAuthority(t *testing.T) {
	now := time.Date(2026, 8, 22, 13, 0, 0, 0, time.UTC)
	administration := &providerCatalogAdministrationStub{
		bootstrapState: providercatalog.PublicationState{
			DesiredRevision: testRevision, Generation: 2, UpdatedAt: now,
		},
		activateState: providercatalog.PublicationState{
			DesiredRevision: testRevision, ActiveRevision: testRevision, Generation: 2, UpdatedAt: now,
		},
	}
	routes, authorizer := newTestProviderCatalogAdministrationRoutes(t, administration, sessionStub{})
	mux := http.NewServeMux()
	routes.Register(mux)

	request := authorizedRequest(t, http.MethodPost, providerCatalogBootstrapPath,
		strings.NewReader(`{"expectedGeneration":"1"}`))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("bootstrap status = %d, body = %s", response.Code, response.Body.String())
	}
	var bootstrap managementapi.ProviderCatalogPublication
	if err := json.Unmarshal(response.Body.Bytes(), &bootstrap); err != nil {
		t.Fatal(err)
	}
	if administration.bootstrapCalls != 1 || administration.bootstrapGeneration != 1 ||
		bootstrap.DesiredRevision != testRevision || bootstrap.Generation != "2" || bootstrap.ActiveRevision != "" {
		t.Fatalf("bootstrap = %+v, administration = %+v", bootstrap, administration)
	}
	if authorizer.last.NamespaceID != "" || authorizer.last.Operation.Scope != managementapi.ScopeCluster ||
		authorizer.last.Operation.Permission.Canonical() != "cluster.manage@cluster" {
		t.Fatalf("bootstrap authorization = %+v", authorizer.last)
	}

	request = authorizedRequest(t, http.MethodPost, providerCatalogActivatePath,
		strings.NewReader(`{"revision":"`+testRevision+`","expectedGeneration":"2"}`))
	request.Header.Set("Content-Type", managementapi.JSONMediaType+"; charset=utf-8")
	response = httptest.NewRecorder()
	mux.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("activate status = %d, body = %s", response.Code, response.Body.String())
	}
	var activated managementapi.ProviderCatalogPublication
	if err := json.Unmarshal(response.Body.Bytes(), &activated); err != nil {
		t.Fatal(err)
	}
	if administration.activateCalls != 1 || administration.activateRevision != testRevision ||
		administration.activateGeneration != 2 || activated.ActiveRevision != testRevision || activated.Generation != "2" {
		t.Fatalf("activation = %+v, administration = %+v", activated, administration)
	}
}

func TestProviderCatalogAdministrationFailsClosed(t *testing.T) {
	tests := []struct {
		name       string
		path       string
		body       string
		sessionErr error
		authzErr   error
		domainErr  error
		wantStatus int
		wantCode   string
	}{
		{name: "numeric generation", path: providerCatalogBootstrapPath, body: `{"expectedGeneration":1}`, wantStatus: 400, wantCode: "invalid_request"},
		{name: "zero generation", path: providerCatalogBootstrapPath, body: `{"expectedGeneration":"0"}`, wantStatus: 400, wantCode: "invalid_request"},
		{name: "unknown field", path: providerCatalogBootstrapPath, body: `{"expectedGeneration":"1","provider":{}}`, wantStatus: 400, wantCode: "invalid_request"},
		{name: "bad activation revision", path: providerCatalogActivatePath, body: `{"revision":"latest","expectedGeneration":"2"}`, wantStatus: 400, wantCode: "invalid_request"},
		{name: "authentication denied", path: providerCatalogBootstrapPath, body: `{"expectedGeneration":"1"}`, sessionErr: errors.New("session store unavailable"), wantStatus: 503, wantCode: "authentication_unavailable"},
		{name: "authorization denied", path: providerCatalogBootstrapPath, body: `{"expectedGeneration":"1"}`, authzErr: managementauthorization.ErrDenied, wantStatus: 403, wantCode: "forbidden"},
		{name: "compare and swap", path: providerCatalogBootstrapPath, body: `{"expectedGeneration":"1"}`, domainErr: providercatalog.ErrPublicationConflict, wantStatus: 409, wantCode: "catalog_conflict"},
		{name: "activation blocked", path: providerCatalogActivatePath, body: `{"revision":"` + testRevision + `","expectedGeneration":"2"}`, domainErr: &providercatalog.ActivationBlockedError{
			Revision: testRevision, Blockers: providercatalog.ActivationBlockers{
				Missing: []providercatalog.RolloutGroup{{Plane: providercatalog.CapabilityPlaneData, ID: "router"}},
			},
		}, wantStatus: 409, wantCode: "activation_blocked"},
		{name: "storage failure", path: providerCatalogBootstrapPath, body: `{"expectedGeneration":"1"}`, domainErr: errors.New("dsn=private"), wantStatus: 503, wantCode: "catalog_unavailable"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			administration := &providerCatalogAdministrationStub{}
			if test.path == providerCatalogActivatePath {
				administration.activateErr = test.domainErr
			} else {
				administration.bootstrapErr = test.domainErr
			}
			routes, authorizer := newTestProviderCatalogAdministrationRoutes(t, administration, sessionStub{err: test.sessionErr})
			authorizer.err = test.authzErr
			request := authorizedRequest(t, http.MethodPost, test.path, strings.NewReader(test.body))
			request.Header.Set("Content-Type", managementapi.JSONMediaType)
			response := httptest.NewRecorder()
			routes.ServeHTTP(response, request)
			if response.Code != test.wantStatus || !strings.Contains(response.Body.String(), `"code":"`+test.wantCode+`"`) {
				t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
			}
			if strings.Contains(response.Body.String(), "private") {
				t.Fatalf("internal error leaked: %s", response.Body.String())
			}
			if test.wantStatus == 400 || test.sessionErr != nil || test.authzErr != nil {
				if administration.bootstrapCalls+administration.activateCalls != 0 {
					t.Fatalf("domain called after rejected request: %+v", administration)
				}
			}
		})
	}
}

func TestProviderCatalogActivationBlockedResponseOmitsReplicaIdentity(t *testing.T) {
	administration := &providerCatalogAdministrationStub{activateErr: &providercatalog.ActivationBlockedError{
		Revision: testRevision,
		Blockers: providercatalog.ActivationBlockers{Incompatible: []providercatalog.ReplicaBlocker{{
			RolloutGroup: providercatalog.RolloutGroup{Plane: providercatalog.CapabilityPlaneControl, ID: "management"},
			ReplicaID:    "private-pod-identity", Reason: "internal validation detail",
		}}},
	}}
	routes, _ := newTestProviderCatalogAdministrationRoutes(t, administration, sessionStub{})
	request := authorizedRequest(t, http.MethodPost, providerCatalogActivatePath,
		strings.NewReader(`{"revision":"`+testRevision+`","expectedGeneration":"2"}`))
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	response := httptest.NewRecorder()
	routes.ServeHTTP(response, request)
	if response.Code != http.StatusConflict || strings.Contains(response.Body.String(), "private-pod-identity") ||
		strings.Contains(response.Body.String(), "internal validation detail") ||
		!strings.Contains(response.Body.String(), "control/management") {
		t.Fatalf("blocked response = %d %s", response.Code, response.Body.String())
	}
}

func newTestProviderCatalogAdministrationRoutes(
	t *testing.T,
	administration ProviderCatalogAdministration,
	sessions SessionAuthenticator,
) (*ProviderCatalogAdministrationRoutes, *authorizerStub) {
	t.Helper()
	authorizer := &authorizerStub{}
	routes, err := NewProviderCatalogAdministrationRoutes(ProviderCatalogAdministrationRoutesOptions{
		Administration: administration, Sessions: sessions, Authorization: authorizer,
		Now: func() time.Time { return time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatal(err)
	}
	return routes, authorizer
}
