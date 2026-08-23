package managementserver

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

type catalogRuntimeStub struct {
	readyErr error
	runErr   error
	runCalls int
}

type readinessRegistrarStub struct{ err error }

func (registrar *readinessRegistrarStub) Register(*http.ServeMux)     {}
func (registrar *readinessRegistrarStub) Ready(context.Context) error { return registrar.err }

func (runtime *catalogRuntimeStub) Ready(context.Context) error { return runtime.readyErr }
func (runtime *catalogRuntimeStub) Run(context.Context) error {
	runtime.runCalls++
	return runtime.runErr
}

func TestServerAggregatesDomainRouteReadiness(t *testing.T) {
	registrar := &readinessRegistrarStub{err: errors.New("domain unavailable")}
	server, err := NewServer(&catalogRuntimeStub{}, registrar)
	if err != nil {
		t.Fatal(err)
	}
	if err := server.Ready(context.Background()); !errors.Is(err, registrar.err) {
		t.Fatalf("domain readiness error = %v", err)
	}
	registrar.err = nil
	if err := server.Ready(context.Background()); err != nil {
		t.Fatalf("healthy domain readiness = %v", err)
	}
}

func TestServerComposesProviderRoutesAndRuntime(t *testing.T) {
	routes, _ := newTestProviderRoutes(t, &providerCatalogStub{
		listResult: providercatalogListResult(),
	}, &providerDiscoveryStub{}, sessionStub{})
	runtime := &catalogRuntimeStub{}
	server, err := NewServer(runtime, routes)
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	server.Register(mux)
	request := authorizedRequest(t, http.MethodGet, providerPath, nil)
	request.Header.Set("Accept", managementapi.JSONMediaType)
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("mounted provider route status = %d, body=%s", response.Code, response.Body.String())
	}
	if err := server.Ready(context.Background()); err != nil {
		t.Fatal(err)
	}
	runtime.readyErr = errors.New("not ready")
	if err := server.Ready(context.Background()); err == nil {
		t.Fatal("runtime readiness failure was ignored")
	}
	runtime.runErr = context.Canceled
	if err := server.Run(context.Background()); !errors.Is(err, context.Canceled) || runtime.runCalls != 1 {
		t.Fatalf("Run() error=%v calls=%d", err, runtime.runCalls)
	}
}

func providercatalogListResult() providercatalog.ListResult {
	return providercatalog.ListResult{
		CatalogRevision: testRevision, Providers: []providercatalog.Definition{catalogDefinition()},
		Categories: []string{"Model APIs"}, PageSize: 50,
	}
}
