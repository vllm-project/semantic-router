package managedruntime

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func TestStandaloneCompositionOwnsDispatchWithoutManagedResources(t *testing.T) {
	cfg := standaloneRuntimeConfig(t)
	_, err := New(context.Background(), &cfg, Options{ManagementFactory: ManagementFactoryFunc(
		func(context.Context, ManagementDependencies) (ManagedAPI, error) { return nil, nil },
	)})
	if err == nil || err.Error() != "standalone mode rejects a managed Management factory" {
		t.Fatalf("New() error = %v", err)
	}
	runtime, err := New(context.Background(), &cfg, Options{})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if runtime.ManagedAPI() != nil || runtime.InferenceAccess() != nil {
		t.Fatal("standalone composition exposed managed dependencies")
	}
	if _, ok := runtime.responseTerminals.(*backendinvoker.LocalResponseTerminalStore); !ok {
		t.Fatalf("standalone response terminal store = %T, want process-local", runtime.responseTerminals)
	}
	if err := runtime.Start(context.Background()); err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	if err := runtime.Ready(context.Background()); err != nil {
		t.Fatalf("Ready() error = %v", err)
	}
	if err := runtime.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func TestManagedCompositionInjectsSharedResponseTerminalStore(t *testing.T) {
	client := redis.NewClient(&redis.Options{Addr: "unused.invalid:6379"})
	t.Cleanup(func() { _ = client.Close() })
	store, err := composeManagedResponseTerminalStore(client, "managed:{prefix}:accepted")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := store.(*backendinvoker.RedisResponseTerminalStore); !ok {
		t.Fatalf("managed response terminal store = %T, want shared Redis store", store)
	}
}

func standaloneRuntimeConfig(t *testing.T) config.RouterConfig {
	t.Helper()
	policyPath := filepath.Join(t.TempDir(), "backend-egress.yaml")
	if err := os.WriteFile(policyPath, []byte("version: v1\nschemes: [http]\nhosts:\n  - host: models.example\n    ports: [80]\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	_ = listener.Close()
	snapshot, err := routingsnapshot.Compile(routingsnapshot.Bundle{
		NamespaceID: "7fc73a6d-2081-55c2-8828-4d74a43840c1", Revision: 1,
		Models: []routingsnapshot.Model{{
			ID: "mdl", Revision: 1,
			CatalogRevision: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			Name:            "model", Execution: routingsnapshot.ModelExecution{RequestTimeout: "30s", StreamTimeout: "2m"},
			Backends: []routingsnapshot.Backend{{
				ID: "backend", ProviderID: "provider", WireFormat: "openai.chat.v1",
				Origin: "http://models.example", ProviderModelID: "model",
				Connection: routingsnapshot.BackendConnection{Path: "/v1/chat/completions"}, Weight: "1",
			}},
		}},
		Recipes: []routingsnapshot.Recipe{{
			ID: "recipe", Revision: 1, Name: "recipe",
			Decisions: []routingsnapshot.Decision{{ID: "decision", Name: "Decision", DispatchCardinality: routingsnapshot.DispatchCardinalitySingle}},
			Document:  json.RawMessage(`{"signals":{},"projections":{},"decisions":[{"name":"Decision","rules":{}}]}`),
		}},
		Entrypoints: []routingsnapshot.Entrypoint{{
			ID: "entrypoint", Revision: 1, Name: "entrypoint", Aliases: []string{"vllm-sr/test"},
			Rules: []routingsnapshot.EntrypointRule{{
				ID: "rule", Name: "default", RecipeID: "recipe", RecipeRevision: 1,
				Assignments: map[string]routingsnapshot.AssignmentSet{
					"decision": {Models: []routingsnapshot.Assignment{{ModelID: "mdl", ModelRevision: 1, Weight: "1"}}},
				},
			}},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	cfg := config.DefaultGlobalConfig()
	cfg.BackendEgress.PolicyFile = policyPath
	cfg.BackendDispatch.BindAddress = "127.0.0.1"
	cfg.BackendDispatch.Port = port
	cfg.RoutingSnapshot = snapshot
	return cfg
}

func TestManagedAPIRoutesRemainMountedBeforeRuntimeReadiness(t *testing.T) {
	runtime := &Runtime{
		mode:       config.ControlPlaneModeManaged,
		management: providerBootstrapManagementStub{},
	}
	api := runtime.ManagedAPI()
	if api == nil {
		t.Fatal("managed API is unavailable")
	}
	if err := api.Ready(context.Background()); err == nil {
		t.Fatal("unstarted managed runtime reported ready")
	}
	mux := http.NewServeMux()
	api.Register(mux)
	for _, path := range []string{
		"/management/v1/provider-catalog:bootstrap",
		"/management/v1/provider-catalog:activate",
	} {
		request := httptest.NewRequest(http.MethodPost, path, nil)
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, request)
		if response.Code != http.StatusNoContent {
			t.Fatalf("unready runtime route %s status = %d", path, response.Code)
		}
	}
}

func TestManagedLifecycleReconcilesBeforeRunAndClosesInReverseOrder(t *testing.T) {
	var orderMu sync.Mutex
	order := make([]string, 0, 8)
	record := func(value string) {
		orderMu.Lock()
		order = append(order, value)
		orderMu.Unlock()
	}
	catalog := &catalogLifecycleStub{record: record, running: make(chan struct{})}
	publisher := &publisherProcessorStub{record: record, running: make(chan struct{})}
	usage := newUsageSupervisorStub(record)
	routing := &routingReplicaStub{record: record, running: make(chan struct{})}
	management := &managementAPIStub{record: record}
	database := &databaseStub{record: record}
	runtime := &Runtime{
		mode: config.ControlPlaneModeManaged, database: database,
		redisReady: func(context.Context) error { record("redis-ready"); return nil },
		redisClose: func() error { record("redis-close"); return nil },
		catalog:    catalog, catalogClose: func() { record("catalog-close") },
		publisherProcessor: publisher,
		publisherWorker:    newPublisherWorker(t, publisher),
		usageSupervisor:    usage,
		routingReplica:     routing,
		backendDispatch:    &backendDispatchLifecycleStub{record: record},
		responseTerminals:  backendinvoker.NewLocalResponseTerminalStore(),
		protocolCodecs:     protocolcodec.NewBuiltinRegistry(),
		management:         management,
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := runtime.Start(ctx); err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	<-catalog.running
	<-publisher.running
	<-usage.running
	<-routing.running
	if err := runtime.ManagedAPI().Ready(context.Background()); err != nil {
		t.Fatalf("ManagedAPI.Ready() error = %v", err)
	}
	if err := runtime.Start(ctx); err == nil {
		t.Fatal("second Start() unexpectedly succeeded")
	}
	if err := runtime.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if err := runtime.Close(); err != nil {
		t.Fatalf("second Close() error = %v", err)
	}

	orderMu.Lock()
	got := append([]string(nil), order...)
	orderMu.Unlock()
	wantPrefix := []string{"catalog-cold-start", "catalog-reconcile", "routing-fleet-lease", "publisher-reconcile", "usage-reconcile", "dispatch-start"}
	if len(got) < len(wantPrefix) || !reflect.DeepEqual(got[:len(wantPrefix)], wantPrefix) {
		t.Fatalf("synchronous lifecycle prefix = %v, want %v", got, wantPrefix)
	}
	wantSuffix := []string{"dispatch-close", "usage-close", "management-close", "catalog-close", "redis-close", "database-close"}
	if !reflect.DeepEqual(got[len(got)-len(wantSuffix):], wantSuffix) {
		t.Fatalf("lifecycle suffix = %v, want %v", got, wantSuffix)
	}
	for _, event := range []string{"catalog-run", "publisher-run", "usage-run", "routing-run", "management-run", "catalog-stopped", "publisher-stopped", "usage-stopped", "routing-stopped", "management-stopped"} {
		if !containsEvent(got, event) {
			t.Fatalf("lifecycle = %v, missing %q", got, event)
		}
	}
}

func TestManagedLifecycleFailsClosedOnReconcileAndBackgroundFailure(t *testing.T) {
	publisher := &publisherProcessorStub{}
	coldStartFailure := &Runtime{
		mode: config.ControlPlaneModeManaged, database: &databaseStub{},
		redisReady: func(context.Context) error { return nil },
		catalog: &catalogLifecycleStub{
			coldStartErr: errors.New("catalog bootstrap unavailable"),
		},
		publisherProcessor: publisher,
		publisherWorker:    newPublisherWorker(t, publisher),
		usageSupervisor:    newUsageSupervisorStub(nil),
		routingReplica:     &routingReplicaStub{},
		backendDispatch:    &backendDispatchLifecycleStub{},
		management:         &managementAPIStub{},
	}
	if err := coldStartFailure.Start(context.Background()); err == nil ||
		err.Error() != "initialize Provider Catalog: catalog bootstrap unavailable" {
		t.Fatalf("cold-start Start() error = %v", err)
	}
	reconcileFailure := &Runtime{
		mode: config.ControlPlaneModeManaged, database: &databaseStub{},
		redisReady:         func(context.Context) error { return nil },
		catalog:            &catalogLifecycleStub{reconcileErr: errors.New("catalog unavailable")},
		publisherProcessor: publisher,
		publisherWorker:    newPublisherWorker(t, publisher),
		usageSupervisor:    newUsageSupervisorStub(nil),
		routingReplica:     &routingReplicaStub{},
		backendDispatch:    &backendDispatchLifecycleStub{},
		management:         &managementAPIStub{},
	}
	if err := reconcileFailure.Start(context.Background()); err == nil {
		t.Fatal("Start() unexpectedly accepted failed reconciliation")
	}
	managementFailure := &Runtime{
		mode: config.ControlPlaneModeManaged, database: &databaseStub{},
		redisReady:         func(context.Context) error { return nil },
		catalog:            &catalogLifecycleStub{},
		publisherProcessor: publisher,
		publisherWorker:    newPublisherWorker(t, publisher),
		usageSupervisor:    newUsageSupervisorStub(nil),
		routingReplica:     &routingReplicaStub{},
		backendDispatch:    &backendDispatchLifecycleStub{},
		management:         &managementAPIStub{readyErr: errors.New("retained HMAC version missing")},
	}
	managementContext, cancelManagement := context.WithCancel(context.Background())
	if err := managementFailure.Start(managementContext); err != nil {
		t.Fatalf("Start() blocked explicit bootstrap on Management readiness: %v", err)
	}
	if err := managementFailure.ManagedAPI().Ready(context.Background()); err == nil {
		t.Fatal("ManagedAPI.Ready() accepted failed Management readiness")
	}
	cancelManagement()
	if err := managementFailure.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	catalog := &catalogLifecycleStub{
		runErr: errors.New("lease worker failed"), running: make(chan struct{}), stopped: make(chan struct{}),
	}
	publisher = &publisherProcessorStub{running: make(chan struct{})}
	runtime := &Runtime{
		mode: config.ControlPlaneModeManaged, database: &databaseStub{},
		redisReady: func(context.Context) error { return nil }, redisClose: func() error { return nil },
		catalog:            catalog,
		publisherProcessor: publisher,
		publisherWorker:    newPublisherWorker(t, publisher),
		usageSupervisor:    newUsageSupervisorStub(nil),
		routingReplica:     &routingReplicaStub{},
		backendDispatch:    &backendDispatchLifecycleStub{},
		management:         &managementAPIStub{},
	}
	if err := runtime.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	<-catalog.running
	<-catalog.stopped
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if err := runtime.Ready(context.Background()); err != nil {
			_ = runtime.Close()
			return
		}
		time.Sleep(time.Millisecond)
	}
	_ = runtime.Close()
	t.Fatal("Ready() did not observe background failure")
}

func TestManagedLifecycleFailsClosedOnPublisherReconciliation(t *testing.T) {
	publisher := &publisherProcessorStub{reconcileErr: errors.New("publication store unavailable")}
	runtime := &Runtime{
		mode: config.ControlPlaneModeManaged, database: &databaseStub{},
		redisReady:         func(context.Context) error { return nil },
		catalog:            &catalogLifecycleStub{},
		publisherProcessor: publisher,
		publisherWorker:    newPublisherWorker(t, publisher),
		usageSupervisor:    newUsageSupervisorStub(nil),
		routingReplica:     &routingReplicaStub{},
		backendDispatch:    &backendDispatchLifecycleStub{},
		management:         &managementAPIStub{},
	}
	if err := runtime.Start(context.Background()); err == nil || err.Error() != "reconcile access publication: publication store unavailable" {
		t.Fatalf("Start() error = %v", err)
	}
}

func TestManagedLifecycleFailsClosedBeforePublicationWithoutFleetLease(t *testing.T) {
	publisher := &publisherProcessorStub{}
	runtime := &Runtime{
		mode: config.ControlPlaneModeManaged, database: &databaseStub{},
		redisReady:         func(context.Context) error { return nil },
		catalog:            &catalogLifecycleStub{},
		publisherProcessor: publisher,
		publisherWorker:    newPublisherWorker(t, publisher),
		usageSupervisor:    newUsageSupervisorStub(nil),
		routingReplica:     &routingReplicaStub{fleetErr: errors.New("membership store unavailable")},
		backendDispatch:    &backendDispatchLifecycleStub{},
		management:         &managementAPIStub{},
	}
	if err := runtime.Start(context.Background()); err == nil ||
		err.Error() != "establish routing fleet lease: membership store unavailable" {
		t.Fatalf("Start() error = %v", err)
	}
	publisher.mu.Lock()
	calls := publisher.calls
	publisher.mu.Unlock()
	if calls != 0 {
		t.Fatalf("publisher reconciliation calls = %d, want 0", calls)
	}
}

func TestManagedLifecycleFailsClosedOnUsageReconciliation(t *testing.T) {
	publisher := &publisherProcessorStub{}
	usage := newUsageSupervisorStub(nil)
	usage.reconcileErr = errors.New("usage store unavailable")
	runtime := &Runtime{
		mode: config.ControlPlaneModeManaged, database: &databaseStub{},
		redisReady:         func(context.Context) error { return nil },
		catalog:            &catalogLifecycleStub{},
		publisherProcessor: publisher,
		publisherWorker:    newPublisherWorker(t, publisher),
		usageSupervisor:    usage,
		routingReplica:     &routingReplicaStub{},
		backendDispatch:    &backendDispatchLifecycleStub{},
		management:         &managementAPIStub{},
	}
	if err := runtime.Start(context.Background()); err == nil || err.Error() != "reconcile usage ledger: usage store unavailable" {
		t.Fatalf("Start() error = %v", err)
	}
}

func newPublisherWorker(t *testing.T, processor accesspublisher.Processor) *accesspublisher.Worker {
	t.Helper()
	worker, err := accesspublisher.NewWorker(accesspublisher.WorkerOptions{
		Processor:  processor,
		IdleDelay:  time.Millisecond,
		MinBackoff: time.Millisecond,
		MaxBackoff: 2 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	return worker
}

func containsEvent(events []string, wanted string) bool {
	for _, event := range events {
		if event == wanted {
			return true
		}
	}
	return false
}

type publisherProcessorStub struct {
	record       func(string)
	running      chan struct{}
	reconcileErr error

	mu      sync.Mutex
	calls   int
	runOnce sync.Once
}

func (stub *publisherProcessorStub) ProcessOnce(ctx context.Context) (accesspublisher.ProcessResult, error) {
	stub.mu.Lock()
	stub.calls++
	call := stub.calls
	stub.mu.Unlock()
	if call == 1 {
		stub.append("publisher-reconcile")
		return accesspublisher.ProcessResult{Disposition: accesspublisher.ProcessNoWork}, stub.reconcileErr
	}
	stub.runOnce.Do(func() {
		stub.append("publisher-run")
		if stub.running != nil {
			close(stub.running)
		}
	})
	<-ctx.Done()
	stub.append("publisher-stopped")
	return accesspublisher.ProcessResult{}, ctx.Err()
}

func (stub *publisherProcessorStub) append(value string) {
	if stub.record != nil {
		stub.record(value)
	}
}

type catalogLifecycleStub struct {
	record       func(string)
	running      chan struct{}
	stopped      chan struct{}
	coldStartErr error
	reconcileErr error
	runErr       error
	readyErr     error
}

func (stub *catalogLifecycleStub) EnsureColdStart(context.Context) error {
	stub.append("catalog-cold-start")
	return stub.coldStartErr
}

func (stub *catalogLifecycleStub) Reconcile(context.Context) error {
	stub.append("catalog-reconcile")
	return stub.reconcileErr
}

func (stub *catalogLifecycleStub) Run(ctx context.Context) error {
	if stub.stopped != nil {
		defer close(stub.stopped)
	}
	stub.append("catalog-run")
	if stub.running != nil {
		close(stub.running)
	}
	if stub.runErr != nil {
		return stub.runErr
	}
	<-ctx.Done()
	stub.append("catalog-stopped")
	return ctx.Err()
}

func (stub *catalogLifecycleStub) Ready(context.Context) error {
	stub.append("catalog-ready")
	return stub.readyErr
}

func (stub *catalogLifecycleStub) append(value string) {
	if stub.record != nil {
		stub.record(value)
	}
}

type managementAPIStub struct {
	record   func(string)
	readyErr error
}

type providerBootstrapManagementStub struct{}

func (providerBootstrapManagementStub) Register(mux *http.ServeMux) {
	for _, path := range []string{
		"POST /management/v1/provider-catalog:bootstrap",
		"POST /management/v1/provider-catalog:activate",
	} {
		mux.HandleFunc(path, func(response http.ResponseWriter, _ *http.Request) {
			response.WriteHeader(http.StatusNoContent)
		})
	}
}

func (providerBootstrapManagementStub) Ready(context.Context) error { return nil }
func (providerBootstrapManagementStub) Run(ctx context.Context) error {
	<-ctx.Done()
	return ctx.Err()
}

type routingReplicaStub struct {
	record   func(string)
	running  chan struct{}
	fleetErr error
	runErr   error
	readyErr error
	runOnce  sync.Once
}

type backendDispatchLifecycleStub struct {
	record   func(string)
	startErr error
	readyErr error
	closeErr error
}

func (stub *backendDispatchLifecycleStub) Attach(
	backendinvoker.RoutingSnapshotSource,
	securitykeyring.Symmetric,
) error {
	return nil
}

func (stub *backendDispatchLifecycleStub) Start(context.Context) error {
	if stub.record != nil {
		stub.record("dispatch-start")
	}
	return stub.startErr
}

func (stub *backendDispatchLifecycleStub) Ready() error { return stub.readyErr }

func (stub *backendDispatchLifecycleStub) Close() error {
	if stub.record != nil {
		stub.record("dispatch-close")
	}
	return stub.closeErr
}

func (stub *routingReplicaStub) EnsureFleetLease(context.Context) error {
	if stub.record != nil {
		stub.record("routing-fleet-lease")
	}
	return stub.fleetErr
}

func (stub *routingReplicaStub) Run(ctx context.Context) error {
	stub.runOnce.Do(func() {
		if stub.record != nil {
			stub.record("routing-run")
		}
		if stub.running != nil {
			close(stub.running)
		}
	})
	if stub.runErr != nil {
		return stub.runErr
	}
	<-ctx.Done()
	if stub.record != nil {
		stub.record("routing-stopped")
	}
	return ctx.Err()
}

func (stub *routingReplicaStub) Ready() error {
	if stub.readyErr != nil {
		return stub.readyErr
	}
	if stub.running == nil {
		return nil
	}
	select {
	case <-stub.running:
		return nil
	default:
		return errors.New("routing replica is starting")
	}
}

func (stub *routingReplicaStub) Current(string) (accesspublisher.RuntimePublicationIdentity, bool) {
	return accesspublisher.RuntimePublicationIdentity{}, false
}

type usageSupervisorStub struct {
	record       func(string)
	running      chan struct{}
	started      chan struct{}
	reconcileErr error
	runErr       error
	readyErr     error
	startOnce    sync.Once
}

func newUsageSupervisorStub(record func(string)) *usageSupervisorStub {
	return &usageSupervisorStub{record: record, running: make(chan struct{}), started: make(chan struct{})}
}

func (stub *usageSupervisorStub) Reconcile(context.Context) error {
	stub.append("usage-reconcile")
	return stub.reconcileErr
}

func (stub *usageSupervisorStub) Run(ctx context.Context) error {
	stub.append("usage-run")
	stub.startOnce.Do(func() {
		close(stub.running)
		close(stub.started)
	})
	if stub.runErr != nil {
		return stub.runErr
	}
	<-ctx.Done()
	stub.append("usage-stopped")
	return ctx.Err()
}

func (stub *usageSupervisorStub) Started() <-chan struct{}    { return stub.started }
func (stub *usageSupervisorStub) Ready(context.Context) error { return stub.readyErr }
func (stub *usageSupervisorStub) Close() error {
	stub.append("usage-close")
	return nil
}

func (stub *usageSupervisorStub) append(value string) {
	if stub.record != nil {
		stub.record(value)
	}
}

func (*managementAPIStub) Register(*http.ServeMux) {}
func (stub *managementAPIStub) Ready(context.Context) error {
	if stub.record != nil {
		stub.record("management-ready")
	}
	return stub.readyErr
}

func (stub *managementAPIStub) Run(ctx context.Context) error {
	if stub.record != nil {
		stub.record("management-run")
	}
	<-ctx.Done()
	if stub.record != nil {
		stub.record("management-stopped")
	}
	return ctx.Err()
}

func (stub *managementAPIStub) Close() error {
	if stub.record != nil {
		stub.record("management-close")
	}
	return nil
}

type databaseStub struct{ record func(string) }

func (stub *databaseStub) PingContext(context.Context) error {
	if stub.record != nil {
		stub.record("database-ready")
	}
	return nil
}

func (stub *databaseStub) Close() error {
	if stub.record != nil {
		stub.record("database-close")
	}
	return nil
}
