package routingruntime

import (
	"context"
	"crypto/sha256"
	"errors"
	"net/http"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/outcomefeedback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

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

type emptyOutcomeProjectionRepository struct{}

func (emptyOutcomeProjectionRepository) PendingNamespaces(context.Context, int) ([]string, error) {
	return nil, nil
}

func (emptyOutcomeProjectionRepository) Build(context.Context, string) (outcomefeedback.Projection, error) {
	return outcomefeedback.Projection{}, errors.New("unexpected projection build")
}

func (emptyOutcomeProjectionRepository) Stage(
	context.Context,
	outcomefeedback.Projection,
	[]byte,
	[sha256.Size]byte,
) error {
	return errors.New("unexpected projection stage")
}

func (emptyOutcomeProjectionRepository) MarkApplied(
	context.Context,
	string,
	int64,
	[sha256.Size]byte,
) error {
	return errors.New("unexpected projection apply")
}

type emptyOutcomeProjectionPublisher struct{}

func (emptyOutcomeProjectionPublisher) Publish(
	context.Context,
	outcomefeedback.Projection,
	[]byte,
	[sha256.Size]byte,
) error {
	return errors.New("unexpected projection publication")
}
