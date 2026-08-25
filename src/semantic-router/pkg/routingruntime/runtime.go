package routingruntime

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dispatchauthority"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/outcomefeedback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/runtimecapabilities"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

type catalogLifecycle interface {
	EnsureColdStart(context.Context) error
	Reconcile(context.Context) error
	Run(context.Context) error
	Ready(context.Context) error
}

type usageSupervisorLifecycle interface {
	Reconcile(context.Context) error
	Run(context.Context) error
	Started() <-chan struct{}
	Ready(context.Context) error
	Close() error
}

type routingReplicaLifecycle interface {
	EnsureFleetLease(context.Context) error
	Run(context.Context) error
	Ready() error
	Current(string) (accesspublisher.RuntimePublicationIdentity, bool)
}

type backendDispatchLifecycle interface {
	Attach(backendinvoker.RoutingSnapshotSource, securitykeyring.Symmetric) error
	Start(context.Context) error
	Ready() error
	Close() error
}

// Runtime is the sole owner of process resources and background
// workers. Router generations and HTTP handlers only borrow its dependencies.
type Runtime struct {
	capabilities           runtimecapabilities.RuntimeCapabilities
	database               databaseResource
	redis                  *redis.Client
	redisReady             func(context.Context) error
	redisClose             func() error
	catalog                catalogLifecycle
	catalogClose           func()
	publisherProcessor     accesspublisher.Processor
	publisherWorker        *accesspublisher.Worker
	publicationCoordinator publicationCoordinator
	usageSupervisor        usageSupervisorLifecycle
	routingReplica         routingReplicaLifecycle
	backendDispatch        backendDispatchLifecycle
	responseTerminals      backendinvoker.ResponseTerminalStore
	protocolCodecs         *protocolcodec.Registry
	dispatchCapabilities   *dispatchauthority.Runtime
	access                 *accessruntime.Runtime
	outcomeFeedback        *outcomefeedback.Service
	outcomeProjector       *outcomefeedback.Projector
	outcomeProjection      *outcomefeedback.RedisProjectionStore
	management             ManagementAPI
	keyrings               DeploymentKeyrings
	replicaID              string
	publicNamespaceID      string
	accessKeyPrefix        string
	filePublication        *accesspublisher.RuntimePublicationIdentity
	credentialCloser       io.Closer
	publicationCloser      io.Closer

	mu            sync.RWMutex
	started       bool
	closed        bool
	backgroundErr error
	cancel        context.CancelFunc
	workers       sync.WaitGroup
	closeOnce     sync.Once
	closeErr      error
}

type databaseResource interface {
	PingContext(context.Context) error
	Close() error
}

func (runtime *Runtime) InferenceAccess() *accessruntime.Runtime {
	if runtime == nil {
		return nil
	}
	return runtime.access
}

func (runtime *Runtime) PublicRoutingNamespace() string {
	if runtime == nil {
		return ""
	}
	return runtime.publicNamespaceID
}

// ResponseTerminals returns the semantic response hand-off used by ExtProc. In
// Native Access shares it across replicas; routing without Access uses a local store.
func (runtime *Runtime) ResponseTerminals() backendinvoker.ResponseTerminalReader {
	if runtime == nil {
		return nil
	}
	return runtime.responseTerminals
}

func (runtime *Runtime) ProtocolCodecs() *protocolcodec.Registry {
	if runtime == nil {
		return nil
	}
	return runtime.protocolCodecs
}

func (runtime *Runtime) OutcomeFeedback() *outcomefeedback.Service {
	if runtime == nil {
		return nil
	}
	return runtime.outcomeFeedback
}

func (runtime *Runtime) OutcomeProjection() *outcomefeedback.RedisProjectionStore {
	if runtime == nil {
		return nil
	}
	return runtime.outcomeProjection
}

// DispatchCapabilities returns the process-owned high-level authority. Router
// generations may borrow it but cannot obtain raw signing primitives or close
// its lifecycle.
func (runtime *Runtime) DispatchCapabilities() dispatchauthority.CapabilityRuntime {
	if runtime == nil {
		return nil
	}
	return runtime.dispatchCapabilities
}

// CurrentRoutingPublication exposes only a healthy process-local routing lease
// for request pin verification. It performs no SQL or Redis read.
func (runtime *Runtime) CurrentRoutingPublication(
	namespaceID string,
) (accesspublisher.RuntimePublicationIdentity, bool) {
	if runtime == nil {
		return accesspublisher.RuntimePublicationIdentity{}, false
	}
	runtime.mu.RLock()
	replica := runtime.routingReplica
	filePublication := runtime.filePublication
	fileRouting := runtime.capabilities.FileRouting
	started, closed := runtime.started, runtime.closed
	runtime.mu.RUnlock()
	if !started || closed {
		return accesspublisher.RuntimePublicationIdentity{}, false
	}
	if fileRouting {
		if filePublication == nil || filePublication.NamespaceID != namespaceID {
			return accesspublisher.RuntimePublicationIdentity{}, false
		}
		return *filePublication, true
	}
	if replica == nil {
		return accesspublisher.RuntimePublicationIdentity{}, false
	}
	return replica.Current(namespaceID)
}

// ManagementAPI returns an aggregate readiness wrapper. The HTTP server mounts
// this value but never starts or closes its lifecycle.
func (runtime *Runtime) ManagementAPI() ManagementAPI {
	if runtime == nil || runtime.management == nil {
		return nil
	}
	return runtimeManagementAPI{runtime: runtime, delegate: runtime.management}
}

func (runtime *Runtime) Start(ctx context.Context) error {
	fileRouting, err := runtime.validateStart()
	if err != nil {
		return err
	}
	if fileRouting {
		return runtime.startFileRuntime(ctx)
	}
	if reconcileErr := runtime.reconcileDurableRuntime(ctx); reconcileErr != nil {
		return reconcileErr
	}
	workerContext, cancel, err := runtime.startDurableWorkers(ctx)
	if err != nil {
		return err
	}
	if waitErr := runtime.waitForWorkers(ctx); waitErr != nil {
		cancel()
		runtime.workers.Wait()
		return waitErr
	}
	_ = workerContext
	return nil
}

func (runtime *Runtime) validateStart() (bool, error) {
	if runtime == nil {
		return false, errors.New("process runtime is nil")
	}
	runtime.mu.Lock()
	defer runtime.mu.Unlock()
	if runtime.closed {
		return false, errors.New("process runtime is closed")
	}
	if runtime.started {
		return false, errors.New("process runtime is already started")
	}
	if runtime.capabilities.FileRouting {
		if runtime.backendDispatch == nil || runtime.dispatchCapabilities == nil || runtime.filePublication == nil {
			return false, errors.New("file-authority runtime dependencies are incomplete")
		}
		return true, nil
	}
	if runtime.catalog == nil || runtime.database == nil ||
		runtime.publisherProcessor == nil || runtime.publisherWorker == nil ||
		runtime.routingReplica == nil || runtime.backendDispatch == nil {
		return false, errors.New("durable routing runtime dependencies are incomplete")
	}
	if runtime.capabilities.DistributedState && runtime.redisReady == nil {
		return false, errors.New("distributed runtime store dependency is incomplete")
	}
	if runtime.capabilities.ManagementAPI != (runtime.management != nil) {
		return false, errors.New("management API composition is inconsistent")
	}
	if runtime.capabilities.NativeAccess != (runtime.access != nil) ||
		runtime.capabilities.NativeAccess != (runtime.outcomeFeedback != nil) ||
		runtime.capabilities.NativeAccess != (runtime.outcomeProjector != nil) ||
		runtime.capabilities.NativeAccess != (runtime.outcomeProjection != nil) {
		return false, errors.New("native access and outcome runtime composition is inconsistent")
	}
	return false, nil
}

func (runtime *Runtime) startFileRuntime(ctx context.Context) error {
	workerContext, cancel := context.WithCancel(ctx)
	if err := runtime.backendDispatch.Start(workerContext); err != nil {
		cancel()
		return fmt.Errorf("start backend dispatch: %w", err)
	}
	runtime.mu.Lock()
	defer runtime.mu.Unlock()
	if runtime.closed {
		cancel()
		return errors.New("file-authority runtime closed during startup")
	}
	runtime.cancel = cancel
	runtime.started = true
	return nil
}

func (runtime *Runtime) reconcileDurableRuntime(ctx context.Context) error {
	// The catalog and publication passes are synchronous so readiness never
	// precedes the active immutable routing and usage state.
	if err := runtime.catalog.EnsureColdStart(ctx); err != nil {
		return fmt.Errorf("initialize Provider Catalog: %w", err)
	}
	if err := runtime.catalog.Reconcile(ctx); err != nil {
		return fmt.Errorf("reconcile active Provider Catalog: %w", err)
	}
	if err := runtime.routingReplica.EnsureFleetLease(ctx); err != nil {
		return fmt.Errorf("establish routing fleet lease: %w", err)
	}
	if _, err := runtime.publisherProcessor.ProcessOnce(ctx); err != nil {
		return fmt.Errorf("reconcile routing publication: %w", err)
	}
	if runtime.capabilities.NativeAccess && runtime.usageSupervisor == nil {
		return errors.New("native access usage supervisor is unavailable")
	}
	if runtime.usageSupervisor != nil {
		if err := runtime.usageSupervisor.Reconcile(ctx); err != nil {
			return fmt.Errorf("reconcile usage ledger: %w", err)
		}
	}
	if runtime.outcomeProjector != nil {
		if _, err := runtime.outcomeProjector.ProcessOnce(ctx); err != nil {
			return fmt.Errorf("reconcile outcome learning projection: %w", err)
		}
	}
	return nil
}

func (runtime *Runtime) startDurableWorkers(
	ctx context.Context,
) (context.Context, context.CancelFunc, error) {
	workerContext, cancel := context.WithCancel(ctx)
	if err := runtime.backendDispatch.Start(workerContext); err != nil {
		cancel()
		return nil, nil, fmt.Errorf("start backend dispatch: %w", err)
	}
	runtime.mu.Lock()
	if runtime.closed {
		runtime.mu.Unlock()
		cancel()
		return nil, nil, errors.New("process runtime closed during startup")
	}
	runtime.cancel = cancel
	runtime.started = true
	workerCount := 3
	if runtime.management != nil {
		workerCount++
	}
	if runtime.usageSupervisor != nil {
		workerCount++
	}
	if runtime.outcomeProjector != nil {
		workerCount++
	}
	runtime.workers.Add(workerCount)
	runtime.mu.Unlock()

	go runtime.runWorker(workerContext, "provider Catalog replica stopped", runtime.catalog.Run)
	go runtime.runWorker(workerContext, "routing publication replica stopped", runtime.routingReplica.Run)
	go runtime.runWorker(workerContext, "routing publication worker stopped", runtime.publisherWorker.Run)
	if runtime.usageSupervisor != nil {
		go runtime.runWorker(workerContext, "usage ledger supervisor stopped", runtime.usageSupervisor.Run)
	}
	if runtime.outcomeProjector != nil {
		go runtime.runWorker(workerContext, "outcome learning projector stopped", runtime.outcomeProjector.Run)
	}
	if runtime.management != nil {
		go runtime.runWorker(workerContext, "Management API worker stopped", runtime.management.Run)
	}
	return workerContext, cancel, nil
}

func (runtime *Runtime) runWorker(
	ctx context.Context, stoppedMessage string, run func(context.Context) error,
) {
	defer runtime.workers.Done()
	err := run(ctx)
	if err == nil || errors.Is(err, context.Canceled) || ctx.Err() != nil {
		return
	}
	runtime.mu.Lock()
	runtime.backgroundErr = errors.New(stoppedMessage)
	runtime.mu.Unlock()
}

func (runtime *Runtime) waitForWorkers(ctx context.Context) error {
	select {
	case <-runtime.publisherWorker.Started():
	case <-ctx.Done():
		return ctx.Err()
	}
	if runtime.usageSupervisor == nil {
		return nil
	}
	select {
	case <-runtime.usageSupervisor.Started():
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (runtime *Runtime) Ready(ctx context.Context) error {
	if runtime == nil {
		return errors.New("process runtime is unavailable")
	}
	runtime.mu.RLock()
	started, closed, backgroundErr := runtime.started, runtime.closed, runtime.backgroundErr
	fileRouting := runtime.capabilities.FileRouting
	runtime.mu.RUnlock()
	if closed {
		return errors.New("process runtime is closed")
	}
	if !started {
		return errors.New("process runtime has not started")
	}
	// The decoder finalizer and ExtProc reader are deliberately the same
	// rendezvous. Starting either half without the other would
	// make successful responses unaccountable.
	if runtime.responseTerminals == nil || runtime.protocolCodecs == nil {
		return errors.New("neutral response runtime is unavailable")
	}
	if fileRouting {
		if runtime.backendDispatch == nil {
			return errors.New("file-authority backend dispatch is unavailable")
		}
		return runtime.backendDispatch.Ready()
	}
	if backgroundErr != nil {
		return backgroundErr
	}
	if runtime.catalog == nil || runtime.database == nil || runtime.publisherWorker == nil ||
		runtime.routingReplica == nil || runtime.backendDispatch == nil {
		return errors.New("durable routing runtime dependencies are unavailable")
	}
	if err := runtime.catalog.Ready(ctx); err != nil {
		return err
	}
	if err := runtime.database.PingContext(ctx); err != nil {
		return errors.New("durable PostgreSQL is unavailable")
	}
	if runtime.capabilities.DistributedState {
		if runtime.redisReady == nil {
			return errors.New("distributed runtime store is unavailable")
		}
		if err := runtime.redisReady(ctx); err != nil {
			return errors.New("distributed runtime store is unavailable")
		}
	}
	if err := runtime.publisherWorker.Ready(ctx); err != nil {
		return err
	}
	if runtime.capabilities.NativeAccess && runtime.usageSupervisor == nil {
		return errors.New("native access usage supervisor is unavailable")
	}
	if runtime.usageSupervisor != nil {
		if err := runtime.usageSupervisor.Ready(ctx); err != nil {
			return err
		}
	}
	if err := runtime.routingReplica.Ready(); err != nil {
		return err
	}
	if err := runtime.backendDispatch.Ready(); err != nil {
		return err
	}
	return nil
}

func (runtime *Runtime) Close() error {
	if runtime == nil {
		return nil
	}
	runtime.closeOnce.Do(func() {
		runtime.mu.Lock()
		runtime.closed = true
		cancel := runtime.cancel
		runtime.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		runtime.workers.Wait()

		var closeErrors []error
		if runtime.backendDispatch != nil {
			closeErrors = append(closeErrors, runtime.backendDispatch.Close())
		}
		if runtime.dispatchCapabilities != nil {
			closeErrors = append(closeErrors, runtime.dispatchCapabilities.Close())
		}
		if runtime.credentialCloser != nil {
			closeErrors = append(closeErrors, runtime.credentialCloser.Close())
		}
		if runtime.publicationCloser != nil {
			closeErrors = append(closeErrors, runtime.publicationCloser.Close())
		}
		if runtime.usageSupervisor != nil {
			closeErrors = append(closeErrors, runtime.usageSupervisor.Close())
		}
		if closer, ok := runtime.management.(io.Closer); ok {
			closeErrors = append(closeErrors, closer.Close())
		}
		if runtime.catalogClose != nil {
			runtime.catalogClose()
		}
		if runtime.redisClose != nil {
			closeErrors = append(closeErrors, runtime.redisClose())
		}
		if runtime.database != nil {
			closeErrors = append(closeErrors, runtime.database.Close())
		}
		runtime.keyrings.zero()
		runtime.closeErr = errors.Join(closeErrors...)
	})
	return runtime.closeErr
}

type runtimeManagementAPI struct {
	runtime  *Runtime
	delegate ManagementAPI
}

func (api runtimeManagementAPI) Register(mux *http.ServeMux) {
	if api.runtime == nil || api.delegate == nil || mux == nil {
		panic("Management API composition is unavailable")
	}
	api.delegate.Register(mux)
}

func (api runtimeManagementAPI) Ready(ctx context.Context) error {
	if err := api.runtime.Ready(ctx); err != nil {
		return err
	}
	if err := api.delegate.Ready(ctx); err != nil {
		return fmt.Errorf("management API is not ready: %w", err)
	}
	return nil
}

func (api runtimeManagementAPI) Run(ctx context.Context) error {
	if api.runtime == nil || api.delegate == nil {
		return errors.New("management API is unavailable")
	}
	return api.delegate.Run(ctx)
}
