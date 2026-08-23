package managedruntime

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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dispatchauthority"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/outcomefeedback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
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

// Runtime is the sole owner of managed process resources and background
// workers. Router generations and HTTP handlers only borrow its dependencies.
type Runtime struct {
	mode                  string
	database              databaseResource
	redis                 *redis.Client
	redisReady            func(context.Context) error
	redisClose            func() error
	catalog               catalogLifecycle
	catalogClose          func()
	publisherProcessor    accesspublisher.Processor
	publisherWorker       *accesspublisher.Worker
	usageSupervisor       usageSupervisorLifecycle
	routingReplica        routingReplicaLifecycle
	backendDispatch       backendDispatchLifecycle
	responseTerminals     backendinvoker.ResponseTerminalStore
	protocolCodecs        *protocolcodec.Registry
	dispatchCapabilities  *dispatchauthority.Runtime
	access                *accessruntime.Runtime
	outcomeFeedback       *outcomefeedback.Service
	outcomeProjector      *outcomefeedback.Projector
	outcomeProjection     *outcomefeedback.RedisProjectionStore
	accessEnabled         bool
	management            ManagedAPI
	keyrings              DeploymentKeyrings
	replicaID             string
	accessKeyPrefix       string
	standalonePublication *accesspublisher.RuntimePublicationIdentity
	credentialCloser      io.Closer

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

func (runtime *Runtime) Mode() string {
	if runtime == nil {
		return ""
	}
	return runtime.mode
}

func (runtime *Runtime) InferenceAccess() *accessruntime.Runtime {
	if runtime == nil {
		return nil
	}
	return runtime.access
}

// ResponseTerminals returns the semantic response hand-off used by ExtProc. In
// managed mode it is shared across replicas; standalone uses a local store.
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
	standalone := runtime.standalonePublication
	mode := runtime.mode
	started, closed := runtime.started, runtime.closed
	runtime.mu.RUnlock()
	if !started || closed {
		return accesspublisher.RuntimePublicationIdentity{}, false
	}
	if mode == config.ControlPlaneModeStandalone {
		if standalone == nil || standalone.NamespaceID != namespaceID {
			return accesspublisher.RuntimePublicationIdentity{}, false
		}
		return *standalone, true
	}
	if replica == nil {
		return accesspublisher.RuntimePublicationIdentity{}, false
	}
	return replica.Current(namespaceID)
}

// ManagedAPI returns an aggregate readiness wrapper. The HTTP server mounts
// this value but never starts or closes its lifecycle.
func (runtime *Runtime) ManagedAPI() ManagedAPI {
	if runtime == nil || runtime.management == nil {
		return nil
	}
	return runtimeManagedAPI{runtime: runtime, delegate: runtime.management}
}

func (runtime *Runtime) Start(ctx context.Context) error {
	if runtime == nil {
		return errors.New("managed runtime is nil")
	}
	runtime.mu.Lock()
	if runtime.closed {
		runtime.mu.Unlock()
		return errors.New("managed runtime is closed")
	}
	if runtime.started {
		runtime.mu.Unlock()
		return errors.New("managed runtime is already started")
	}
	if runtime.mode == config.ControlPlaneModeStandalone {
		if runtime.backendDispatch == nil || runtime.dispatchCapabilities == nil || runtime.standalonePublication == nil {
			runtime.mu.Unlock()
			return errors.New("standalone runtime dependencies are incomplete")
		}
		runtime.mu.Unlock()
		workerContext, cancel := context.WithCancel(ctx)
		if err := runtime.backendDispatch.Start(workerContext); err != nil {
			cancel()
			return fmt.Errorf("start backend dispatch: %w", err)
		}
		runtime.mu.Lock()
		if runtime.closed {
			runtime.mu.Unlock()
			cancel()
			return errors.New("standalone runtime closed during startup")
		}
		runtime.cancel = cancel
		runtime.started = true
		runtime.mu.Unlock()
		return nil
	}
	if runtime.catalog == nil || runtime.management == nil || runtime.database == nil || runtime.redisReady == nil ||
		runtime.publisherProcessor == nil || runtime.publisherWorker == nil ||
		runtime.routingReplica == nil || runtime.backendDispatch == nil {
		runtime.mu.Unlock()
		return errors.New("managed runtime dependencies are incomplete")
	}
	if runtime.accessEnabled != (runtime.access != nil) ||
		runtime.accessEnabled != (runtime.outcomeFeedback != nil) ||
		runtime.accessEnabled != (runtime.outcomeProjector != nil) ||
		runtime.accessEnabled != (runtime.outcomeProjection != nil) {
		runtime.mu.Unlock()
		return errors.New("managed access and outcome runtime composition is inconsistent")
	}
	runtime.mu.Unlock()

	// The application-installed catalog is the only cold-start source. Its
	// compare-and-swap lifecycle mutates only a completely empty durable state;
	// existing desired or active revisions remain explicit rollout operations.
	if err := runtime.catalog.EnsureColdStart(ctx); err != nil {
		return fmt.Errorf("initialize Provider Catalog: %w", err)
	}
	// The first reconciliation is synchronous so no listener can report ready
	// before this replica has checked and acknowledged the active immutable
	// snapshot, including on an idempotent restart.
	if err := runtime.catalog.Reconcile(ctx); err != nil {
		return fmt.Errorf("reconcile active Provider Catalog: %w", err)
	}
	// Establish process membership before the synchronous publication pass. A
	// cold start with pending work must never publish against an empty fleet or
	// wait for a background loop that cannot start until reconciliation ends.
	if err := runtime.routingReplica.EnsureFleetLease(ctx); err != nil {
		return fmt.Errorf("establish routing fleet lease: %w", err)
	}
	// Management routes are deliberately allowed to start before inference is
	// ready. A new cluster must be able to perform identity and routing
	// publication bootstrap while the data plane remains fail-closed.
	// Drain or stage one publication synchronously. Startup cannot claim ready
	// while the durable outbox or applied runtime is already known to be
	// corrupt or unreachable.
	if _, err := runtime.publisherProcessor.ProcessOnce(ctx); err != nil {
		return fmt.Errorf("reconcile access publication: %w", err)
	}
	// Usage ingestion discovers namespace partitions from the durable desired
	// state. Reconcile once before readiness so a replica cannot accept traffic
	// while its immutable usage ledger is disconnected.
	if runtime.accessEnabled && runtime.usageSupervisor == nil {
		return errors.New("managed access usage supervisor is unavailable")
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

	workerContext, cancel := context.WithCancel(ctx)
	if err := runtime.backendDispatch.Start(workerContext); err != nil {
		cancel()
		return fmt.Errorf("start backend dispatch: %w", err)
	}
	runtime.mu.Lock()
	if runtime.closed {
		runtime.mu.Unlock()
		cancel()
		return errors.New("managed runtime closed during startup")
	}
	runtime.cancel = cancel
	runtime.started = true
	workerCount := 4
	if runtime.usageSupervisor != nil {
		workerCount++
	}
	if runtime.outcomeProjector != nil {
		workerCount++
	}
	runtime.workers.Add(workerCount)
	runtime.mu.Unlock()
	go func() {
		defer runtime.workers.Done()
		err := runtime.catalog.Run(workerContext)
		if err == nil || errors.Is(err, context.Canceled) || workerContext.Err() != nil {
			return
		}
		runtime.mu.Lock()
		runtime.backgroundErr = fmt.Errorf("provider Catalog replica stopped")
		runtime.mu.Unlock()
	}()
	go func() {
		defer runtime.workers.Done()
		err := runtime.routingReplica.Run(workerContext)
		if err == nil || errors.Is(err, context.Canceled) || workerContext.Err() != nil {
			return
		}
		runtime.mu.Lock()
		runtime.backgroundErr = fmt.Errorf("routing publication replica stopped")
		runtime.mu.Unlock()
	}()
	go func() {
		defer runtime.workers.Done()
		err := runtime.publisherWorker.Run(workerContext)
		if err == nil || errors.Is(err, context.Canceled) || workerContext.Err() != nil {
			return
		}
		runtime.mu.Lock()
		runtime.backgroundErr = fmt.Errorf("access publication worker stopped")
		runtime.mu.Unlock()
	}()
	if runtime.usageSupervisor != nil {
		go func() {
			defer runtime.workers.Done()
			err := runtime.usageSupervisor.Run(workerContext)
			if err == nil || errors.Is(err, context.Canceled) || workerContext.Err() != nil {
				return
			}
			runtime.mu.Lock()
			runtime.backgroundErr = fmt.Errorf("usage ledger supervisor stopped")
			runtime.mu.Unlock()
		}()
	}
	if runtime.outcomeProjector != nil {
		go func() {
			defer runtime.workers.Done()
			err := runtime.outcomeProjector.Run(workerContext)
			if err == nil || errors.Is(err, context.Canceled) || workerContext.Err() != nil {
				return
			}
			runtime.mu.Lock()
			runtime.backgroundErr = fmt.Errorf("outcome learning projector stopped")
			runtime.mu.Unlock()
		}()
	}
	go func() {
		defer runtime.workers.Done()
		err := runtime.management.Run(workerContext)
		if err == nil || errors.Is(err, context.Canceled) || workerContext.Err() != nil {
			return
		}
		runtime.mu.Lock()
		runtime.backgroundErr = fmt.Errorf("managed Management worker stopped")
		runtime.mu.Unlock()
	}()
	select {
	case <-runtime.publisherWorker.Started():
	case <-ctx.Done():
		cancel()
		runtime.workers.Wait()
		return ctx.Err()
	}
	if runtime.usageSupervisor != nil {
		select {
		case <-runtime.usageSupervisor.Started():
		case <-ctx.Done():
			cancel()
			runtime.workers.Wait()
			return ctx.Err()
		}
	}
	return nil
}

func (runtime *Runtime) Ready(ctx context.Context) error {
	if runtime == nil {
		return errors.New("managed runtime is unavailable")
	}
	runtime.mu.RLock()
	started, closed, backgroundErr := runtime.started, runtime.closed, runtime.backgroundErr
	mode := runtime.mode
	runtime.mu.RUnlock()
	if closed {
		return errors.New("managed runtime is closed")
	}
	if !started {
		return errors.New("managed runtime has not started")
	}
	// The decoder finalizer and ExtProc reader are deliberately the same
	// rendezvous. Starting either half without the other would
	// make successful responses unaccountable.
	if runtime.responseTerminals == nil || runtime.protocolCodecs == nil {
		return errors.New("neutral response runtime is unavailable")
	}
	if mode == config.ControlPlaneModeStandalone {
		if runtime.backendDispatch == nil {
			return errors.New("standalone backend dispatch is unavailable")
		}
		return runtime.backendDispatch.Ready()
	}
	if backgroundErr != nil {
		return backgroundErr
	}
	if runtime.catalog == nil || runtime.database == nil || runtime.redisReady == nil || runtime.publisherWorker == nil ||
		runtime.routingReplica == nil || runtime.backendDispatch == nil {
		return errors.New("managed runtime dependencies are unavailable")
	}
	if err := runtime.catalog.Ready(ctx); err != nil {
		return err
	}
	if err := runtime.database.PingContext(ctx); err != nil {
		return errors.New("managed PostgreSQL is unavailable")
	}
	if err := runtime.redisReady(ctx); err != nil {
		return errors.New("managed Valkey is unavailable")
	}
	if err := runtime.publisherWorker.Ready(ctx); err != nil {
		return err
	}
	if runtime.accessEnabled && runtime.usageSupervisor == nil {
		return errors.New("managed access usage supervisor is unavailable")
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

type runtimeManagedAPI struct {
	runtime  *Runtime
	delegate ManagedAPI
}

func (api runtimeManagedAPI) Register(mux *http.ServeMux) {
	if api.runtime == nil || api.delegate == nil || mux == nil {
		panic("managed API composition is unavailable")
	}
	api.delegate.Register(mux)
}

func (api runtimeManagedAPI) Ready(ctx context.Context) error {
	if err := api.runtime.Ready(ctx); err != nil {
		return err
	}
	if err := api.delegate.Ready(ctx); err != nil {
		return fmt.Errorf("managed API is not ready: %w", err)
	}
	return nil
}

func (api runtimeManagedAPI) Run(ctx context.Context) error {
	if api.runtime == nil || api.delegate == nil {
		return errors.New("managed API is unavailable")
	}
	return api.delegate.Run(ctx)
}
