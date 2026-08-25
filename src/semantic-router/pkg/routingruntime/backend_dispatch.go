package routingruntime

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backenddispatch"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

type backendDispatchComposition struct {
	config      config.BackendDispatchConfig
	credentials backendinvoker.CredentialResolver
	codecs      *protocolcodec.Registry
	journal     backendinvoker.Journal
	finalizer   backendinvoker.ResponseFinalizer
	observer    backendinvoker.ResponseObserver
	egress      backendegress.Policy
	dialTimeout time.Duration

	mu        sync.RWMutex
	runtime   *backenddispatch.Runtime
	server    *backenddispatch.Server
	attached  bool
	started   bool
	closed    bool
	closeOnce sync.Once
	closeErr  error
}

func newBackendDispatchComposition(
	cfg config.BackendDispatchConfig,
	credentials backendinvoker.CredentialResolver,
	codecs *protocolcodec.Registry,
	journal backendinvoker.Journal,
	finalizer backendinvoker.ResponseFinalizer,
	egress backendegress.Policy,
	dialTimeout time.Duration,
) (*backendDispatchComposition, error) {
	if credentials == nil || codecs == nil || journal == nil || finalizer == nil {
		return nil, errors.New("backend dispatch dependencies are incomplete")
	}
	if _, err := cfg.CapabilityLifetime(); err != nil {
		return nil, err
	}
	return &backendDispatchComposition{
		config: cfg, credentials: credentials, codecs: codecs, journal: journal, finalizer: finalizer,
		observer: backendinvoker.ForwardResponseObserver{}, egress: egress, dialTimeout: dialTimeout,
	}, nil
}

func (composition *backendDispatchComposition) Attach(
	snapshots backendinvoker.RoutingSnapshotSource,
	keyring securitykeyring.Symmetric,
) error {
	if composition == nil || snapshots == nil {
		return errors.New("backend dispatch snapshot source is required")
	}
	composition.mu.Lock()
	defer composition.mu.Unlock()
	if composition.closed || composition.started {
		return errors.New("backend dispatch cannot attach after startup or close")
	}
	if composition.attached {
		return errors.New("backend dispatch is already attached")
	}
	lifetime, err := composition.config.CapabilityLifetime()
	if err != nil {
		return err
	}
	transport, err := backendegress.NewTransport(backendegress.TransportOptions{
		Guard: backendegress.Guard{Policy: composition.egress}, DialTimeout: composition.dialTimeout,
	})
	if err != nil {
		return fmt.Errorf("compose backend dispatch transport: %w", err)
	}
	dispatchRuntime, err := backenddispatch.New(backenddispatch.Options{
		Audience: composition.config.Audience,
		CapabilityKeyring: backendinvoker.SigningKeyring{
			ActiveVersion: keyring.ActiveVersion,
			Keys:          keyring.Keys,
			MaxLifetime:   lifetime,
		},
		Snapshots: snapshots, Credentials: composition.credentials,
		Codecs: composition.codecs, Journal: composition.journal,
		Finalizer: composition.finalizer,
		Observer:  composition.observer, Transport: transport,
		MaxRequestBodyBytes: composition.config.MaxRequestBodyBytes,
	})
	zeroSymmetric(&keyring)
	if err != nil {
		transport.CloseIdleConnections()
		return fmt.Errorf("compose backend dispatch runtime: %w", err)
	}
	server, err := backenddispatch.NewServer(backenddispatch.ServerOptions{
		BindAddress: composition.config.BindAddress,
		Port:        composition.config.Port,
		Handler:     dispatchRuntime.Handler(),
	})
	if err != nil {
		_ = dispatchRuntime.Close()
		return fmt.Errorf("compose backend dispatch listener: %w", err)
	}
	composition.runtime = dispatchRuntime
	composition.server = server
	composition.attached = true
	return nil
}

func (composition *backendDispatchComposition) Start(ctx context.Context) error {
	if composition == nil {
		return errors.New("backend dispatch is unavailable")
	}
	composition.mu.Lock()
	defer composition.mu.Unlock()
	if composition.closed {
		return errors.New("backend dispatch is closed")
	}
	if !composition.attached || composition.server == nil {
		return errors.New("backend dispatch is not attached")
	}
	if composition.started {
		return errors.New("backend dispatch is already started")
	}
	if err := composition.server.Start(ctx); err != nil {
		return err
	}
	composition.started = true
	return nil
}

func (composition *backendDispatchComposition) Ready() error {
	if composition == nil {
		return errors.New("backend dispatch is unavailable")
	}
	composition.mu.RLock()
	defer composition.mu.RUnlock()
	if composition.closed || !composition.started || composition.server == nil {
		return errors.New("backend dispatch is not running")
	}
	return composition.server.Ready()
}

func (composition *backendDispatchComposition) Close() error {
	if composition == nil {
		return nil
	}
	composition.closeOnce.Do(func() {
		composition.mu.Lock()
		composition.closed = true
		server, dispatchRuntime := composition.server, composition.runtime
		composition.mu.Unlock()
		var closeErrors []error
		if server != nil {
			closeErrors = append(closeErrors, server.Close())
		}
		if dispatchRuntime != nil {
			closeErrors = append(closeErrors, dispatchRuntime.Close())
		}
		composition.closeErr = errors.Join(closeErrors...)
	})
	return composition.closeErr
}

func zeroSymmetric(keyring *securitykeyring.Symmetric) {
	if keyring == nil {
		return
	}
	for _, key := range keyring.Keys {
		clear(key)
	}
	*keyring = securitykeyring.Symmetric{}
}
