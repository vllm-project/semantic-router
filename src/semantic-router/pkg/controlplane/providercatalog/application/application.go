package application

import (
	"crypto/tls"
	"database/sql"
	"fmt"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	catalogpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providerdiscovery"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

type ApplicationOptions struct {
	DB                 *sql.DB
	Registry           *providercatalog.Registry
	DiscoveryAdapters  *providerdiscovery.Registry
	CredentialMetadata providerdiscovery.CredentialMetadataReader
	Credentials        providerdiscovery.CredentialResolver
	EgressPolicy       backendegress.Policy
	DialTimeout        time.Duration
	TLSConfig          *tls.Config
	CatalogCursorKeys  securitykeyring.Symmetric
	DiscoveryClaims    providerdiscovery.ClaimCodec
	DiscoveryClaimTTL  time.Duration
	Replica            ReplicaOptions
}

type Application struct {
	Registry    *providercatalog.Registry
	Coordinator *catalogpostgres.Coordinator
	Catalog     *providercatalog.Service
	Discovery   *providerdiscovery.Executor
	Replica     *Replica
	transport   *backendegress.Transport
}

func NewApplication(options ApplicationOptions) (*Application, error) {
	if options.Registry == nil {
		return nil, fmt.Errorf("provider integration registry is required")
	}
	if options.DiscoveryAdapters == nil {
		return nil, fmt.Errorf("provider discovery adapter registry is required")
	}
	coordinator, err := catalogpostgres.New(options.DB, options.Registry)
	if err != nil {
		return nil, err
	}
	discoveryPlugins, err := providercatalog.NewDiscoveryRegistry(options.DiscoveryAdapters.Validators())
	if err != nil {
		return nil, fmt.Errorf("compose Provider Catalog discovery validators: %w", err)
	}
	catalog, err := providercatalog.NewService(coordinator, providercatalog.ServiceOptions{
		CursorKeyring: options.CatalogCursorKeys, DiscoveryPlugins: discoveryPlugins,
	})
	if err != nil {
		return nil, err
	}
	transport, err := backendegress.NewTransport(backendegress.TransportOptions{
		Guard: backendegress.Guard{Policy: options.EgressPolicy}, DialTimeout: options.DialTimeout,
		TLSConfig: options.TLSConfig,
	})
	if err != nil {
		return nil, fmt.Errorf("compose Provider Catalog egress transport: %w", err)
	}
	replica, err := NewReplica(coordinator, options.Registry, options.Replica)
	if err != nil {
		transport.CloseIdleConnections()
		return nil, err
	}
	executor := &providerdiscovery.Executor{
		Registry: options.DiscoveryAdapters, CredentialMetadata: options.CredentialMetadata,
		Credentials: options.Credentials, EgressPolicy: options.EgressPolicy,
		Transport: transport, Claims: options.DiscoveryClaims, ClaimTTL: options.DiscoveryClaimTTL,
	}
	return &Application{
		Registry: options.Registry, Coordinator: coordinator, Catalog: catalog,
		Discovery: executor, Replica: replica, transport: transport,
	}, nil
}

func (application *Application) Close() {
	if application == nil {
		return
	}
	if application.transport != nil {
		application.transport.CloseIdleConnections()
	}
	if application.Catalog != nil {
		application.Catalog.Close()
	}
}

var _ http.RoundTripper = (*backendegress.Transport)(nil)
