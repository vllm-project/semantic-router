package managedruntime

import (
	"context"
	"database/sql"
	"net/http"
	"time"

	"github.com/redis/go-redis/v9"

	accesspostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	catalogmanaged "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog/managed"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providerdiscovery"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	managementauthpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential/backendresolver"
)

// InferenceDelegationAudience is owned by the Router binary. Management
// clients can select a key, but cannot mint credentials for another audience.
const InferenceDelegationAudience = "vllm-sr-inference"

// ManagedAPI is mounted on the Router's existing Management listener. Its
// background lifecycle is owned by Runtime, never by the HTTP server.
type ManagedAPI interface {
	Register(*http.ServeMux)
	Ready(context.Context) error
	Run(context.Context) error
}

// ManagementFactory is the one production injection point for Management
// identity, authorization, and domain route composition. Managed startup must
// fail when it is absent or returns nil; no allow-all or Dashboard fallback is
// permitted.
type ManagementFactory interface {
	Build(context.Context, ManagementDependencies) (ManagedAPI, error)
}

type ManagementFactoryFunc func(context.Context, ManagementDependencies) (ManagedAPI, error)

func (function ManagementFactoryFunc) Build(
	ctx context.Context,
	dependencies ManagementDependencies,
) (ManagedAPI, error) {
	return function(ctx, dependencies)
}

// ManagementDependencies are process-owned borrowed resources. A Management
// application must not close them or retain key bytes after Runtime.Close.
type ManagementDependencies struct {
	Database                   *sql.DB
	Redis                      *redis.Client
	AccessStore                *accesspostgres.Store
	SessionStore               *managementauthpostgres.Store
	Catalog                    *catalogmanaged.Application
	EgressPolicy               backendegress.Policy
	ProtocolCodecs             *protocolcodec.Registry
	CredentialAdapters         backendresolver.StaticRegistry
	DiscoveryAdapters          *providerdiscovery.Registry
	ProviderCredentialCodec    providercredential.Codec
	ProviderCredentialResolver backendresolver.Resolver
	ModelProber                routingmanagement.Prober
	BootstrapToken             []byte
	BootstrapTokenPresent      func() (bool, error)
	RecoveryToken              []byte
	Keyrings                   DeploymentKeyrings
	ReplicaID                  string
	DelegationAudience         string
}

type Options struct {
	ManagementFactory        ManagementFactory
	ProviderIntegrations     []providercatalog.Integration
	ProviderBackendCompilers []providercatalog.BackendCompiler
	StartupTimeout           time.Duration
	BackendDialTimeout       time.Duration
	DiscoveryClaimTTL        time.Duration
}
