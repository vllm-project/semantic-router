package management

import (
	"context"
	"net/netip"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	accesspostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
)

type Catalog interface {
	Get(context.Context, string) (providercatalog.DetailResult, error)
}

type EgressPolicy interface {
	AuthorizeOrigin(string) (backendegress.Target, error)
}

type Repository interface {
	RepositoryLifecycle
	CredentialReader
	CredentialCommandReplay
	CredentialMutationRepository
	CredentialDeletionRepository
}

type RepositoryLifecycle interface {
	ValidateManagementCommandHMACVersions(context.Context, *managementcommand.Codec) error
}

type CredentialReader interface {
	GetProviderCredential(context.Context, accesscontrol.NamespaceID, string) (providercredential.Credential, error)
	ListProviderCredentials(context.Context, accesscontrol.NamespaceID, accesspostgres.ProviderCredentialListRequest) (accesspostgres.ProviderCredentialListResult, error)
}

type CredentialCommandReplay interface {
	ReplayProviderCredentialCommand(context.Context, managementcommand.Command) (accesspostgres.MutationResult[providercredential.Credential], bool, error)
}

type CredentialMutationRepository interface {
	CreateProviderCredential(context.Context, providercredential.Credential, providercredential.Version, managementcommand.Command, accesspostgres.MutationMeta) (accesspostgres.MutationResult[providercredential.Credential], error)
	RenameProviderCredential(context.Context, accesscontrol.NamespaceID, string, accesscontrol.Revision, string, accesspostgres.MutationMeta) (accesspostgres.MutationResult[providercredential.Credential], error)
	RotateProviderCredential(context.Context, accesscontrol.NamespaceID, string, accesscontrol.Revision, accesspostgres.ProviderCredentialRotation, managementcommand.Command, accesspostgres.MutationMeta) (accesspostgres.MutationResult[providercredential.Credential], error)
	ReactivateProviderCredential(context.Context, accesscontrol.NamespaceID, string, accesscontrol.Revision, providercredential.Version, accesspostgres.MutationMeta) (accesspostgres.MutationResult[providercredential.Credential], error)
	DisableProviderCredential(context.Context, accesscontrol.NamespaceID, string, accesscontrol.Revision, accesspostgres.MutationMeta) (accesspostgres.MutationResult[providercredential.Credential], error)
}

type CredentialDeletionRepository interface {
	DeleteProviderCredential(context.Context, accesscontrol.NamespaceID, string, accesscontrol.Revision, accesspostgres.MutationMeta) (accesspostgres.MutationResult[providercredential.Credential], error)
}

type Actor struct {
	PrincipalID string
	ActorChain  []string
	RequestID   string
	SourceIP    netip.Addr
}

type Metadata struct {
	CredentialID     string
	NamespaceID      string
	Name             string
	ProviderID       string
	CatalogRevision  string
	NormalizedOrigin string
	Status           providercredential.Status
	Revision         uint64
	CreatedAt        time.Time
	UpdatedAt        time.Time
	DeletedAt        *time.Time
}

type MutationResult struct {
	CredentialID string
	Revision     uint64
	Replayed     bool
}

type ListRequest struct {
	NamespaceID string
	ProviderID  string
	Status      providercredential.Status
	Cursor      string
	PageSize    int
	Scope       accesscontrol.ResultScope
}

type ListResult struct {
	Credentials []Metadata
	NextCursor  string
	HasMore     bool
	PageSize    int
}

type CreateRequest struct {
	NamespaceID    string
	Name           string
	ProviderID     string
	BaseURL        string
	Secret         []byte
	IdempotencyKey string
	Actor          Actor
}

type RotateRequest struct {
	NamespaceID      string
	CredentialID     string
	ExpectedRevision uint64
	Secret           []byte
	IdempotencyKey   string
	Actor            Actor
}

type RenameRequest struct {
	NamespaceID      string
	CredentialID     string
	ExpectedRevision uint64
	Name             string
	Actor            Actor
}

type LifecycleRequest struct {
	NamespaceID      string
	CredentialID     string
	ExpectedRevision uint64
	Secret           []byte
	Actor            Actor
}
