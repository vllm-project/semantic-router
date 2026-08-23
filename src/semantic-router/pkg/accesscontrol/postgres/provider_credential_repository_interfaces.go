package postgres

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
)

type ProviderCredentialListRequest struct {
	ProviderID  string
	Status      providercredential.Status
	AfterStatus providercredential.Status
	AfterID     string
	PageSize    int
	Scope       accesscontrol.ResultScope
}

type ProviderCredentialListResult struct {
	Credentials []providercredential.Credential
	HasMore     bool
}

// ProviderCredentialRotation advances one active pointer while retaining the
// previous version for a bounded set of already-journaled dispatches.
type ProviderCredentialRotation struct {
	Version           providercredential.Version
	PreviousVersionID string
	RetireAt          time.Time
}

// ProviderCredentialRepository owns the durable backend-secret lifecycle. It
// returns ciphertext only to the in-process backend resolver, never to API
// serializers.
type ProviderCredentialRepository interface {
	GetProviderCredential(context.Context, accesscontrol.NamespaceID, string) (providercredential.Credential, error)
	ListProviderCredentials(context.Context, accesscontrol.NamespaceID, ProviderCredentialListRequest) (ProviderCredentialListResult, error)
	ReplayProviderCredentialCommand(context.Context, managementcommand.Command) (MutationResult[providercredential.Credential], bool, error)
	LoadActiveProviderCredential(context.Context, string) (providercredential.Credential, providercredential.Version, error)
	LoadPinnedProviderCredential(context.Context, string, string) (providercredential.Credential, providercredential.Version, error)
	CreateProviderCredential(context.Context, providercredential.Credential, providercredential.Version, managementcommand.Command, MutationMeta) (MutationResult[providercredential.Credential], error)
	RenameProviderCredential(context.Context, accesscontrol.NamespaceID, string, accesscontrol.Revision, string, MutationMeta) (MutationResult[providercredential.Credential], error)
	RotateProviderCredential(context.Context, accesscontrol.NamespaceID, string, accesscontrol.Revision, ProviderCredentialRotation, managementcommand.Command, MutationMeta) (MutationResult[providercredential.Credential], error)
	ReactivateProviderCredential(context.Context, accesscontrol.NamespaceID, string, accesscontrol.Revision, providercredential.Version, MutationMeta) (MutationResult[providercredential.Credential], error)
	DisableProviderCredential(context.Context, accesscontrol.NamespaceID, string, accesscontrol.Revision, MutationMeta) (MutationResult[providercredential.Credential], error)
	DeleteProviderCredential(context.Context, accesscontrol.NamespaceID, string, accesscontrol.Revision, MutationMeta) (MutationResult[providercredential.Credential], error)
}
