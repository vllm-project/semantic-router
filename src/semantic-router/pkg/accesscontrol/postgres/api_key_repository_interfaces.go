package postgres

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

type LogicalAPIKeyReader interface {
	GetAPIKey(context.Context, accesscontrol.NamespaceID, accesscontrol.APIKeyID) (accesscontrol.APIKey, error)
}

type LogicalAPIKeyWriter interface {
	CreateAPIKey(context.Context, accesscontrol.APIKey, accesscontrol.CredentialVersion, MutationMeta) (MutationResult[accesscontrol.APIKey], error)
	UpdateAPIKey(context.Context, accesscontrol.APIKey, accesscontrol.Revision, MutationMeta) (MutationResult[accesscontrol.APIKey], error)
	SoftDeleteAPIKey(context.Context, accesscontrol.NamespaceID, accesscontrol.APIKeyID, accesscontrol.Revision, MutationMeta) (MutationResult[accesscontrol.APIKey], error)
}

type LogicalAPIKeyRepository interface {
	LogicalAPIKeyReader
	LogicalAPIKeyWriter
}

type CredentialReader interface {
	ListCredentialVersions(context.Context, accesscontrol.NamespaceID, accesscontrol.APIKeyID) ([]CredentialRecord, error)
}

type CredentialWriter interface {
	RotateCredential(context.Context, accesscontrol.NamespaceID, accesscontrol.APIKeyID, accesscontrol.Revision, CredentialRotation, MutationMeta) (MutationResult[accesscontrol.APIKey], error)
	RevokeCredential(context.Context, accesscontrol.NamespaceID, accesscontrol.APIKeyID, accesscontrol.CredentialVersionID, accesscontrol.Revision, MutationMeta) (MutationResult[accesscontrol.APIKey], error)
}

type CredentialRepository interface {
	CredentialReader
	CredentialWriter
}

type APIKeyRepository interface {
	LogicalAPIKeyRepository
	CredentialRepository
}
