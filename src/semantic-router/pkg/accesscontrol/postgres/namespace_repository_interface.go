package postgres

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

type NamespaceReader interface {
	GetNamespace(context.Context, accesscontrol.NamespaceID) (accesscontrol.Namespace, error)
}

type NamespaceWriter interface {
	CreateNamespace(context.Context, accesscontrol.Namespace, MutationMeta) (MutationResult[accesscontrol.Namespace], error)
	SetNamespaceStatus(context.Context, accesscontrol.NamespaceID, accesscontrol.Revision, accesscontrol.NamespaceStatus, MutationMeta) (MutationResult[accesscontrol.Namespace], error)
}

type NamespaceRepository interface {
	NamespaceReader
	NamespaceWriter
}
