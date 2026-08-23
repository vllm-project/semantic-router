package routingmanagement

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// PublicationValidator compiles one immutable snapshot against the process's
// managed bootstrap contract. It runs inside the Entrypoint publication
// transaction, before desired state and the outbox revision are committed.
type PublicationValidator func(*routingsnapshot.Snapshot) error

type Store interface {
	NamespaceCurrency(context.Context, string) (string, error)

	ListModels(context.Context, string, ListQuery) (ListResult[Model], error)
	GetModel(context.Context, string, string) (Model, error)
	CreateModels(context.Context, string, []routingsnapshot.Model, MutationContext) ([]Model, RevisionReceipt, error)
	UpdateModel(context.Context, string, string, int64, routingsnapshot.Model, MutationContext) (Model, RevisionReceipt, error)
	DeleteModel(context.Context, string, string, int64, MutationContext) (RevisionReceipt, error)

	ListRecipes(context.Context, string, ListQuery) (ListResult[Recipe], error)
	GetRecipe(context.Context, string, string) (Recipe, error)
	CreateRecipe(context.Context, string, string, routingsnapshot.Recipe, MutationContext) (Recipe, RevisionReceipt, error)
	UpdateRecipe(context.Context, string, string, int64, string, routingsnapshot.Recipe, MutationContext) (Recipe, RevisionReceipt, error)
	DeleteRecipe(context.Context, string, string, int64, MutationContext) (RevisionReceipt, error)

	ListEntrypoints(context.Context, string, ListQuery) (ListResult[Entrypoint], error)
	GetEntrypoint(context.Context, string, string) (Entrypoint, error)
	CreateEntrypoint(context.Context, string, routingsnapshot.Entrypoint, MutationContext) (Entrypoint, RevisionReceipt, error)
	UpdateEntrypoint(context.Context, string, string, int64, routingsnapshot.Entrypoint, MutationContext) (Entrypoint, RevisionReceipt, error)
	DeleteEntrypoint(context.Context, string, string, int64, MutationContext) (RevisionReceipt, error)
	PublishEntrypoint(context.Context, string, string, int64, MutationContext) (*routingsnapshot.Snapshot, RevisionReceipt, error)
	UnpublishEntrypoint(context.Context, string, string, int64, MutationContext) (*routingsnapshot.Snapshot, RevisionReceipt, error)
	ActiveSnapshot(context.Context, string) (*routingsnapshot.Snapshot, error)

	ListSnapshots(context.Context, string, SnapshotListQuery) (ListResult[SnapshotMetadata], error)
	GetSnapshot(context.Context, string, int64) (SnapshotDetail, error)
}

// BuiltInRecipeStore owns the atomic, cross-replica installation seam for
// Router distribution Recipes. It is separate from Store so ordinary authoring
// repositories and tests do not acquire bootstrap-only responsibilities.
type BuiltInRecipeStore interface {
	PendingBuiltInRecipeNamespaces(context.Context, BuiltInRecipeDistribution, int) ([]string, error)
	InstallBuiltInRecipes(context.Context, string, BuiltInRecipeDistribution, MutationContext) ([]Recipe, error)
	VerifyBuiltInRecipes(context.Context, BuiltInRecipeDistribution) error
}
