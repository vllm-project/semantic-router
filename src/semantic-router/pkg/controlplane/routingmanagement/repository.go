package routingmanagement

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// PublicationValidator compiles one immutable snapshot against the process's
// durable bootstrap contract. It runs inside the Entrypoint publication
// transaction, before desired state and the outbox revision are committed.
type PublicationValidator func(*routingsnapshot.Snapshot) error

// ManifestCodec is the sole file-authoring boundary. The routing domain owns
// immutable values and never depends on the public config package.
type ManifestCodec interface {
	Decode([]byte) (*routingsnapshot.Snapshot, error)
	Encode(*routingsnapshot.Snapshot) ([]byte, error)
}

type Store interface {
	NamespaceStore
	ProviderCredentialReferenceStore
	ModelStore
	RecipeStore
	EntrypointStore
	PublicationStore
	SnapshotStore
}

// ProviderCredentialReferenceStore is the Namespace-scoped name/identity
// boundary used only by human-readable routing manifests. Runtime snapshots
// and every mutation below this interface carry immutable credential UUIDs.
type ProviderCredentialReferenceStore interface {
	ProviderCredentialIDsByName(context.Context, string, []string) (map[string]string, error)
	ProviderCredentialNamesByID(context.Context, string, []string) (map[string]string, error)
}

type NamespaceStore interface {
	NamespaceCurrency(context.Context, string) (string, error)
}

type ModelStore interface {
	ListModels(context.Context, string, ListQuery) (ListResult[Model], error)
	GetModel(context.Context, string, string) (Model, error)
	CreateModels(context.Context, string, []routingsnapshot.Model, MutationContext) ([]Model, RevisionReceipt, error)
	UpdateModel(context.Context, string, string, int64, routingsnapshot.Model, MutationContext) (Model, RevisionReceipt, error)
	DeleteModel(context.Context, string, string, int64, MutationContext) (RevisionReceipt, error)
}

type RecipeStore interface {
	ListRecipes(context.Context, string, ListQuery) (ListResult[Recipe], error)
	GetRecipe(context.Context, string, string) (Recipe, error)
	CreateRecipe(context.Context, string, string, routingsnapshot.Recipe, MutationContext) (Recipe, RevisionReceipt, error)
	UpdateRecipe(context.Context, string, string, int64, string, routingsnapshot.Recipe, MutationContext) (Recipe, RevisionReceipt, error)
	DeleteRecipe(context.Context, string, string, int64, MutationContext) (RevisionReceipt, error)
}

type EntrypointStore interface {
	ListEntrypoints(context.Context, string, ListQuery) (ListResult[Entrypoint], error)
	GetEntrypoint(context.Context, string, string) (Entrypoint, error)
	CreateEntrypoint(context.Context, string, routingsnapshot.Entrypoint, MutationContext) (Entrypoint, RevisionReceipt, error)
	UpdateEntrypoint(context.Context, string, string, int64, routingsnapshot.Entrypoint, MutationContext) (Entrypoint, RevisionReceipt, error)
	DeleteEntrypoint(context.Context, string, string, int64, MutationContext) (RevisionReceipt, error)
}

type PublicationStore interface {
	PublishEntrypoint(context.Context, string, string, int64, MutationContext) (*routingsnapshot.Snapshot, RevisionReceipt, error)
	UnpublishEntrypoint(context.Context, string, string, int64, MutationContext) (*routingsnapshot.Snapshot, RevisionReceipt, error)
	ActiveSnapshot(context.Context, string) (*routingsnapshot.Snapshot, error)
}

type SnapshotStore interface {
	ListSnapshots(context.Context, string, SnapshotListQuery) (ListResult[SnapshotMetadata], error)
	GetSnapshot(context.Context, string, int64) (SnapshotDetail, error)
	PreviewManifest(context.Context, string, int64, *routingsnapshot.Snapshot) (ManifestDiff, error)
	ImportManifest(context.Context, string, int64, *routingsnapshot.Snapshot, MutationContext) (ManifestDiff, RevisionReceipt, error)
	CurrentManifest(context.Context, string) (*routingsnapshot.Snapshot, int64, error)
}

// BuiltInRecipeStore owns the atomic, cross-replica installation seam for
// Router distribution Recipes. It is separate from Store so ordinary authoring
// repositories and tests do not acquire bootstrap-only responsibilities.
type BuiltInRecipeStore interface {
	PendingBuiltInRecipeNamespaces(context.Context, BuiltInRecipeDistribution, int) ([]string, error)
	InstallBuiltInRecipes(context.Context, string, BuiltInRecipeDistribution, MutationContext) ([]Recipe, error)
	VerifyBuiltInRecipes(context.Context, BuiltInRecipeDistribution) error
}
