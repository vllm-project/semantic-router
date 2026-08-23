package managementserver

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// RoutingManagementService is the narrow Model/Recipe/Entrypoint application
// seam consumed by the Management transport. Models, Recipes, and Entrypoints
// are the complete public routing resource model; the Dashboard is one client.
type RoutingManagementService interface {
	ListModels(context.Context, string, routingmanagement.PageRequest) (routingmanagement.Page[routingmanagement.Model], error)
	GetModel(context.Context, string, string) (routingmanagement.Model, error)
	CreateModel(context.Context, string, routingmanagement.ModelInput, routingmanagement.MutationContext) (routingmanagement.Model, routingmanagement.RevisionReceipt, error)
	PatchModel(context.Context, string, string, int64, routingmanagement.ModelPatch, routingmanagement.MutationContext) (routingmanagement.Model, routingmanagement.RevisionReceipt, error)
	DeleteModel(context.Context, string, string, int64, routingmanagement.MutationContext) (routingmanagement.RevisionReceipt, error)
	BulkImport(context.Context, routingmanagement.BulkImportRequest, routingmanagement.MutationContext) ([]routingmanagement.Model, routingmanagement.RevisionReceipt, error)
	ProbeModel(context.Context, string, string, time.Duration) (routingmanagement.ProbeResult, error)

	ListRecipes(context.Context, string, routingmanagement.PageRequest) (routingmanagement.Page[routingmanagement.Recipe], error)
	GetRecipe(context.Context, string, string) (routingmanagement.Recipe, error)
	CreateRecipe(context.Context, string, routingmanagement.RecipeInput, routingmanagement.MutationContext) (routingmanagement.Recipe, routingmanagement.RevisionReceipt, error)
	UpdateRecipe(context.Context, string, string, int64, routingmanagement.RecipeInput, routingmanagement.MutationContext) (routingmanagement.Recipe, routingmanagement.RevisionReceipt, error)
	DeleteRecipe(context.Context, string, string, int64, routingmanagement.MutationContext) (routingmanagement.RevisionReceipt, error)

	ListEntrypoints(context.Context, string, routingmanagement.PageRequest) (routingmanagement.Page[routingmanagement.Entrypoint], error)
	GetEntrypoint(context.Context, string, string) (routingmanagement.Entrypoint, error)
	CreateEntrypoint(context.Context, string, routingmanagement.EntrypointInput, routingmanagement.MutationContext) (routingmanagement.Entrypoint, routingmanagement.RevisionReceipt, error)
	UpdateEntrypoint(context.Context, string, string, int64, routingmanagement.EntrypointInput, routingmanagement.MutationContext) (routingmanagement.Entrypoint, routingmanagement.RevisionReceipt, error)
	DeleteEntrypoint(context.Context, string, string, int64, routingmanagement.MutationContext) (routingmanagement.RevisionReceipt, error)
	PublishEntrypoint(context.Context, string, string, int64, routingmanagement.MutationContext) (*routingsnapshot.Snapshot, routingmanagement.RevisionReceipt, error)
	UnpublishEntrypoint(context.Context, string, string, int64, routingmanagement.MutationContext) (*routingsnapshot.Snapshot, routingmanagement.RevisionReceipt, error)
	ResolveEntrypoint(context.Context, string, string, string, map[string]routingsnapshot.ClaimValue) (routingsnapshot.Resolution, error)

	ListSnapshots(context.Context, string, routingmanagement.SnapshotPageRequest) (routingmanagement.Page[routingmanagement.SnapshotMetadata], error)
	GetSnapshot(context.Context, string, int64) (routingmanagement.SnapshotDetail, error)
}

// RoutingCommandResults reads immutable command receipts before mutable
// Provider catalog, credential, and discovery-claim validation. A completed
// retry therefore remains replayable after catalog rotation or claim expiry.
type RoutingCommandResults interface {
	Lookup(context.Context, managementcommand.Command) (managementcommand.StoredResult, bool, error)
	Ready(context.Context, *managementcommand.Codec) error
}

type RoutingRoutesOptions struct {
	Service        RoutingManagementService
	Commands       *managementcommand.Codec
	CommandResults RoutingCommandResults
	Namespaces     NamespaceResolver
	Sessions       SessionAuthenticator
	Authorization  Authorizer
	Scopes         ResultScopeResolver
	IdempotencyTTL time.Duration
	Now            func() time.Time
}
