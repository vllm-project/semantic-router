// Package managed composes Router-native Model, Recipe, and Entrypoint
// authoring with the Management HTTP registrar. It is a narrow production
// factory: the outer runtime owns listeners, lifecycle, and Provider adapter
// installation.
package managed

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providerdiscovery"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	routingpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementserver"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

type ApplicationOptions struct {
	DB                       *sql.DB
	ModelCompiler            providercatalog.ModelCompiler
	DiscoveryClaims          providerdiscovery.ClaimCodec
	CredentialVersions       routingmanagement.CredentialVersionReader
	Prober                   routingmanagement.Prober
	ValidatePublication      routingmanagement.PublicationValidator
	CommandCodec             *managementcommand.Codec
	CursorKeyring            securitykeyring.Symmetric
	IdempotencyTTL           time.Duration
	Namespaces               managementserver.NamespaceResolver
	Sessions                 managementserver.SessionAuthenticator
	Authorization            managementserver.Authorizer
	BuiltInRecipes           routingmanagement.BuiltInRecipeDistribution
	BuiltInReconcileInterval time.Duration
	Now                      func() time.Time
}

type Application struct {
	Store          *routingpostgres.Store
	Service        *routingmanagement.Service
	Routes         *managementserver.RoutingRoutes
	BuiltInRecipes *routingmanagement.BuiltInRecipeInstaller
	db             *sql.DB
}

func NewApplication(options ApplicationOptions) (*Application, error) {
	if options.DB == nil || options.ValidatePublication == nil || options.CommandCodec == nil || options.Namespaces == nil ||
		options.Sessions == nil || options.Authorization == nil {
		return nil, fmt.Errorf("managed Routing application requires database, publication validation, command, namespace, session, and authorization dependencies")
	}
	store, err := routingpostgres.New(options.DB, options.ValidatePublication)
	if err != nil {
		return nil, err
	}
	builtInRecipes, err := routingmanagement.NewBuiltInRecipeInstaller(
		routingmanagement.BuiltInRecipeInstallerOptions{
			Store: store, Distribution: options.BuiltInRecipes,
			ReconcileInterval: options.BuiltInReconcileInterval, Now: options.Now,
		},
	)
	if err != nil {
		return nil, fmt.Errorf("compose built-in Recipe installer: %w", err)
	}
	service, err := routingmanagement.NewService(routingmanagement.ServiceOptions{
		Store: store, ModelCompiler: options.ModelCompiler, DiscoveryClaims: options.DiscoveryClaims,
		CredentialVersions: options.CredentialVersions, Prober: options.Prober,
		CursorKeyring: options.CursorKeyring, Now: options.Now,
	})
	if err != nil {
		return nil, err
	}
	commandResults := postgresCommandResults{db: options.DB}
	routes, err := managementserver.NewRoutingRoutes(managementserver.RoutingRoutesOptions{
		Service: service, Commands: options.CommandCodec, CommandResults: commandResults,
		Namespaces: options.Namespaces, Sessions: options.Sessions, Authorization: options.Authorization,
		IdempotencyTTL: options.IdempotencyTTL, Now: options.Now,
	})
	if err != nil {
		return nil, err
	}
	return &Application{
		Store: store, Service: service, Routes: routes, BuiltInRecipes: builtInRecipes, db: options.DB,
	}, nil
}

func (application *Application) Close() error {
	if application != nil && application.Service != nil {
		application.Service.Close()
	}
	return nil
}

// ReconcileBuiltInRecipes is the startup gate. The managed listener is not
// returned to the process until the distribution is durably present in every
// active Namespace.
func (application *Application) ReconcileBuiltInRecipes(ctx context.Context) error {
	if application == nil || application.BuiltInRecipes == nil {
		return errors.New("managed built-in Recipes are unavailable")
	}
	return application.BuiltInRecipes.Reconcile(ctx)
}

// Run keeps Namespaces created after startup converged. PostgreSQL serializes
// competing replicas, so every Router replica may run the same worker.
func (application *Application) Run(ctx context.Context) error {
	if application == nil || application.BuiltInRecipes == nil {
		return errors.New("managed built-in Recipes are unavailable")
	}
	return application.BuiltInRecipes.Run(ctx)
}

func (application *Application) Register(mux *http.ServeMux) {
	if application == nil || application.Routes == nil {
		panic("managed Routing application is unavailable")
	}
	application.Routes.Register(mux)
}

func (application *Application) Ready(ctx context.Context) error {
	if application == nil || application.db == nil || application.Routes == nil {
		return errors.New("managed Routing application is unavailable")
	}
	if err := application.db.PingContext(ctx); err != nil {
		return fmt.Errorf("managed Routing PostgreSQL is unavailable: %w", err)
	}
	if err := application.BuiltInRecipes.Ready(ctx); err != nil {
		return err
	}
	return application.Routes.Ready(ctx)
}

type postgresCommandResults struct{ db *sql.DB }

func (results postgresCommandResults) Lookup(
	ctx context.Context, command managementcommand.Command,
) (managementcommand.StoredResult, bool, error) {
	return commandpostgres.Lookup(ctx, results.db, command)
}

func (results postgresCommandResults) Ready(ctx context.Context, codec *managementcommand.Codec) error {
	return commandpostgres.ValidateReferencedHMACVersions(ctx, results.db, codec)
}

var _ managementserver.RouteRegistrar = (*Application)(nil)
