package managed

import (
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementserver"
)

type ManagementServerOptions struct {
	Namespaces       managementserver.NamespaceResolver
	Sessions         managementserver.SessionAuthenticator
	Authorization    managementserver.Authorizer
	AdditionalRoutes []managementserver.RouteRegistrar
	Now              func() time.Time
}

// NewManagementServer binds this durable Catalog application to the
// Router-native Management HTTP transport. Identity and authorization remain
// injected control-plane dependencies; no Dashboard state or proxy is used.
func (application *Application) NewManagementServer(
	options ManagementServerOptions,
) (*managementserver.Server, error) {
	if application == nil || application.Catalog == nil || application.Discovery == nil || application.Replica == nil {
		return nil, fmt.Errorf("managed Provider Catalog application is not initialized")
	}
	routes, err := managementserver.NewProviderRoutes(managementserver.ProviderRoutesOptions{
		Catalog: application.Catalog, Discovery: application.Discovery,
		Namespaces: options.Namespaces, Sessions: options.Sessions,
		Authorization: options.Authorization, Now: options.Now,
	})
	if err != nil {
		return nil, err
	}
	administration, err := managementserver.NewProviderCatalogAdministrationRoutes(
		managementserver.ProviderCatalogAdministrationRoutesOptions{
			Administration: application.Replica, Sessions: options.Sessions,
			Authorization: options.Authorization, Now: options.Now,
		},
	)
	if err != nil {
		return nil, err
	}
	registrars := make([]managementserver.RouteRegistrar, 0, len(options.AdditionalRoutes)+2)
	registrars = append(registrars, routes, administration)
	registrars = append(registrars, options.AdditionalRoutes...)
	return managementserver.NewServer(application.Replica, registrars...)
}
