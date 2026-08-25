package managementcomposition

import (
	"errors"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementserver"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementstatistics"
	statisticspostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementstatistics/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingruntime"
)

func composeStatistics(
	dependencies routingruntime.ManagementDependencies,
	authorization managementauthorization.Runtime,
	namespaces managementserver.NamespaceResolver,
	sessions managementserver.SessionAuthenticator,
	now func() time.Time,
) (*managementserver.StatisticsRoutes, error) {
	if dependencies.Database == nil || authorization.Loader == nil || namespaces == nil || sessions == nil {
		return nil, errors.New("management statistics dependencies are incomplete")
	}
	repository, err := statisticspostgres.New(dependencies.Database)
	if err != nil {
		return nil, err
	}
	service, err := managementstatistics.NewService(managementstatistics.Options{Repository: repository, Now: now})
	if err != nil {
		return nil, err
	}
	return managementserver.NewStatisticsRoutes(managementserver.StatisticsRoutesOptions{
		Service: service, Scopes: authorization, Namespaces: namespaces, Sessions: sessions, Now: now,
	})
}
