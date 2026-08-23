package managementcomposition

import (
	"errors"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managedruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementserver"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/runtimediagnostics"
)

func composeRuntimeDiagnostics(
	dependencies managedruntime.ManagementDependencies,
	keyPrefix string,
	maxUsageBacklog int64,
	sessions managementserver.SessionAuthenticator,
	authorization managementserver.Authorizer,
	now func() time.Time,
) (*managementserver.RuntimeDiagnosticsRoutes, error) {
	if dependencies.Database == nil || dependencies.Redis == nil || keyPrefix == "" || maxUsageBacklog < 1 ||
		sessions == nil || authorization == nil {
		return nil, errors.New("management runtime diagnostics dependencies are incomplete")
	}
	publications, err := accesspublisher.NewRedisStore(accesspublisher.RedisStoreOptions{
		Client: dependencies.Redis, KeyPrefix: keyPrefix,
	})
	if err != nil {
		return nil, fmt.Errorf("compose publication diagnostics: %w", err)
	}
	quota, err := quotaruntime.NewRedisDiagnostics(dependencies.Redis, keyPrefix)
	if err != nil {
		return nil, fmt.Errorf("compose quota diagnostics: %w", err)
	}
	service, err := runtimediagnostics.New(runtimediagnostics.Options{
		Database: dependencies.Database, Valkey: dependencies.Redis,
		Publications: publications, Quota: quota,
		MaxUsageBacklog: maxUsageBacklog, Now: now,
	})
	if err != nil {
		return nil, err
	}
	routes, err := managementserver.NewRuntimeDiagnosticsRoutes(managementserver.RuntimeDiagnosticsRoutesOptions{
		Service: service, Sessions: sessions, Authorization: authorization, Now: now,
	})
	if err != nil {
		return nil, err
	}
	return routes, nil
}
