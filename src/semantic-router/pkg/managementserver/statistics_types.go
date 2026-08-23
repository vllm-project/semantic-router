package managementserver

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementstatistics"
)

type StatisticsQueryService interface {
	Ready(context.Context) error
	Snapshot(context.Context, managementstatistics.Request) (managementstatistics.Snapshot, error)
}

type StatisticsRoutesOptions struct {
	Service    StatisticsQueryService
	Scopes     ResultScopeResolver
	Namespaces NamespaceResolver
	Sessions   SessionAuthenticator
	Now        func() time.Time
}
