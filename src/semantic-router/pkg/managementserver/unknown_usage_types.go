package managementserver

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotareconciliation"
)

type UnknownUsageReadService interface {
	Ready(context.Context) error
	Get(context.Context, string, string) (quotareconciliation.Fence, error)
	GetOperation(context.Context, string, string) (quotareconciliation.Operation, error)
	List(context.Context, quotareconciliation.ListRequest) (quotareconciliation.Page, error)
}

type UnknownUsageWorker interface {
	Reconcile(context.Context, quotareconciliation.ReconcileRequest) (quotareconciliation.EnqueueResult, error)
	Run(context.Context) error
	Close()
}

type UnknownUsageService interface {
	UnknownUsageReadService
	UnknownUsageWorker
}

type UnknownUsageRoutesOptions struct {
	Service       UnknownUsageService
	Namespaces    NamespaceResolver
	Sessions      SessionAuthenticator
	Authorization Authorizer
	Scopes        ResultScopeResolver
	Now           func() time.Time
}
