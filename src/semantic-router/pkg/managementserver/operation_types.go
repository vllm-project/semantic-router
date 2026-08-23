package managementserver

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policybulk"
)

type OperationService interface {
	Ready(context.Context) error
	Get(context.Context, string, string) (policybulk.Operation, error)
	List(context.Context, policybulk.ListRequest) (policybulk.Page, error)
	Cancel(context.Context, policybulk.CancelRequest) (policybulk.CancelResult, error)
}

type OperationRoutesOptions struct {
	Service       OperationService
	DetailReaders []OperationDetailReader
	Namespaces    NamespaceResolver
	Sessions      SessionAuthenticator
	Authorization Authorizer
	Scopes        ResultScopeResolver
	Now           func() time.Time
}
