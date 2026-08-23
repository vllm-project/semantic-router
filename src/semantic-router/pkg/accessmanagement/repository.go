package accessmanagement

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

type Repository interface {
	Ready(context.Context) error
	LoadPolicySnapshot(context.Context, string, Subject) (PolicySnapshot, error)
	ResourceExists(context.Context, string, accesscontrol.GrantResource) (bool, error)
	UpdateRoutingContext(context.Context, UpdateRoutingContextRequest) (RoutingContextMutation, error)
}

type AppliedPolicyReader interface {
	ReadAppliedPolicy(context.Context, string, string, string) (accessruntime.AppliedPolicy, error)
}

type MeterReader interface {
	ReadMeters(context.Context, quotaruntime.MeterReadRequest) (quotaruntime.MeterReadResult, error)
}

type PublicationWaiter interface {
	WaitApplied(context.Context, string, string, uint64) error
}
