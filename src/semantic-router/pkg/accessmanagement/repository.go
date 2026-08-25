package accessmanagement

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
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

// RoutingSnapshotReader returns the immutable Router snapshot pinned by an
// applied key policy. Management clients never read mutable authoring rows to
// derive a consumer-visible catalog.
type RoutingSnapshotReader interface {
	ReadRoutingSnapshot(context.Context, string, int64) (*routingsnapshot.Snapshot, error)
}

type MeterReader interface {
	ReadMeters(context.Context, quotaruntime.MeterReadRequest) (quotaruntime.MeterReadResult, error)
}

type PublicationWaiter interface {
	WaitApplied(context.Context, string, string, uint64) error
}
