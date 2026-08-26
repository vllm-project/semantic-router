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

// RoutingPublicationPin is the complete coupled generation selected by one
// applied API-key policy. RoutingDocumentDigest identifies the immutable routing
// document used by the data plane; it is deliberately distinct from the
// nested routingsnapshot.Snapshot digest.
type RoutingPublicationPin struct {
	NamespaceID           string
	QuotaPartition        string
	PublicationID         string
	RuntimeEpoch          uint64
	RoutingRevision       int64
	RoutingDocumentDigest string
}

// RoutingPublication is the verified consumer-safe part of one exact runtime
// publication. RoutingDocumentDigest remains the data-plane document digest while
// Snapshot carries its independently verified nested snapshot digest.
type RoutingPublication struct {
	RoutingDocumentDigest string
	Snapshot              routingsnapshot.Snapshot
}

// RoutingPublicationReader resolves only the exact active runtime publication
// named by a key policy. It must not fall back to a newer authoring snapshot.
type RoutingPublicationReader interface {
	ReadRoutingPublication(context.Context, RoutingPublicationPin) (*RoutingPublication, error)
}

type MeterReader interface {
	ReadMeters(context.Context, quotaruntime.MeterReadRequest) (quotaruntime.MeterReadResult, error)
}

type PublicationWaiter interface {
	WaitApplied(context.Context, string, string, uint64) error
}
