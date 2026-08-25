package managementcomposition

import (
	"context"
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// accessRoutingSnapshotReader adapts the routing authoring service to the
// immutable snapshot seam used by key-scoped consumer catalog reads.
type accessRoutingSnapshotReader struct {
	service *routingmanagement.Service
}

func (reader accessRoutingSnapshotReader) ReadRoutingSnapshot(
	ctx context.Context,
	namespaceID string,
	routingRevision int64,
) (*routingsnapshot.Snapshot, error) {
	if reader.service == nil {
		return nil, errors.New("routing snapshot service is unavailable")
	}
	detail, err := reader.service.GetSnapshot(ctx, namespaceID, routingRevision)
	if err != nil {
		return nil, err
	}
	snapshot := detail.Export
	return &snapshot, nil
}

var _ accessmanagement.RoutingSnapshotReader = accessRoutingSnapshotReader{}
