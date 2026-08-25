package routingruntime

import (
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/publicationreplica"
)

type routingSnapshotLifecycle interface {
	publicationreplica.SnapshotLifecycle
	backendinvoker.RoutingSnapshotSource
}

// AttachRoutingSnapshots binds the process-owned immutable routing-generation
// lifecycle before Start. The process runtime owns publication discovery and
// replica leases; the caller owns the generation lifecycle itself so request
// serving can acquire its typed leases without a package-global registry.
func (runtime *Runtime) AttachRoutingSnapshots(snapshots routingSnapshotLifecycle) error {
	if runtime == nil || snapshots == nil {
		return errors.New("routing snapshot lifecycle is required")
	}
	runtime.mu.Lock()
	defer runtime.mu.Unlock()
	if !runtime.capabilities.DurableRouting {
		return errors.New("routing snapshots require durable routing")
	}
	if runtime.closed || runtime.started {
		return errors.New("routing snapshots must be attached before startup")
	}
	if runtime.routingReplica != nil {
		return errors.New("routing snapshot lifecycle is already attached")
	}
	if runtime.publicationCoordinator == nil || runtime.replicaID == "" ||
		runtime.backendDispatch == nil || runtime.dispatchCapabilities == nil {
		return errors.New("routing publication dependencies are unavailable")
	}
	replica, err := publicationreplica.New(publicationreplica.Options{
		Store: runtime.publicationCoordinator, Snapshots: snapshots, ReplicaID: runtime.replicaID,
	})
	if err != nil {
		return err
	}
	if err := runtime.dispatchCapabilities.AttachRoutingSnapshots(snapshots); err != nil {
		return err
	}
	if err := runtime.backendDispatch.Attach(
		snapshots,
		runtime.keyrings.Routing.BackendDispatch.Symmetric(),
	); err != nil {
		return err
	}
	runtime.routingReplica = replica
	return nil
}
