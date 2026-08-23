package managedruntime

import (
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/publicationreplica"
)

type routingSnapshotLifecycle interface {
	publicationreplica.SnapshotLifecycle
	backendinvoker.RoutingSnapshotSource
}

// AttachRoutingSnapshots binds the process-owned immutable routing-generation
// lifecycle before Start. The managed runtime owns publication discovery and
// replica leases; the caller owns the generation lifecycle itself so request
// serving can acquire its typed leases without a package-global registry.
func (runtime *Runtime) AttachRoutingSnapshots(snapshots routingSnapshotLifecycle) error {
	if runtime == nil || snapshots == nil {
		return errors.New("routing snapshot lifecycle is required")
	}
	runtime.mu.Lock()
	defer runtime.mu.Unlock()
	if runtime.mode != "managed" {
		return errors.New("routing snapshots require managed control-plane mode")
	}
	if runtime.closed || runtime.started {
		return errors.New("routing snapshots must be attached before startup")
	}
	if runtime.routingReplica != nil {
		return errors.New("routing snapshot lifecycle is already attached")
	}
	if runtime.redis == nil || runtime.replicaID == "" || runtime.accessKeyPrefix == "" ||
		runtime.backendDispatch == nil || runtime.dispatchCapabilities == nil {
		return errors.New("routing publication dependencies are unavailable")
	}
	store, err := accesspublisher.NewRedisStore(accesspublisher.RedisStoreOptions{
		Client: runtime.redis, KeyPrefix: runtime.accessKeyPrefix,
	})
	if err != nil {
		return err
	}
	replica, err := publicationreplica.New(publicationreplica.Options{
		Store: store, Snapshots: snapshots, ReplicaID: runtime.replicaID,
	})
	if err != nil {
		return err
	}
	if err := runtime.dispatchCapabilities.AttachRoutingSnapshots(snapshots); err != nil {
		return err
	}
	if err := runtime.backendDispatch.Attach(
		snapshots,
		runtime.keyrings.ControlPlane.BackendDispatch.Symmetric(),
	); err != nil {
		return err
	}
	runtime.routingReplica = replica
	return nil
}
