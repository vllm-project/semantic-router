package accesspublisher

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// ActiveGeneration is the exact coupled access and routing generation that a
// Management publication waiter observed while resolving a credential. The
// waiter may only release that credential while this generation is still the
// active gate on every live namespace replica.
type ActiveGeneration struct {
	PublicationID         string
	Revision              uint64
	RuntimeEpoch          uint64
	RoutingSnapshotDigest string
}

func (generation ActiveGeneration) Validate() error {
	if strings.TrimSpace(generation.PublicationID) == "" || generation.Revision == 0 ||
		generation.RuntimeEpoch == 0 || !validDigest(generation.RoutingSnapshotDigest) {
		return fmt.Errorf("active generation is incomplete")
	}
	return nil
}

// ActiveReplicaStatus is a point-in-time, partition-local observation of the
// live replicas serving one exact generation. An empty fleet never counts as
// ready: Management must not return a credential that no data plane can use.
type ActiveReplicaStatus struct {
	Required []string
	Missing  []string
}

func (status ActiveReplicaStatus) Complete() bool {
	return len(status.Required) > 0 && len(status.Missing) == 0
}

// ActiveReplicaAcknowledgements proves that every currently live namespace
// replica installed expected as its process-local generation. Membership is
// sampled before the script only to supply Redis Cluster with the complete set
// of partition-local registration keys. The script then validates membership,
// coupled gates, Redis TIME, leases, and registrations atomically in one slot.
func (s *RedisStore) ActiveReplicaAcknowledgements(
	ctx context.Context,
	namespaceID string,
	partition string,
	expected ActiveGeneration,
) (ActiveReplicaStatus, error) {
	if s == nil || s.client == nil {
		return ActiveReplicaStatus{}, errors.New("redis publication store is unavailable")
	}
	if err := expected.Validate(); err != nil {
		return ActiveReplicaStatus{}, err
	}
	keys, err := NewKeyspace(s.keyPrefix, namespaceID, partition)
	if err != nil {
		return ActiveReplicaStatus{}, err
	}

	replicas, err := s.client.ZRange(ctx, keys.ReplicaIndex(), 0, -1).Result()
	if err != nil {
		return ActiveReplicaStatus{}, fmt.Errorf("read active replica membership: %w", err)
	}
	replicas = uniqueStrings(replicas)
	scriptKeys := []string{keys.AccessGate(), keys.RoutingGate(), keys.ReplicaIndex()}
	arguments := []any{
		expected.PublicationID,
		strconv.FormatUint(expected.Revision, 10),
		strconv.FormatUint(expected.RuntimeEpoch, 10),
		expected.RoutingSnapshotDigest,
	}
	for _, replicaID := range replicas {
		scriptKeys = append(scriptKeys, keys.Replica(replicaID))
		arguments = append(arguments, replicaID)
	}

	missing, err := activeReplicaReadinessScript.Run(
		ctx, s.client, scriptKeys, arguments...,
	).StringSlice()
	if err != nil {
		return ActiveReplicaStatus{}, classifyRedisPublicationError(err)
	}
	return ActiveReplicaStatus{Required: replicas, Missing: uniqueStrings(missing)}, nil
}
