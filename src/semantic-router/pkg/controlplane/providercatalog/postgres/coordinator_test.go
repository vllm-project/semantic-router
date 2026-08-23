package postgres

import (
	"bytes"
	"reflect"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
)

func TestClassifyBlockersRejectsMixedLiveCapabilities(t *testing.T) {
	now := time.Date(2026, time.August, 22, 2, 0, 0, 0, time.UTC)
	group := providercatalog.RolloutGroup{
		Plane: providercatalog.CapabilityPlaneData,
		ID:    "router",
	}
	acknowledgement := func(replica string, digest byte) ReplicaAcknowledgement {
		return ReplicaAcknowledgement{
			Revision:         "sha256:" + string(bytes.Repeat([]byte{'a'}, 64)),
			ReplicaID:        replica,
			RolloutGroup:     group,
			CapabilityDigest: bytes.Repeat([]byte{digest}, 32),
			Status:           AckCompatible,
			AcknowledgedAt:   now.Add(-time.Second),
			LeaseExpiresAt:   now.Add(time.Minute),
		}
	}

	blockers := classifyBlockers(
		[]providercatalog.RolloutGroup{group},
		map[string][]ReplicaAcknowledgement{
			group.Key(): {acknowledgement("router-a", 0x11), acknowledgement("router-b", 0x22)},
		},
		now,
	)
	if !reflect.DeepEqual(blockers.Divergent, []providercatalog.RolloutGroup{group}) {
		t.Fatalf("mixed live capability blockers = %+v", blockers)
	}
	if blockers.Empty() {
		t.Fatal("mixed live capabilities did not block activation")
	}

	blockers = classifyBlockers(
		[]providercatalog.RolloutGroup{group},
		map[string][]ReplicaAcknowledgement{
			group.Key(): {acknowledgement("router-a", 0x11), acknowledgement("router-b", 0x11)},
		},
		now,
	)
	if !blockers.Empty() {
		t.Fatalf("homogeneous live capabilities were blocked: %+v", blockers)
	}
}
