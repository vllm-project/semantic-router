package accesspublisher

import (
	"encoding/base64"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

// Keyspace is the shared publication/runtime naming contract. All mutable
// publication state for one namespace uses the namespace quota partition hash
// tag. The credential directory is global and may only locate that partition;
// it is never an authorization assertion by itself.
type Keyspace struct {
	prefix      string
	namespaceID string
	partition   string
	tag         string
	quota       quotaruntime.AccessProjectionKeyspace
}

func NewKeyspace(prefix, namespaceID, partition string) (Keyspace, error) {
	if strings.TrimSpace(namespaceID) == "" {
		return Keyspace{}, fmt.Errorf("namespace id is required")
	}
	quotaKeys, err := quotaruntime.NewAccessProjectionKeyspaceWithPrefix(prefix, partition)
	if err != nil {
		return Keyspace{}, err
	}
	return Keyspace{
		prefix: prefix, namespaceID: namespaceID, partition: partition,
		tag: "{" + partition + "}", quota: quotaKeys,
	}, nil
}

func (k Keyspace) local(parts ...string) string {
	value := strings.Join(parts, ":")
	if k.prefix == "" {
		return value
	}
	return k.prefix + ":" + value
}

func encoded(value string) string { return base64.RawURLEncoding.EncodeToString([]byte(value)) }

// namespaceDirectoryKey is the bounded, non-authoritative locator used by
// data-plane replicas to discover namespace publication slots without SCAN.
// Authorization and routing values remain in the namespace partition slot.
func namespaceDirectoryKey(prefix string) string {
	if prefix == "" {
		return "routing:namespace-directory:v1"
	}
	return prefix + ":routing:namespace-directory:v1"
}

// fleetReplicaIndexKey is the global live data-plane membership locator. It
// deliberately has no namespace hash tag: publishers read it before entering
// one namespace's atomic publication slot and pin that membership there.
func fleetReplicaIndexKey(prefix string) string {
	if prefix == "" {
		return "routing:fleet-replicas:v1"
	}
	return prefix + ":routing:fleet-replicas:v1"
}

func (k Keyspace) AccessGate() string {
	return k.local("access", k.tag, "publication-gate", encoded(k.namespaceID))
}

func (k Keyspace) RoutingGate() string {
	return k.local("routing", k.tag, "publication-gate", encoded(k.namespaceID))
}

func (k Keyspace) Publication(publicationID string) string {
	return k.local("access", k.tag, "publication", encoded(k.namespaceID), encoded(publicationID))
}

func (k Keyspace) PublicationBarriers(publicationID string) string {
	return k.Publication(publicationID) + ":barriers"
}

func (k Keyspace) PublicationRequiredReplicas(publicationID string) string {
	return k.Publication(publicationID) + ":required-replicas"
}

func (k Keyspace) PublicationBarrierAcks(publicationID string) string {
	return k.Publication(publicationID) + ":barrier-acks"
}

func (k Keyspace) PublicationRoutingAcks(publicationID string) string {
	return k.Publication(publicationID) + ":routing-acks"
}

func (k Keyspace) PublicationPointers(publicationID string) string {
	return k.Publication(publicationID) + ":pointers"
}

func (k Keyspace) OpenPublications() string {
	return k.local("access", k.tag, "open-publications", encoded(k.namespaceID))
}

func (k Keyspace) PendingPublication() string {
	return k.local("access", k.tag, "pending-publication", encoded(k.namespaceID))
}

func (k Keyspace) ReplicaIndex() string {
	return k.local("routing", k.tag, "replicas", encoded(k.namespaceID))
}

func (k Keyspace) Replica(replicaID string) string {
	return k.local("routing", k.tag, "replica", encoded(k.namespaceID), encoded(replicaID))
}

func (k Keyspace) Manifest(publicationID string) string {
	return k.local("access", k.tag, "manifest", encoded(k.namespaceID), encoded(publicationID))
}

func (k Keyspace) AccessPointer(keyID string) string { return k.quota.Active(keyID) }

func (k Keyspace) LogicalKey(keyID string) string { return k.quota.LogicalKey(keyID) }

func (k Keyspace) AccessDocument(keyID string, revision uint64) string {
	return k.quota.Policy(keyID, fmt.Sprintf("%d", revision))
}

func (k Keyspace) CredentialPointer(kind, publicID string) string {
	return k.quota.Credential(kind, publicID)
}

func (k Keyspace) CredentialDocument(kind, publicID string, revision uint64) string {
	return k.local(
		"access", k.tag, "credential-document", encoded(kind), encoded(publicID), fmt.Sprintf("%d", revision),
	)
}

func (k Keyspace) ProviderCredentialDocument(publicationID, credentialID string) string {
	return k.local(
		"routing", k.tag, "provider-credential", encoded(k.namespaceID),
		encoded(publicationID), encoded(credentialID),
	)
}

func (k Keyspace) CredentialDirectory(kind, publicID string) (string, error) {
	return quotaruntime.CredentialDirectoryKeyWithPrefix(k.prefix, kind, publicID)
}

func (k Keyspace) RoutingSnapshot(revision uint64) string {
	return k.local("routing", k.tag, "snapshot", encoded(k.namespaceID), fmt.Sprintf("%d", revision))
}

func (k Keyspace) Deny(kind, resourceID string) string { return k.quota.Deny(kind, resourceID) }

func (k Keyspace) AppliedRevision() string {
	return k.local("access", k.tag, "applied-revision", encoded(k.namespaceID))
}

func (k Keyspace) RuntimeEpoch() string {
	return k.local("access", k.tag, "runtime-epoch", encoded(k.namespaceID))
}
