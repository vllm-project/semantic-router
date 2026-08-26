package accessruntime

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

func TestCoherentLocatorAcceptsPublicationActivatedAfterInitialDirectoryRead(t *testing.T) {
	const (
		prefix      = "access-reader-test"
		namespaceID = "namespace-1"
		partition   = "partition-1"
		publicID    = "public-key"
	)
	directoryKey, err := quotaruntime.CredentialDirectoryKeyWithPrefix(prefix, string(accesscredential.KindAPIKey), publicID)
	if err != nil {
		t.Fatal(err)
	}
	keys, err := accesspublisher.NewKeyspace(prefix, namespaceID, partition)
	if err != nil {
		t.Fatal(err)
	}
	oldDirectory := credentialDirectoryValues("publication-1", namespaceID, partition, publicID)
	activatedDirectory := cloneStringMap(oldDirectory)
	for field, value := range credentialDirectoryValues("publication-2", namespaceID, partition, publicID) {
		activatedDirectory["pending_"+field] = value
	}
	hashes := &publicationActivationHashReader{
		directoryKey: directoryKey, accessGateKey: keys.AccessGate(), routingGateKey: keys.RoutingGate(),
		initialDirectory: oldDirectory, activatedDirectory: activatedDirectory,
		accessGate:  publicationGateValues("publication-2", 2, false),
		routingGate: publicationGateValues("publication-2", 2, true),
	}
	reader := &RedisProjectionReader{client: hashes, keyPrefix: prefix}

	location, err := reader.LocateCredentialCoherent(context.Background(), accesscredential.KindAPIKey, publicID)
	if err != nil {
		t.Fatalf("LocateCredentialCoherent() = %v", err)
	}
	if location.PublicationID != "publication-2" || location.RuntimeEpoch != 2 ||
		location.RoutingRevision != 2 || location.NamespaceID != namespaceID ||
		location.QuotaPartition != partition {
		t.Fatalf("activated location = %#v", location)
	}
	if hashes.directoryReads != 2 || hashes.accessGateReads != 2 || hashes.routingGateReads != 2 {
		t.Fatalf("coherent read commands directory/access/routing = %d/%d/%d",
			hashes.directoryReads, hashes.accessGateReads, hashes.routingGateReads)
	}
}

func TestCoherentLocatorClassifiesGateChangeDuringReadAsPending(t *testing.T) {
	const (
		prefix      = "access-reader-gate-test"
		namespaceID = "namespace-1"
		partition   = "partition-1"
		publicID    = "public-key"
	)
	directoryKey, err := quotaruntime.CredentialDirectoryKeyWithPrefix(prefix, string(accesscredential.KindAPIKey), publicID)
	if err != nil {
		t.Fatal(err)
	}
	keys, err := accesspublisher.NewKeyspace(prefix, namespaceID, partition)
	if err != nil {
		t.Fatal(err)
	}
	directory := credentialDirectoryValues("publication-1", namespaceID, partition, publicID)
	reader := &RedisProjectionReader{
		client: &publicationActivationHashReader{
			directoryKey: directoryKey, accessGateKey: keys.AccessGate(), routingGateKey: keys.RoutingGate(),
			initialDirectory: directory, activatedDirectory: directory,
			firstAccessGate: publicationGateValues("publication-1", 1, false),
			accessGate:      publicationGateValues("publication-2", 2, false),
			routingGate:     publicationGateValues("publication-2", 2, true),
		},
		keyPrefix: prefix,
	}

	_, err = reader.LocateCredentialCoherent(context.Background(), accesscredential.KindAPIKey, publicID)
	if !errors.Is(err, ErrPublicationPending) {
		t.Fatalf("LocateCredentialCoherent() error = %v, want pending publication", err)
	}
}

func TestLocateCredentialKeepsOneHotPathSnapshot(t *testing.T) {
	const (
		prefix      = "access-reader-hot-path-test"
		namespaceID = "namespace-1"
		partition   = "partition-1"
		publicID    = "public-key"
	)
	directoryKey, err := quotaruntime.CredentialDirectoryKeyWithPrefix(prefix, string(accesscredential.KindAPIKey), publicID)
	if err != nil {
		t.Fatal(err)
	}
	keys, err := accesspublisher.NewKeyspace(prefix, namespaceID, partition)
	if err != nil {
		t.Fatal(err)
	}
	directory := credentialDirectoryValues("publication-2", namespaceID, partition, publicID)
	hashes := &publicationActivationHashReader{
		directoryKey: directoryKey, accessGateKey: keys.AccessGate(), routingGateKey: keys.RoutingGate(),
		initialDirectory: directory, activatedDirectory: directory,
		accessGate:  publicationGateValues("publication-2", 2, false),
		routingGate: publicationGateValues("publication-2", 2, true),
	}
	reader := &RedisProjectionReader{client: hashes, keyPrefix: prefix}

	if _, err := reader.LocateCredential(context.Background(), accesscredential.KindAPIKey, publicID); err != nil {
		t.Fatalf("LocateCredential() = %v", err)
	}
	if hashes.directoryReads != 1 || hashes.accessGateReads != 1 || hashes.routingGateReads != 1 {
		t.Fatalf("hot-path read commands directory/access/routing = %d/%d/%d",
			hashes.directoryReads, hashes.accessGateReads, hashes.routingGateReads)
	}
}

type publicationActivationHashReader struct {
	directoryKey       string
	accessGateKey      string
	routingGateKey     string
	initialDirectory   map[string]string
	activatedDirectory map[string]string
	firstAccessGate    map[string]string
	accessGate         map[string]string
	routingGate        map[string]string
	directoryReads     int
	accessGateReads    int
	routingGateReads   int
}

func (reader *publicationActivationHashReader) HGetAll(
	ctx context.Context,
	key string,
) *redis.MapStringStringCmd {
	command := redis.NewMapStringStringCmd(ctx)
	var values map[string]string
	switch key {
	case reader.directoryKey:
		reader.directoryReads++
		values = reader.initialDirectory
		if reader.directoryReads > 1 {
			values = reader.activatedDirectory
		}
	case reader.accessGateKey:
		reader.accessGateReads++
		values = reader.accessGate
		if reader.accessGateReads == 1 && reader.firstAccessGate != nil {
			values = reader.firstAccessGate
		}
	case reader.routingGateKey:
		reader.routingGateReads++
		values = reader.routingGate
	default:
		command.SetErr(redis.Nil)
		return command
	}
	command.SetVal(cloneStringMap(values))
	return command
}

func credentialDirectoryValues(publicationID, namespaceID, partition, publicID string) map[string]string {
	return map[string]string{
		"publication_id": publicationID,
		"state":          string(accesspublisher.PointerStateActive),
		"partition":      partition,
		"namespace_id":   namespaceID,
		"kind":           string(accesscredential.KindAPIKey),
		"public_id":      publicID,
	}
}

func publicationGateValues(publicationID string, revision uint64, routing bool) map[string]string {
	values := map[string]string{
		"publication_id":     publicationID,
		"revision":           "2",
		"runtime_epoch":      "2",
		"publication_digest": strings.Repeat("a", 64),
	}
	if revision != 2 {
		values["revision"] = "1"
		values["runtime_epoch"] = "1"
	}
	if routing {
		values["snapshot_digest"] = strings.Repeat("b", 64)
		values["snapshot_key"] = "routing:snapshot:2"
	} else {
		values["manifest_digest"] = strings.Repeat("c", 64)
	}
	return values
}

func cloneStringMap(values map[string]string) map[string]string {
	result := make(map[string]string, len(values))
	for key, value := range values {
		result[key] = value
	}
	return result
}
