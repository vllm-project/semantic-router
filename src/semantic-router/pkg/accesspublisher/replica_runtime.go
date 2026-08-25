package accesspublisher

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// NamespacePublication locates one namespace's partition-local publication
// state. It is discovery metadata only and grants no access by itself.
type NamespacePublication struct {
	NamespaceID    string
	QuotaPartition string
}

func (n NamespacePublication) Validate() error {
	if strings.TrimSpace(n.NamespaceID) == "" || strings.TrimSpace(n.QuotaPartition) == "" {
		return fmt.Errorf("namespace and quota partition are required")
	}
	if strings.ContainsRune(n.NamespaceID, 0) || strings.ContainsRune(n.QuotaPartition, 0) {
		return fmt.Errorf("namespace or quota partition contains NUL")
	}
	return nil
}

// PublicationRuntimeState is the immutable publication's lifecycle state in
// the runtime store. Replicas only load candidates after validation.
type PublicationRuntimeState string

const (
	PublicationStatePrepared          PublicationRuntimeState = "prepared"
	PublicationStateBarriersInstalled PublicationRuntimeState = "barriers_installed"
	PublicationStateStaged            PublicationRuntimeState = "staged"
	PublicationStateValidated         PublicationRuntimeState = "validated"
	PublicationStateActive            PublicationRuntimeState = "active"
	PublicationStateCompacted         PublicationRuntimeState = "compacted"
	PublicationStateApplied           PublicationRuntimeState = "applied"
	PublicationStateFinalized         PublicationRuntimeState = "finalized"
)

// RuntimePublicationIdentity pins every value a replica must compare before
// it acknowledges or serves a routing generation.
type RuntimePublicationIdentity struct {
	PublicationID     string
	NamespaceID       string
	QuotaPartition    string
	DesiredRevision   uint64
	RuntimeEpoch      uint64
	PublicationDigest string
	ManifestDigest    string
	RoutingDigest     string
	State             PublicationRuntimeState
	Restrictive       bool
}

func (p RuntimePublicationIdentity) Validate() error {
	if strings.TrimSpace(p.PublicationID) == "" || strings.TrimSpace(p.NamespaceID) == "" ||
		strings.TrimSpace(p.QuotaPartition) == "" || p.DesiredRevision == 0 || p.RuntimeEpoch == 0 {
		return fmt.Errorf("runtime publication identity is incomplete")
	}
	if !validDigest(p.PublicationDigest) || !validDigest(p.ManifestDigest) || !validDigest(p.RoutingDigest) {
		return fmt.Errorf("runtime publication digest is invalid")
	}
	return nil
}

func (p RuntimePublicationIdentity) SameGeneration(other RuntimePublicationIdentity) bool {
	return p.PublicationID == other.PublicationID && p.NamespaceID == other.NamespaceID &&
		p.QuotaPartition == other.QuotaPartition && p.DesiredRevision == other.DesiredRevision &&
		p.RuntimeEpoch == other.RuntimeEpoch && p.PublicationDigest == other.PublicationDigest &&
		p.ManifestDigest == other.ManifestDigest && p.RoutingDigest == other.RoutingDigest
}

func (p RuntimePublicationIdentity) Loadable() bool {
	switch p.State {
	case PublicationStateValidated, PublicationStateActive, PublicationStateCompacted,
		PublicationStateApplied, PublicationStateFinalized:
		return true
	default:
		return false
	}
}

func (p RuntimePublicationIdentity) Activated() bool {
	switch p.State {
	case PublicationStateActive, PublicationStateCompacted, PublicationStateApplied, PublicationStateFinalized:
		return true
	default:
		return false
	}
}

// PublicationHeads is a bounded control read. Active is the coupled access and
// routing gate. Candidate is the newest staged head and remains invisible to
// inference until Active changes to the same identity.
type PublicationHeads struct {
	Namespace NamespacePublication
	Active    *RuntimePublicationIdentity
	Candidate *RuntimePublicationIdentity
}

// LoadedRoutingPublication is a strictly decoded and digest-verified runtime
// generation. Snapshot contains rebuilt lookup indexes and is ready for a
// loader to compile or warm, independent of the durable-store implementation.
type LoadedRoutingPublication struct {
	Identity RuntimePublicationIdentity
	Manifest Manifest
	Routing  RoutingDocument
	Snapshot routingsnapshot.Snapshot
}

func (s *RedisStore) registerNamespace(ctx context.Context, namespaceID, partition string) error {
	reference := NamespacePublication{NamespaceID: namespaceID, QuotaPartition: partition}
	if err := reference.Validate(); err != nil {
		return err
	}
	_, err := registerNamespaceScript.Run(ctx, s.client, []string{namespaceDirectoryKey(s.keyPrefix)},
		encoded(namespaceID), encoded(partition), s.maxNamespaces).Result()
	return classifyRedisPublicationError(err)
}

// ListPublicationNamespaces reads the bounded locator in one operation. It
// intentionally never uses SCAN and never treats directory entries as policy.
func (s *RedisStore) ListPublicationNamespaces(ctx context.Context) ([]NamespacePublication, error) {
	values, err := s.client.HGetAll(ctx, namespaceDirectoryKey(s.keyPrefix)).Result()
	if err != nil {
		return nil, fmt.Errorf("read publication namespace directory: %w", err)
	}
	result := make([]NamespacePublication, 0, len(values))
	seenNamespaces := make(map[string]struct{}, len(values))
	for encodedNamespace, encodedPartition := range values {
		namespace, err := base64.RawURLEncoding.DecodeString(encodedNamespace)
		if err != nil {
			return nil, fmt.Errorf("%w: namespace directory identity is invalid", ErrStagedCorrupt)
		}
		partition, err := base64.RawURLEncoding.DecodeString(encodedPartition)
		if err != nil {
			return nil, fmt.Errorf("%w: namespace directory partition is invalid", ErrStagedCorrupt)
		}
		reference := NamespacePublication{NamespaceID: string(namespace), QuotaPartition: string(partition)}
		if err := reference.Validate(); err != nil {
			return nil, fmt.Errorf("%w: namespace directory: %w", ErrStagedCorrupt, err)
		}
		if _, exists := seenNamespaces[reference.NamespaceID]; exists {
			return nil, fmt.Errorf("%w: namespace directory contains a duplicate namespace", ErrStagedCorrupt)
		}
		seenNamespaces[reference.NamespaceID] = struct{}{}
		result = append(result, reference)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].NamespaceID < result[j].NamespaceID })
	return result, nil
}

// CountPublicationNamespaces returns a bounded directory cardinality without
// transferring every namespace locator to the caller.
func (s *RedisStore) CountPublicationNamespaces(ctx context.Context) (int64, error) {
	if s == nil || s.client == nil {
		return 0, errors.New("redis publication store is unavailable")
	}
	count, err := s.client.HLen(ctx, namespaceDirectoryKey(s.keyPrefix)).Result()
	if err != nil {
		return 0, fmt.Errorf("count publication namespace directory: %w", err)
	}
	return count, nil
}

// GetPublicationNamespace resolves one exact namespace locator in constant
// space. The locator is discovery metadata and grants no authority.
func (s *RedisStore) GetPublicationNamespace(
	ctx context.Context,
	namespaceID string,
) (NamespacePublication, error) {
	if s == nil || s.client == nil {
		return NamespacePublication{}, errors.New("redis publication store is unavailable")
	}
	if strings.TrimSpace(namespaceID) == "" || strings.ContainsRune(namespaceID, 0) {
		return NamespacePublication{}, fmt.Errorf("namespace is invalid")
	}
	encodedPartition, err := s.client.HGet(
		ctx,
		namespaceDirectoryKey(s.keyPrefix),
		encoded(namespaceID),
	).Result()
	if errors.Is(err, redis.Nil) {
		return NamespacePublication{}, ErrNamespaceNotFound
	}
	if err != nil {
		return NamespacePublication{}, fmt.Errorf("read publication namespace directory: %w", err)
	}
	partition, err := base64.RawURLEncoding.DecodeString(encodedPartition)
	if err != nil {
		return NamespacePublication{}, fmt.Errorf("%w: namespace directory partition is invalid", ErrStagedCorrupt)
	}
	reference := NamespacePublication{NamespaceID: namespaceID, QuotaPartition: string(partition)}
	if err := reference.Validate(); err != nil {
		return NamespacePublication{}, fmt.Errorf("%w: namespace directory: %w", ErrStagedCorrupt, err)
	}
	return reference, nil
}

type rawPublicationHeads struct {
	access  map[string]string
	routing map[string]string
	pending map[string]string
}

func (s *RedisStore) readRawPublicationHeads(
	ctx context.Context,
	keys Keyspace,
) (rawPublicationHeads, error) {
	pipeline := s.client.Pipeline()
	access := pipeline.HGetAll(ctx, keys.AccessGate())
	routing := pipeline.HGetAll(ctx, keys.RoutingGate())
	pending := pipeline.HGetAll(ctx, keys.PendingPublication())
	_, err := pipeline.Exec(ctx)
	if err != nil && !errors.Is(err, redis.Nil) {
		return rawPublicationHeads{}, fmt.Errorf("read publication heads: %w", err)
	}
	return rawPublicationHeads{access: access.Val(), routing: routing.Val(), pending: pending.Val()}, nil
}

// ReadPublicationHeads returns a self-consistent active/candidate observation.
// A concurrent activation is reported as ErrPublicationChanged so a replica
// retries instead of acting on a mixed generation.
func (s *RedisStore) ReadPublicationHeads(
	ctx context.Context,
	reference NamespacePublication,
) (PublicationHeads, error) {
	if err := reference.Validate(); err != nil {
		return PublicationHeads{}, err
	}
	keys, readPublicationHeadsErr := NewKeyspace(s.keyPrefix, reference.NamespaceID, reference.QuotaPartition)
	if readPublicationHeadsErr != nil {
		return PublicationHeads{}, readPublicationHeadsErr
	}
	before, readPublicationHeadsErr := s.readRawPublicationHeads(ctx, keys)
	if readPublicationHeadsErr != nil {
		return PublicationHeads{}, readPublicationHeadsErr
	}
	heads := PublicationHeads{Namespace: reference}
	accessPresent, routingPresent := len(before.access) != 0, len(before.routing) != 0
	if accessPresent != routingPresent {
		return PublicationHeads{}, fmt.Errorf("%w: access and routing publication gates disagree", ErrStagedCorrupt)
	}
	if accessPresent {
		accessGate, err := ParsePublicationGate(before.access)
		if err != nil {
			return PublicationHeads{}, fmt.Errorf("%w: active access gate: %w", ErrStagedCorrupt, err)
		}
		routingGate, err := ParsePublicationGate(before.routing)
		if err != nil {
			return PublicationHeads{}, fmt.Errorf("%w: active routing gate: %w", ErrStagedCorrupt, err)
		}
		if accessGate.PublicationID != routingGate.PublicationID || accessGate.Revision != routingGate.Revision ||
			accessGate.RuntimeEpoch != routingGate.RuntimeEpoch ||
			accessGate.PublicationDigest != routingGate.PublicationDigest {
			return PublicationHeads{}, fmt.Errorf("%w: active access and routing gates are not coupled", ErrStagedCorrupt)
		}
		identity, err := s.readRuntimePublicationIdentity(ctx, keys, accessGate.PublicationID)
		if err != nil {
			return PublicationHeads{}, err
		}
		if identity.DesiredRevision != accessGate.Revision || identity.RuntimeEpoch != accessGate.RuntimeEpoch ||
			identity.PublicationDigest != accessGate.PublicationDigest ||
			identity.ManifestDigest != accessGate.ManifestDigest || identity.RoutingDigest != routingGate.SnapshotDigest ||
			!identity.Activated() {
			return PublicationHeads{}, fmt.Errorf("%w: active gate disagrees with publication metadata", ErrStagedCorrupt)
		}
		heads.Active = &identity
	}
	if len(before.pending) != 0 {
		publicationID := before.pending[FieldPublicationID]
		revision, parseErr := strconv.ParseUint(before.pending["revision"], 10, 64)
		if publicationID == "" || revision == 0 || parseErr != nil || !validDigest(before.pending["digest"]) {
			return PublicationHeads{}, fmt.Errorf("%w: candidate publication head is invalid", ErrStagedCorrupt)
		}
		identity, err := s.readRuntimePublicationIdentity(ctx, keys, publicationID)
		if err != nil {
			return PublicationHeads{}, err
		}
		if identity.DesiredRevision != revision || identity.PublicationDigest != before.pending["digest"] {
			return PublicationHeads{}, fmt.Errorf("%w: candidate head disagrees with publication metadata", ErrStagedCorrupt)
		}
		if heads.Active == nil || !identity.SameGeneration(*heads.Active) {
			heads.Candidate = &identity
		}
	}
	after, readPublicationHeadsErr := s.readRawPublicationHeads(ctx, keys)
	if readPublicationHeadsErr != nil {
		return PublicationHeads{}, readPublicationHeadsErr
	}
	if !reflect.DeepEqual(before, after) {
		return PublicationHeads{}, ErrPublicationChanged
	}
	return heads, nil
}

func (s *RedisStore) readRuntimePublicationIdentity(
	ctx context.Context,
	keys Keyspace,
	publicationID string,
) (RuntimePublicationIdentity, error) {
	values, err := s.client.HGetAll(ctx, keys.Publication(publicationID)).Result()
	if err != nil {
		return RuntimePublicationIdentity{}, fmt.Errorf("read runtime publication metadata: %w", err)
	}
	if len(values) == 0 {
		return RuntimePublicationIdentity{}, fmt.Errorf("%w: runtime publication metadata is absent", ErrStagedCorrupt)
	}
	revision, revisionErr := strconv.ParseUint(values["desired_revision"], 10, 64)
	epoch, epochErr := strconv.ParseUint(values["runtime_epoch"], 10, 64)
	barrierCount := uint64(0)
	if values["barrier_count"] != "" {
		barrierCount, err = strconv.ParseUint(values["barrier_count"], 10, 64)
		if err != nil {
			return RuntimePublicationIdentity{}, fmt.Errorf("%w: publication barrier count is invalid", ErrStagedCorrupt)
		}
	}
	identity := RuntimePublicationIdentity{
		PublicationID: publicationID, NamespaceID: values["namespace_id"], QuotaPartition: values["quota_partition"],
		DesiredRevision: revision, RuntimeEpoch: epoch, PublicationDigest: values["publication_digest"],
		ManifestDigest: values["manifest_digest"], RoutingDigest: values["routing_digest"],
		State: PublicationRuntimeState(values["state"]), Restrictive: barrierCount > 0,
	}
	if revisionErr != nil || epochErr != nil || values["publication_id"] != publicationID ||
		identity.NamespaceID != keys.namespaceID || identity.QuotaPartition != keys.partition {
		return RuntimePublicationIdentity{}, fmt.Errorf("%w: runtime publication identity is invalid", ErrStagedCorrupt)
	}
	if err := identity.Validate(); err != nil {
		return RuntimePublicationIdentity{}, fmt.Errorf("%w: %w", ErrStagedCorrupt, err)
	}
	return identity, nil
}

// LoadRoutingPublication decodes exactly the immutable manifest and routing
// document named by identity, verifies every digest and cross-reference, and
// rebuilds the routing snapshot indexes used by the request path.
func (s *RedisStore) LoadRoutingPublication(
	ctx context.Context,
	identity RuntimePublicationIdentity,
) (LoadedRoutingPublication, error) {
	if err := identity.Validate(); err != nil {
		return LoadedRoutingPublication{}, err
	}
	if !identity.Loadable() {
		return LoadedRoutingPublication{}, fmt.Errorf("%w: runtime publication is not validated", ErrNotReady)
	}
	keys, loadRoutingPublicationErr := NewKeyspace(s.keyPrefix, identity.NamespaceID, identity.QuotaPartition)
	if loadRoutingPublicationErr != nil {
		return LoadedRoutingPublication{}, loadRoutingPublicationErr
	}
	stored, loadRoutingPublicationErr := s.readRuntimePublicationIdentity(ctx, keys, identity.PublicationID)
	if loadRoutingPublicationErr != nil {
		return LoadedRoutingPublication{}, loadRoutingPublicationErr
	}
	if !stored.SameGeneration(identity) || !stored.Loadable() {
		return LoadedRoutingPublication{}, fmt.Errorf("%w: publication metadata changed before load", ErrPublicationChanged)
	}
	pipeline := s.client.Pipeline()
	manifestCommand := pipeline.Get(ctx, keys.Manifest(identity.PublicationID))
	routingCommand := pipeline.Get(ctx, keys.RoutingSnapshot(identity.DesiredRevision))
	_, loadRoutingPublicationErr = pipeline.Exec(ctx)
	if loadRoutingPublicationErr != nil && !errors.Is(loadRoutingPublicationErr, redis.Nil) {
		return LoadedRoutingPublication{}, fmt.Errorf("read routing publication: %w", loadRoutingPublicationErr)
	}
	manifestPayload, manifestErr := manifestCommand.Bytes()
	routingPayload, routingErr := routingCommand.Bytes()
	if manifestErr != nil || routingErr != nil {
		return LoadedRoutingPublication{}, fmt.Errorf("%w: immutable manifest or routing document is absent", ErrStagedCorrupt)
	}
	var manifest Manifest
	if err := decodeStrict(manifestPayload, &manifest); err != nil {
		return LoadedRoutingPublication{}, fmt.Errorf("%w: decode immutable manifest: %w", ErrStagedCorrupt, err)
	}
	if err := verifyManifest(manifest); err != nil {
		return LoadedRoutingPublication{}, err
	}
	var routing RoutingDocument
	if err := decodeStrict(routingPayload, &routing); err != nil {
		return LoadedRoutingPublication{}, fmt.Errorf("%w: decode immutable routing document: %w", ErrStagedCorrupt, err)
	}
	if err := verifyRoutingDocument(routing); err != nil {
		return LoadedRoutingPublication{}, err
	}
	if manifest.PublicationID != identity.PublicationID || manifest.NamespaceID != identity.NamespaceID ||
		manifest.QuotaPartition != identity.QuotaPartition || manifest.DesiredRevision != identity.DesiredRevision ||
		manifest.RuntimeEpoch != identity.RuntimeEpoch || manifest.Digest != identity.ManifestDigest ||
		manifest.RoutingDigest != identity.RoutingDigest || routing.NamespaceID != identity.NamespaceID ||
		routing.DesiredRevision != identity.DesiredRevision || routing.Digest != identity.RoutingDigest ||
		!reflect.DeepEqual(manifest.RoutingResources, routing.ResourceDigests) {
		return LoadedRoutingPublication{}, fmt.Errorf("%w: manifest and routing publication identities disagree", ErrStagedCorrupt)
	}
	references := providerCredentialReferences(routing.Snapshot)
	if len(references) != len(manifest.ProviderCredentials) {
		return LoadedRoutingPublication{}, fmt.Errorf("%w: routing and provider credential manifest differ", ErrStagedCorrupt)
	}
	providerDocuments := make(map[string]ProviderCredentialDocument, len(references))
	for _, credentialID := range sortedMapKeys(manifest.ProviderCredentials) {
		document, err := s.loadProviderCredentialDocument(
			ctx, keys, identity.PublicationID, credentialID, manifest.ProviderCredentials[credentialID],
		)
		if err != nil {
			return LoadedRoutingPublication{}, err
		}
		if _, referenced := references[credentialID]; !referenced {
			return LoadedRoutingPublication{}, fmt.Errorf("%w: provider credential is not referenced by routing", ErrStagedCorrupt)
		}
		providerDocuments[credentialID] = document
	}
	for _, model := range routing.Snapshot.Models {
		for _, backend := range model.Backends {
			if backend.ProviderCredentialID == "" {
				continue
			}
			document := providerDocuments[backend.ProviderCredentialID]
			if document.Credential.ProviderID != backend.ProviderID ||
				document.Credential.NormalizedOrigin != backend.Origin {
				return LoadedRoutingPublication{}, fmt.Errorf("%w: provider credential binding differs from routing", ErrStagedCorrupt)
			}
		}
	}
	compiled, loadRoutingPublicationErr := routingsnapshot.Compile(routing.Snapshot.Bundle)
	if loadRoutingPublicationErr != nil || compiled.Digest != routing.Snapshot.Digest ||
		compiled.SemanticDigest != routing.Snapshot.SemanticDigest {
		return LoadedRoutingPublication{}, fmt.Errorf("%w: routing snapshot cannot be rebuilt", ErrStagedCorrupt)
	}
	routing.Snapshot = *compiled
	return LoadedRoutingPublication{
		Identity: identity, Manifest: manifest, Routing: routing, Snapshot: *compiled,
	}, nil
}
