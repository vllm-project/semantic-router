package accesspublisher

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
)

const (
	defaultReplicaLease     = 30 * time.Second
	defaultMaxNamespaces    = 100_000
	maximumNamespaceEntries = 1_000_000
)

type RedisStoreOptions struct {
	Client               redis.UniversalClient
	KeyPrefix            string
	ReplicaLease         time.Duration
	MaxNamespaces        int
	RequireFleetReplicas bool
}

// RedisStore publishes immutable access and routing documents and coordinates
// their activation. It never reads or writes authoritative management state.
// All authorization facts originate in PostgreSQL and arrive here through a
// compiled Publication.
type RedisStore struct {
	client               redis.UniversalClient
	keyPrefix            string
	replicaLease         time.Duration
	maxNamespaces        int
	requireFleetReplicas bool
}

var _ RuntimeStore = (*RedisStore)(nil)

func NewRedisStore(options RedisStoreOptions) (*RedisStore, error) {
	if options.Client == nil {
		return nil, fmt.Errorf("redis publication client is required")
	}
	if _, err := NewKeyspace(options.KeyPrefix, "validation", "validation"); err != nil {
		return nil, fmt.Errorf("redis publication key prefix: %w", err)
	}
	lease := options.ReplicaLease
	if lease == 0 {
		lease = defaultReplicaLease
	}
	if lease < time.Second || lease > 5*time.Minute {
		return nil, fmt.Errorf("replica lease must be between one second and five minutes")
	}
	maxNamespaces := options.MaxNamespaces
	if maxNamespaces == 0 {
		maxNamespaces = defaultMaxNamespaces
	}
	if maxNamespaces < 1 || maxNamespaces > maximumNamespaceEntries {
		return nil, fmt.Errorf("maximum namespaces must be between 1 and %d", maximumNamespaceEntries)
	}
	return &RedisStore{
		client: options.Client, keyPrefix: options.KeyPrefix, replicaLease: lease,
		maxNamespaces: maxNamespaces, requireFleetReplicas: options.RequireFleetReplicas,
	}, nil
}

type storedPlan struct {
	Previous         *Manifest `json:"previous,omitempty"`
	Barriers         []Barrier `json:"barriers"`
	Supersedes       []string  `json:"supersedes"`
	PriorAccessGate  string    `json:"priorAccessGate"`
	PriorRoutingGate string    `json:"priorRoutingGate"`
}

func (s *RedisStore) Prepare(ctx context.Context, publication Publication) (PublicationPlan, error) {
	if err := publication.Validate(); err != nil {
		return PublicationPlan{}, err
	}
	keys, prepareErr := NewKeyspace(s.keyPrefix, publication.NamespaceID, publication.QuotaPartition)
	if prepareErr != nil {
		return PublicationPlan{}, prepareErr
	}
	accessGate, routingGate, prepareErr := s.readGatePair(ctx, keys)
	if prepareErr != nil {
		return PublicationPlan{}, prepareErr
	}
	if accessGate != routingGate {
		return PublicationPlan{}, fmt.Errorf("%w: access and routing gates disagree", ErrStagedCorrupt)
	}
	storedValues, prepareErr := s.client.HGetAll(ctx, keys.Publication(publication.ID)).Result()
	if prepareErr != nil {
		return PublicationPlan{}, fmt.Errorf("read prepared publication: %w", prepareErr)
	}
	if len(storedValues) > 0 {
		if storedValues["publication_digest"] != publication.Digest ||
			storedValues["namespace_id"] != publication.NamespaceID ||
			storedValues["quota_partition"] != publication.QuotaPartition ||
			storedValues["desired_revision"] != strconv.FormatUint(publication.DesiredRevision, 10) ||
			storedValues["runtime_epoch"] != strconv.FormatUint(publication.RuntimeEpoch, 10) {
			return PublicationPlan{}, fmt.Errorf("%w: prepared publication identity changed", ErrConflict)
		}
		var stored storedPlan
		if err := decodeStrict([]byte(storedValues["plan"]), &stored); err != nil {
			return PublicationPlan{}, fmt.Errorf("%w: prepared publication plan is invalid", ErrStagedCorrupt)
		}
		if (accessGate != stored.PriorAccessGate && accessGate != publication.ID) ||
			(routingGate != stored.PriorRoutingGate && routingGate != publication.ID) {
			return PublicationPlan{}, ErrSuperseded
		}
		return PublicationPlan{
			Publication: publication, Previous: stored.Previous, Barriers: stored.Barriers,
			Supersedes: stored.Supersedes, PriorAccessGate: stored.PriorAccessGate,
			PriorRoutingGate: stored.PriorRoutingGate,
		}, nil
	}
	previous, documents, prepareErr := s.readPrevious(ctx, keys, accessGate)
	if prepareErr != nil {
		return PublicationPlan{}, prepareErr
	}
	barriers, prepareErr := Diff(documents, publication)
	if prepareErr != nil {
		return PublicationPlan{}, fmt.Errorf("compile publication barriers: %w", prepareErr)
	}
	supersedes, prepareErr := s.client.ZRangeByScore(ctx, keys.OpenPublications(), &redis.ZRangeBy{
		Min: "-inf", Max: strconv.FormatUint(publication.DesiredRevision, 10),
	}).Result()
	if prepareErr != nil {
		return PublicationPlan{}, fmt.Errorf("read open publications: %w", prepareErr)
	}
	supersedes = withoutString(uniqueStrings(supersedes), publication.ID)
	stored := storedPlan{
		Previous: previous, Barriers: barriers, Supersedes: supersedes,
		PriorAccessGate: accessGate, PriorRoutingGate: routingGate,
	}
	planJSON, prepareErr := json.Marshal(stored)
	if prepareErr != nil {
		return PublicationPlan{}, fmt.Errorf("encode publication plan: %w", prepareErr)
	}
	priorPublicationKey := keys.Publication(accessGate)
	if accessGate == "" {
		priorPublicationKey = keys.Publication(publication.ID)
	}
	_, prepareErr = preparePublicationScript.Run(ctx, s.client, []string{
		keys.RuntimeEpoch(), keys.AccessGate(), keys.RoutingGate(), keys.Publication(publication.ID),
		keys.OpenPublications(), priorPublicationKey, keys.PendingPublication(),
	}, publication.RuntimeEpoch, accessGate, routingGate, publication.ID, publication.Digest,
		publication.NamespaceID, publication.QuotaPartition, publication.DesiredRevision,
		publication.Manifest.Digest, publication.Routing.Digest, string(planJSON)).Result()
	if prepareErr != nil {
		return PublicationPlan{}, classifyRedisPublicationError(prepareErr)
	}
	return PublicationPlan{
		Publication: publication, Previous: previous, Barriers: barriers, Supersedes: supersedes,
		PriorAccessGate: accessGate, PriorRoutingGate: routingGate,
	}, nil
}

func (s *RedisStore) readGatePair(ctx context.Context, keys Keyspace) (string, string, error) {
	pipeline := s.client.Pipeline()
	access := pipeline.HGet(ctx, keys.AccessGate(), "publication_id")
	routing := pipeline.HGet(ctx, keys.RoutingGate(), "publication_id")
	_, err := pipeline.Exec(ctx)
	if err != nil && !errors.Is(err, redis.Nil) {
		return "", "", fmt.Errorf("read publication gates: %w", err)
	}
	return nilString(access.Val()), nilString(routing.Val()), nil
}

func (s *RedisStore) readPrevious(
	ctx context.Context,
	keys Keyspace,
	publicationID string,
) (*Manifest, PreviousDocuments, error) {
	if publicationID == "" {
		return nil, PreviousDocuments{}, nil
	}
	payload, bytesErr := s.client.Get(ctx, keys.Manifest(publicationID)).Bytes()
	if errors.Is(bytesErr, redis.Nil) {
		return nil, PreviousDocuments{}, fmt.Errorf("%w: active manifest is absent", ErrStagedCorrupt)
	}
	if bytesErr != nil {
		return nil, PreviousDocuments{}, fmt.Errorf("read active manifest: %w", bytesErr)
	}
	var manifest Manifest
	if err := decodeStrict(payload, &manifest); err != nil {
		return nil, PreviousDocuments{}, fmt.Errorf("%w: decode active manifest: %w", ErrStagedCorrupt, err)
	}
	if manifest.PublicationID != publicationID || manifest.NamespaceID != keys.namespaceID ||
		manifest.QuotaPartition != keys.partition {
		return nil, PreviousDocuments{}, fmt.Errorf("%w: active manifest identity disagrees with gate", ErrStagedCorrupt)
	}
	if err := verifyManifest(manifest); err != nil {
		return nil, PreviousDocuments{}, err
	}
	documents := PreviousDocuments{
		Manifest: &manifest, Access: make(map[string]AccessDocument, len(manifest.Access)),
		Credentials:         make(map[string]CredentialDocument, len(manifest.Credentials)),
		ProviderCredentials: make(map[string]ProviderCredentialDocument, len(manifest.ProviderCredentials)),
	}
	for _, keyID := range sortedMapKeys(manifest.Access) {
		entry := manifest.Access[keyID]
		values, err := s.client.HGetAll(ctx, keys.AccessDocument(keyID, entry.Revision)).Result()
		if err != nil {
			return nil, PreviousDocuments{}, fmt.Errorf("read active access document: %w", err)
		}
		if values["digest"] != entry.Digest || values["document"] == "" {
			return nil, PreviousDocuments{}, fmt.Errorf("%w: active access document %s disagrees with manifest", ErrStagedCorrupt, keyID)
		}
		var projection accessprojection.Projection
		if err := decodeStrict([]byte(values["document"]), &projection); err != nil {
			return nil, PreviousDocuments{}, fmt.Errorf("%w: decode access document %s", ErrStagedCorrupt, keyID)
		}
		if err := projection.VerifyDigest(entry.Digest); err != nil {
			return nil, PreviousDocuments{}, fmt.Errorf("%w: verify access document %s: %w", ErrStagedCorrupt, keyID, err)
		}
		documents.Access[keyID] = AccessDocument{
			NamespaceID: manifest.NamespaceID, QuotaPartition: manifest.QuotaPartition,
			DesiredRevision: entry.Revision, KeyID: keyID, Projection: projection, Digest: entry.Digest,
		}
	}
	for _, identity := range sortedMapKeys(manifest.Credentials) {
		entry := manifest.Credentials[identity]
		kind, publicID, ok := strings.Cut(identity, ":")
		if !ok || kind == "" || publicID == "" {
			return nil, PreviousDocuments{}, fmt.Errorf("%w: credential identity %q is invalid", ErrStagedCorrupt, identity)
		}
		payload, err := s.client.Get(ctx, keys.CredentialDocument(kind, publicID, entry.Revision)).Bytes()
		if errors.Is(err, redis.Nil) {
			return nil, PreviousDocuments{}, fmt.Errorf("%w: active credential document %s is absent", ErrStagedCorrupt, identity)
		}
		if err != nil {
			return nil, PreviousDocuments{}, fmt.Errorf("read active credential document: %w", err)
		}
		var document CredentialDocument
		if err := decodeStrict(payload, &document); err != nil || document.Digest != entry.Digest {
			return nil, PreviousDocuments{}, fmt.Errorf("%w: active credential document %s is invalid", ErrStagedCorrupt, identity)
		}
		if err := verifyCredentialDocument(document); err != nil {
			return nil, PreviousDocuments{}, err
		}
		documents.Credentials[identity] = document
	}
	for _, credentialID := range sortedMapKeys(manifest.ProviderCredentials) {
		entry := manifest.ProviderCredentials[credentialID]
		document, err := s.loadProviderCredentialDocument(ctx, keys, manifest.PublicationID, credentialID, entry)
		if err != nil {
			return nil, PreviousDocuments{}, err
		}
		documents.ProviderCredentials[credentialID] = document
	}
	routingPayload, bytesErr := s.client.Get(ctx, keys.RoutingSnapshot(manifest.DesiredRevision)).Bytes()
	if errors.Is(bytesErr, redis.Nil) {
		return nil, PreviousDocuments{}, fmt.Errorf("%w: active routing document is absent", ErrStagedCorrupt)
	}
	if bytesErr != nil {
		return nil, PreviousDocuments{}, fmt.Errorf("read active routing document: %w", bytesErr)
	}
	var routing RoutingDocument
	if err := decodeStrict(routingPayload, &routing); err != nil || routing.Digest != manifest.RoutingDigest {
		return nil, PreviousDocuments{}, fmt.Errorf("%w: active routing document is invalid", ErrStagedCorrupt)
	}
	if err := verifyRoutingDocument(routing); err != nil {
		return nil, PreviousDocuments{}, err
	}
	documents.Routing = &routing
	return &manifest, documents, nil
}

func (s *RedisStore) InstallBarriers(ctx context.Context, plan PublicationPlan) error {
	if err := validatePlan(plan); err != nil {
		return err
	}
	keys, _ := NewKeyspace(s.keyPrefix, plan.Publication.NamespaceID, plan.Publication.QuotaPartition)
	for _, barrier := range plan.Barriers {
		denyKey := keys.Deny(barrier.Kind, barrier.ResourceID)
		if _, err := installOneBarrierScript.Run(ctx, s.client,
			[]string{denyKey, keys.PublicationBarriers(plan.Publication.ID)}, plan.Publication.ID).Result(); err != nil {
			return fmt.Errorf("install deny barrier: %w", err)
		}
	}
	_, err := installBarriersScript.Run(ctx, s.client, []string{
		keys.Publication(plan.Publication.ID), keys.ReplicaIndex(),
		keys.PublicationRequiredReplicas(plan.Publication.ID),
	}, len(plan.Barriers)).Result()
	if err != nil {
		return classifyRedisPublicationError(err)
	}
	return nil
}

func (s *RedisStore) Stage(ctx context.Context, plan PublicationPlan) error {
	if err := validatePlan(plan); err != nil {
		return err
	}
	var fleetReplicas []string
	if s.requireFleetReplicas {
		var err error
		fleetReplicas, err = s.liveFleetReplicas(ctx)
		if err != nil {
			return err
		}
		if len(fleetReplicas) == 0 {
			return ErrAcknowledgements
		}
	}
	p := plan.Publication
	keys, _ := NewKeyspace(s.keyPrefix, p.NamespaceID, p.QuotaPartition)
	if err := s.stagePublicationDocuments(ctx, keys, p); err != nil {
		return err
	}
	if err := s.stageAccessDocuments(ctx, keys, plan); err != nil {
		return err
	}
	if err := s.stageCredentialDocuments(ctx, keys, plan); err != nil {
		return err
	}
	if err := s.finalizeRedisStage(ctx, keys, plan, fleetReplicas); err != nil {
		return err
	}
	return s.registerNamespace(ctx, p.NamespaceID, p.QuotaPartition)
}

func (s *RedisStore) stagePublicationDocuments(ctx context.Context, keys Keyspace, p Publication) error {
	manifestPayload, stageErr := json.Marshal(p.Manifest)
	if stageErr != nil {
		return stageErr
	}
	if err := s.putImmutableString(ctx, keys.Manifest(p.ID), manifestPayload); err != nil {
		return fmt.Errorf("stage manifest: %w", err)
	}
	routingPayload, stageErr := json.Marshal(p.Routing)
	if stageErr != nil {
		return stageErr
	}
	if err := s.putImmutableString(ctx, keys.RoutingSnapshot(p.DesiredRevision), routingPayload); err != nil {
		return fmt.Errorf("stage routing snapshot: %w", err)
	}
	for _, document := range p.ProviderCredentials {
		payload, err := json.Marshal(document)
		if err != nil {
			return err
		}
		if err := s.putImmutableString(ctx,
			keys.ProviderCredentialDocument(p.ID, document.Credential.ID), payload); err != nil {
			return fmt.Errorf("stage provider credential document: %w", err)
		}
	}
	return nil
}

func (s *RedisStore) stageAccessDocuments(ctx context.Context, keys Keyspace, plan PublicationPlan) error {
	p := plan.Publication
	currentAccess := make(map[string]struct{}, len(p.Access))
	for _, document := range p.Access {
		currentAccess[document.KeyID] = struct{}{}
		projectionPayload, err := json.Marshal(document.Projection)
		if err != nil {
			return err
		}
		if _, err := putImmutableHashScript.Run(ctx, s.client,
			[]string{keys.AccessDocument(document.KeyID, document.DesiredRevision)},
			document.Digest, string(projectionPayload), p.NamespaceID, p.ID, document.DesiredRevision).Result(); err != nil {
			return classifyRedisPublicationError(err)
		}
		if err := s.stagePointer(ctx, keys, "access", keys.AccessPointer(document.KeyID), accessPointerFields(p, document)); err != nil {
			return err
		}
		if err := s.stagePointer(ctx, keys, "logical", keys.LogicalKey(document.KeyID), logicalKeyFields(p, document)); err != nil {
			return err
		}
	}
	if plan.Previous != nil {
		for keyID := range plan.Previous.Access {
			if _, exists := currentAccess[keyID]; exists {
				continue
			}
			fields := tombstoneFields(p)
			if err := s.stagePointer(ctx, keys, "access", keys.AccessPointer(keyID), fields); err != nil {
				return err
			}
			if err := s.stagePointer(ctx, keys, "logical", keys.LogicalKey(keyID), fields); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *RedisStore) stageCredentialDocuments(ctx context.Context, keys Keyspace, plan PublicationPlan) error {
	p := plan.Publication
	currentCredentials := make(map[string]struct{}, len(p.Credentials))
	for _, document := range p.Credentials {
		identity := credentialIdentity(document.Kind, document.PublicID)
		currentCredentials[identity] = struct{}{}
		payload, stageErr2 := json.Marshal(document)
		if stageErr2 != nil {
			return stageErr2
		}
		if err := s.putImmutableString(ctx,
			keys.CredentialDocument(document.Kind, document.PublicID, document.DesiredRevision), payload); err != nil {
			return fmt.Errorf("stage credential document: %w", err)
		}
		if err := s.stagePointer(ctx, keys, "credential", keys.CredentialPointer(document.Kind, document.PublicID), credentialPointerFields(p, document)); err != nil {
			return err
		}
		directoryKey, stageErr2 := keys.CredentialDirectory(document.Kind, document.PublicID)
		if stageErr2 != nil {
			return stageErr2
		}
		if err := s.stagePointer(ctx, keys, "directory", directoryKey, directoryFields(p, document)); err != nil {
			return err
		}
	}
	if plan.Previous != nil {
		for identity := range plan.Previous.Credentials {
			if _, exists := currentCredentials[identity]; exists {
				continue
			}
			kind, publicID, ok := strings.Cut(identity, ":")
			if !ok {
				return fmt.Errorf("%w: invalid prior credential identity", ErrStagedCorrupt)
			}
			fields := tombstoneFields(p)
			if err := s.stagePointer(ctx, keys, "credential", keys.CredentialPointer(kind, publicID), fields); err != nil {
				return err
			}
			directoryKey, err := keys.CredentialDirectory(kind, publicID)
			if err != nil {
				return err
			}
			if err := s.stagePointer(ctx, keys, "directory", directoryKey, fields); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *RedisStore) finalizeRedisStage(
	ctx context.Context,
	keys Keyspace,
	plan PublicationPlan,
	fleetReplicas []string,
) error {
	p := plan.Publication
	pointerCount, stageErr := s.client.ZCard(ctx, keys.PublicationPointers(p.ID)).Result()
	if stageErr != nil {
		return fmt.Errorf("count staged pointers: %w", stageErr)
	}
	arguments := make([]any, 0, 2+len(fleetReplicas))
	arguments = append(arguments, pointerCount, boolDigit(plan.Restrictive()))
	for _, replicaID := range fleetReplicas {
		arguments = append(arguments, replicaID)
	}
	_, stageErr = finalizeStageScript.Run(ctx, s.client, []string{
		keys.Publication(p.ID), keys.ReplicaIndex(), keys.PublicationRequiredReplicas(p.ID),
	}, arguments...).Result()
	if stageErr != nil {
		return classifyRedisPublicationError(stageErr)
	}
	return nil
}

func (s *RedisStore) putImmutableString(ctx context.Context, key string, payload []byte) error {
	_, err := putImmutableStringScript.Run(ctx, s.client, []string{key}, string(payload)).Result()
	return classifyRedisPublicationError(err)
}

type pointerReference struct {
	Kind string `json:"kind"`
	Key  string `json:"key"`
}

func (s *RedisStore) stagePointer(ctx context.Context, keys Keyspace, kind, key string, fields map[string]string) error {
	arguments := make([]any, 0, 1+2*len(fields))
	arguments = append(arguments, fields["pending_publication_id"])
	for _, field := range sortedMapKeys(fields) {
		arguments = append(arguments, field, fields[field])
	}
	if _, err := stagePointerScript.Run(ctx, s.client, []string{key}, arguments...).Result(); err != nil {
		return classifyRedisPublicationError(err)
	}
	reference, _ := json.Marshal(pointerReference{Kind: kind, Key: key})
	member := base64.RawURLEncoding.EncodeToString(reference)
	if err := s.client.ZAdd(ctx, keys.PublicationPointers(fields["pending_publication_id"]),
		redis.Z{Score: 0, Member: member}).Err(); err != nil {
		return fmt.Errorf("index staged pointer: %w", err)
	}
	return nil
}

func accessPointerFields(publication Publication, document AccessDocument) map[string]string {
	return map[string]string{
		"pending_publication_id": publication.ID, "pending_state": "active",
		"pending_revision": strconv.FormatUint(document.DesiredRevision, 10), "pending_digest": document.Digest,
	}
}

func logicalKeyFields(publication Publication, document AccessDocument) map[string]string {
	projection := document.Projection
	fields := map[string]string{
		"pending_publication_id": publication.ID, "pending_state": "active",
		"pending_revision":         strconv.FormatUint(document.DesiredRevision, 10),
		"pending_status":           string(projection.KeyStatus),
		"pending_policy_epoch":     strconv.FormatUint(projection.PolicyEpoch, 10),
		"pending_delegation_epoch": strconv.FormatUint(projection.DelegationEpoch, 10),
	}
	if projection.KeyExpiresAt != nil {
		fields["pending_expires_at_ms"] = strconv.FormatInt(projection.KeyExpiresAt.UnixMilli(), 10)
	}
	return fields
}

func credentialPointerFields(publication Publication, document CredentialDocument) map[string]string {
	projection := document.Projection
	fields := map[string]string{
		"pending_publication_id": publication.ID, "pending_state": "active",
		"pending_revision": strconv.FormatUint(document.DesiredRevision, 10),
		"pending_kind":     projection.Kind,
		"pending_kid":      projection.KID, "pending_key_id": projection.KeyID,
		"pending_secret_hmac":    base64.RawURLEncoding.EncodeToString(projection.SecretHMAC),
		"pending_pepper_version": projection.PepperVersion, "pending_status": projection.Status,
		"pending_not_before_ms": strconv.FormatInt(projection.NotBefore.UnixMilli(), 10),
		"pending_digest":        document.Digest,
	}
	if projection.ExpiresAt != nil {
		fields["pending_expires_at_ms"] = strconv.FormatInt(projection.ExpiresAt.UnixMilli(), 10)
	}
	if projection.Kind == CredentialKindDelegation {
		fields["pending_management_session_id"] = projection.ManagementSessionID
		fields["pending_principal_id"] = projection.PrincipalID
		fields["pending_delegation_epoch"] = strconv.FormatUint(projection.DelegationEpoch, 10)
		fields["pending_user_id"] = projection.UserID
		fields["pending_team_id"] = projection.TeamID
		fields["pending_audience"] = projection.Audience
	}
	return fields
}

func directoryFields(publication Publication, document CredentialDocument) map[string]string {
	return map[string]string{
		"pending_publication_id": publication.ID, "pending_state": "active",
		"pending_revision":  strconv.FormatUint(document.DesiredRevision, 10),
		"pending_partition": publication.QuotaPartition, "pending_namespace_id": publication.NamespaceID,
		"pending_kind": document.Kind, "pending_public_id": document.PublicID,
	}
}

func tombstoneFields(publication Publication) map[string]string {
	return map[string]string{
		"pending_publication_id": publication.ID, "pending_state": "tombstone",
		"pending_revision": strconv.FormatUint(publication.DesiredRevision, 10),
	}
}

func validatePlan(plan PublicationPlan) error {
	if err := plan.Publication.Validate(); err != nil {
		return err
	}
	barriers, err := canonicalBarriers(plan.Barriers)
	if err != nil {
		return err
	}
	if !equalBarriers(barriers, plan.Barriers) {
		return fmt.Errorf("publication plan barriers are not canonical")
	}
	return nil
}

func equalBarriers(left, right []Barrier) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func classifyRedisPublicationError(err error) error {
	if err == nil {
		return nil
	}
	message := err.Error()
	switch {
	case strings.Contains(message, "EPOCH_MISMATCH"):
		return ErrEpochMismatch
	case strings.Contains(message, "HEAD_SUPERSEDED"):
		return ErrSuperseded
	case strings.Contains(message, "ACK_INCOMPLETE"), strings.Contains(message, "REPLICA_LEASE_EXPIRED"),
		strings.Contains(message, "NO_ACTIVE_REPLICAS"):
		return ErrAcknowledgements
	case strings.Contains(message, "EXPECTED_PUBLICATION_CHANGED"), strings.Contains(message, "ACTIVE_MEMBERSHIP_CHANGED"):
		return ErrPublicationChanged
	case strings.Contains(message, "NAMESPACE_DIRECTORY_FULL"):
		return ErrDirectoryFull
	case strings.Contains(message, "IMMUTABLE_CONFLICT"), strings.Contains(message, "POINTER_CONFLICT"),
		strings.Contains(message, "GATE_CONFLICT"), strings.Contains(message, "PUBLICATION_CONFLICT"),
		strings.Contains(message, "NAMESPACE_PARTITION_CONFLICT"),
		strings.Contains(message, "PRIOR_NOT_COMPACTED"), strings.Contains(message, "REVISION_CONFLICT"),
		strings.Contains(message, "STATE_CONFLICT"), strings.Contains(message, "BARRIERS_REQUIRED"):
		return fmt.Errorf("%w: Redis publication compare-and-set failed", ErrConflict)
	case strings.Contains(message, "NOT_VALIDATED"), strings.Contains(message, "VALIDATION_CONFLICT"),
		strings.Contains(message, "POINTER_STATE_INVALID"), strings.Contains(message, "ACTIVE_GATE_CORRUPT"),
		strings.Contains(message, "ACTIVE_READINESS_INPUT_INVALID"):
		return fmt.Errorf("%w: Redis staged publication failed validation", ErrStagedCorrupt)
	default:
		return err
	}
}

func decodeStrict(payload []byte, destination any) error {
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return fmt.Errorf("trailing JSON data")
	}
	return nil
}

func nilString(value string) string { return value }

func boolDigit(value bool) string {
	if value {
		return "1"
	}
	return "0"
}

func uniqueStrings(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}

func withoutString(values []string, excluded string) []string {
	result := values[:0]
	for _, value := range values {
		if value != excluded {
			result = append(result, value)
		}
	}
	return result
}
