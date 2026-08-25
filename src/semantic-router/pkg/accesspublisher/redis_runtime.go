package accesspublisher

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
)

func (s *RedisStore) ValidateStaged(ctx context.Context, plan PublicationPlan) error {
	if err := validatePlan(plan); err != nil {
		return err
	}
	p := plan.Publication
	if err := verifyPublication(p); err != nil {
		return fmt.Errorf("%w: %w", ErrStagedCorrupt, err)
	}
	keys, _ := NewKeyspace(s.keyPrefix, p.NamespaceID, p.QuotaPartition)
	if err := s.validateStagedPublication(ctx, keys, p); err != nil {
		return err
	}
	currentAccess, err := s.validateStagedAccess(ctx, keys, plan)
	if err != nil {
		return err
	}
	currentCredentials, err := s.validateStagedCredentials(ctx, keys, plan)
	if err != nil {
		return err
	}
	if err := s.validateStagedPointerInventory(ctx, keys, plan, currentAccess, currentCredentials); err != nil {
		return err
	}
	_, validateStagedErr := markValidatedScript.Run(ctx, s.client, []string{keys.Publication(p.ID)}, p.Digest).Result()
	if validateStagedErr != nil {
		return classifyRedisPublicationError(validateStagedErr)
	}
	return nil
}

func (s *RedisStore) validateStagedPublication(ctx context.Context, keys Keyspace, p Publication) error {
	manifestPayload, validateStagedErr := s.client.Get(ctx, keys.Manifest(p.ID)).Bytes()
	if validateStagedErr != nil {
		return stagedReadError("manifest", validateStagedErr)
	}
	var manifest Manifest
	if err := decodeStrict(manifestPayload, &manifest); err != nil || manifest.Digest != p.Manifest.Digest {
		return fmt.Errorf("%w: staged manifest differs from compiled publication", ErrStagedCorrupt)
	}
	if err := verifyManifest(manifest); err != nil {
		return err
	}
	routingPayload, validateStagedErr := s.client.Get(ctx, keys.RoutingSnapshot(p.DesiredRevision)).Bytes()
	if validateStagedErr != nil {
		return stagedReadError("routing snapshot", validateStagedErr)
	}
	var routing RoutingDocument
	if err := decodeStrict(routingPayload, &routing); err != nil || routing.Digest != p.Routing.Digest {
		return fmt.Errorf("%w: staged routing snapshot differs from compiled publication", ErrStagedCorrupt)
	}
	if err := verifyRoutingDocument(routing); err != nil {
		return err
	}
	for _, document := range p.ProviderCredentials {
		entry, exists := manifest.ProviderCredentials[document.Credential.ID]
		if !exists || entry.Digest != document.Digest || entry.Revision != p.DesiredRevision {
			return fmt.Errorf("%w: staged provider credential manifest entry differs", ErrStagedCorrupt)
		}
		stored, err := s.loadProviderCredentialDocument(ctx, keys, p.ID, document.Credential.ID, entry)
		if err != nil {
			return err
		}
		if stored.Digest != document.Digest {
			return fmt.Errorf("%w: staged provider credential %s changed", ErrStagedCorrupt, document.Credential.ID)
		}
	}
	return nil
}

func (s *RedisStore) validateStagedAccess(
	ctx context.Context,
	keys Keyspace,
	plan PublicationPlan,
) (map[string]struct{}, error) {
	p := plan.Publication
	currentAccess := make(map[string]struct{}, len(p.Access))
	for _, document := range p.Access {
		currentAccess[document.KeyID] = struct{}{}
		values, err := s.client.HGetAll(ctx, keys.AccessDocument(document.KeyID, document.DesiredRevision)).Result()
		if err != nil {
			return nil, stagedReadError("access document", err)
		}
		if values["digest"] != document.Digest || values["publication_id"] != p.ID || values["document"] == "" {
			return nil, fmt.Errorf("%w: staged access document %s is incomplete", ErrStagedCorrupt, document.KeyID)
		}
		var projectionDocument json.RawMessage
		if err := json.Unmarshal([]byte(values["document"]), &projectionDocument); err != nil {
			return nil, fmt.Errorf("%w: staged access document %s is invalid", ErrStagedCorrupt, document.KeyID)
		}
		compiled, _ := json.Marshal(document.Projection)
		if string(projectionDocument) != string(compiled) {
			return nil, fmt.Errorf("%w: staged access document %s changed", ErrStagedCorrupt, document.KeyID)
		}
		if err := document.Projection.VerifyDigest(document.Digest); err != nil {
			return nil, fmt.Errorf("%w: staged access document %s failed digest verification", ErrStagedCorrupt, document.KeyID)
		}
		if err := s.validatePointer(ctx, keys, keys.AccessPointer(document.KeyID), accessPointerFields(p, document)); err != nil {
			return nil, fmt.Errorf("access pointer %s: %w", document.KeyID, err)
		}
		if err := s.validatePointer(ctx, keys, keys.LogicalKey(document.KeyID), logicalKeyFields(p, document)); err != nil {
			return nil, fmt.Errorf("logical key %s: %w", document.KeyID, err)
		}
	}
	if plan.Previous != nil {
		for keyID := range plan.Previous.Access {
			if _, exists := currentAccess[keyID]; exists {
				continue
			}
			if err := s.validatePointer(ctx, keys, keys.AccessPointer(keyID), tombstoneFields(p)); err != nil {
				return nil, fmt.Errorf("removed access pointer %s: %w", keyID, err)
			}
			if err := s.validatePointer(ctx, keys, keys.LogicalKey(keyID), tombstoneFields(p)); err != nil {
				return nil, fmt.Errorf("removed logical key %s: %w", keyID, err)
			}
		}
	}
	return currentAccess, nil
}

func (s *RedisStore) validateStagedCredentials(
	ctx context.Context,
	keys Keyspace,
	plan PublicationPlan,
) (map[string]struct{}, error) {
	p := plan.Publication
	currentCredentials := make(map[string]struct{}, len(p.Credentials))
	for _, document := range p.Credentials {
		identity := credentialIdentity(document.Kind, document.PublicID)
		currentCredentials[identity] = struct{}{}
		payload, err := s.client.Get(ctx,
			keys.CredentialDocument(document.Kind, document.PublicID, document.DesiredRevision)).Bytes()
		if err != nil {
			return nil, stagedReadError("credential document", err)
		}
		var stored CredentialDocument
		if err := decodeStrict(payload, &stored); err != nil || stored.Digest != document.Digest {
			return nil, fmt.Errorf("%w: staged credential document %s changed", ErrStagedCorrupt, identity)
		}
		if err := verifyCredentialDocument(stored); err != nil {
			return nil, err
		}
		if err := s.validatePointer(ctx, keys,
			keys.CredentialPointer(document.Kind, document.PublicID), credentialPointerFields(p, document)); err != nil {
			return nil, fmt.Errorf("credential pointer %s: %w", identity, err)
		}
		directory, _ := keys.CredentialDirectory(document.Kind, document.PublicID)
		if err := s.validatePointer(ctx, keys, directory, directoryFields(p, document)); err != nil {
			return nil, fmt.Errorf("credential directory %s: %w", identity, err)
		}
	}
	if plan.Previous != nil {
		for identity := range plan.Previous.Credentials {
			if _, exists := currentCredentials[identity]; exists {
				continue
			}
			kind, publicID, ok := strings.Cut(identity, ":")
			if !ok {
				return nil, fmt.Errorf("%w: invalid prior credential identity", ErrStagedCorrupt)
			}
			if err := s.validatePointer(ctx, keys, keys.CredentialPointer(kind, publicID), tombstoneFields(p)); err != nil {
				return nil, fmt.Errorf("removed credential pointer %s: %w", identity, err)
			}
			directory, _ := keys.CredentialDirectory(kind, publicID)
			if err := s.validatePointer(ctx, keys, directory, tombstoneFields(p)); err != nil {
				return nil, fmt.Errorf("removed credential directory %s: %w", identity, err)
			}
		}
	}
	return currentCredentials, nil
}

func (s *RedisStore) validateStagedPointerInventory(
	ctx context.Context,
	keys Keyspace,
	plan PublicationPlan,
	currentAccess, currentCredentials map[string]struct{},
) error {
	p := plan.Publication
	indexed, validateStagedErr := s.client.ZCard(ctx, keys.PublicationPointers(p.ID)).Result()
	if validateStagedErr != nil {
		return fmt.Errorf("read staged pointer inventory: %w", validateStagedErr)
	}
	wantPointers := int64(2*len(p.Access) + 2*len(p.Credentials))
	if plan.Previous != nil {
		for keyID := range plan.Previous.Access {
			if _, exists := currentAccess[keyID]; !exists {
				wantPointers += 2
			}
		}
		for identity := range plan.Previous.Credentials {
			if _, exists := currentCredentials[identity]; !exists {
				wantPointers += 2
			}
		}
	}
	if indexed != wantPointers {
		return fmt.Errorf("%w: staged pointer inventory has %d entries, want %d", ErrStagedCorrupt, indexed, wantPointers)
	}
	return nil
}

func (s *RedisStore) validatePointer(ctx context.Context, keys Keyspace, key string, expected map[string]string) error {
	values, err := s.client.HGetAll(ctx, key).Result()
	if err != nil {
		return stagedReadError("pointer", err)
	}
	if len(values) == 0 {
		if expected["pending_state"] == "tombstone" {
			gate, gateErr := s.client.HGet(ctx, keys.AccessGate(), "publication_id").Result()
			if gateErr == nil && gate == expected["pending_publication_id"] {
				return nil
			}
		}
		return fmt.Errorf("%w: staged pointer is absent", ErrStagedCorrupt)
	}
	publicationID := expected["pending_publication_id"]
	if values["pending_publication_id"] == publicationID {
		for field, value := range expected {
			if values[field] != value {
				return fmt.Errorf("%w: pending field %s differs", ErrStagedCorrupt, field)
			}
		}
		return nil
	}
	// Validation is idempotent after activation and compaction. In that case all
	// pending_ fields have been promoted to their unprefixed names.
	if values["publication_id"] == publicationID {
		for field, value := range expected {
			if values[strings.TrimPrefix(field, "pending_")] != value {
				return fmt.Errorf("%w: promoted field %s differs", ErrStagedCorrupt, field)
			}
		}
		return nil
	}
	return fmt.Errorf("%w: pointer publication differs", ErrStagedCorrupt)
}

func stagedReadError(kind string, err error) error {
	if errors.Is(err, redis.Nil) {
		return fmt.Errorf("%w: staged %s is absent", ErrStagedCorrupt, kind)
	}
	return fmt.Errorf("read staged %s: %w", kind, err)
}

// RegisterFleetReplica creates or renews this Router process's global
// Redis-TIME lease. The fleet index contains only liveness, never namespace
// authorization or publication state.
func (s *RedisStore) RegisterFleetReplica(ctx context.Context, replicaID string) (time.Time, error) {
	if strings.TrimSpace(replicaID) == "" || len(replicaID) > 256 || strings.ContainsRune(replicaID, 0) {
		return time.Time{}, fmt.Errorf("replica id is required and must not exceed 256 bytes")
	}
	value, err := registerFleetReplicaScript.Run(
		ctx, s.client, []string{fleetReplicaIndexKey(s.keyPrefix)}, replicaID, s.replicaLease.Milliseconds(),
	).Text()
	if err != nil {
		return time.Time{}, fmt.Errorf("register fleet replica: %w", err)
	}
	milliseconds, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return time.Time{}, fmt.Errorf("redis fleet replica lease timestamp is invalid")
	}
	return time.UnixMilli(milliseconds).UTC(), nil
}

func (s *RedisStore) liveFleetReplicas(ctx context.Context) ([]string, error) {
	replicas, err := liveFleetReplicasScript.Run(
		ctx, s.client, []string{fleetReplicaIndexKey(s.keyPrefix)},
	).StringSlice()
	if err != nil {
		return nil, fmt.Errorf("read live fleet replicas: %w", err)
	}
	return uniqueStrings(replicas), nil
}

// RegisterReplica creates or renews a Redis-TIME lease only when the replica's
// loaded access and routing publications match both active gates.
func (s *RedisStore) RegisterReplica(
	ctx context.Context,
	namespaceID, partition string,
	registration ReplicaRegistration,
) (time.Time, error) {
	if err := registration.Validate(); err != nil {
		return time.Time{}, err
	}
	keys, err := NewKeyspace(s.keyPrefix, namespaceID, partition)
	if err != nil {
		return time.Time{}, err
	}
	value, err := registerReplicaScript.Run(ctx, s.client, []string{
		keys.RuntimeEpoch(), keys.AccessGate(), keys.RoutingGate(),
		keys.Replica(registration.ReplicaID), keys.ReplicaIndex(),
	}, registration.ReplicaID, registration.RuntimeEpoch, registration.AccessPublication,
		registration.RoutingPublication, s.replicaLease.Milliseconds()).Text()
	if err != nil {
		return time.Time{}, classifyRedisPublicationError(err)
	}
	milliseconds, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return time.Time{}, fmt.Errorf("redis replica lease timestamp is invalid")
	}
	return time.UnixMilli(milliseconds).UTC(), nil
}

// AcknowledgeBarriers records that a live replica has observed every barrier
// installed for publicationID. The digest prevents acknowledging a reused or
// corrupted publication identity.
func (s *RedisStore) AcknowledgeBarriers(
	ctx context.Context,
	namespaceID, partition, replicaID, publicationID, publicationDigest string,
) error {
	return s.acknowledge(ctx, namespaceID, partition, replicaID, publicationID, publicationDigest, true)
}

// AcknowledgeRouting records that a live replica loaded the candidate routing
// snapshot and its coupled access publication.
func (s *RedisStore) AcknowledgeRouting(
	ctx context.Context,
	namespaceID, partition, replicaID, publicationID, publicationDigest string,
) error {
	return s.acknowledge(ctx, namespaceID, partition, replicaID, publicationID, publicationDigest, false)
}

func (s *RedisStore) acknowledge(
	ctx context.Context,
	namespaceID, partition, replicaID, publicationID, publicationDigest string,
	barrier bool,
) error {
	if strings.TrimSpace(replicaID) == "" || strings.TrimSpace(publicationID) == "" || !validDigest(publicationDigest) {
		return fmt.Errorf("replica, publication, and digest are required")
	}
	keys, err := NewKeyspace(s.keyPrefix, namespaceID, partition)
	if err != nil {
		return err
	}
	ackKey := keys.PublicationRoutingAcks(publicationID)
	if barrier {
		ackKey = keys.PublicationBarrierAcks(publicationID)
	}
	_, err = acknowledgeScript.Run(ctx, s.client, []string{
		keys.ReplicaIndex(), keys.Publication(publicationID), ackKey,
		keys.PublicationRequiredReplicas(publicationID),
	}, replicaID, publicationID, publicationDigest).Result()
	return classifyRedisPublicationError(err)
}

func (s *RedisStore) BarrierAcknowledgements(ctx context.Context, plan PublicationPlan) (AckStatus, error) {
	return s.acknowledgements(ctx, plan, true)
}

func (s *RedisStore) RoutingAcknowledgements(ctx context.Context, plan PublicationPlan) (AckStatus, error) {
	return s.acknowledgements(ctx, plan, false)
}

func (s *RedisStore) acknowledgements(ctx context.Context, plan PublicationPlan, barrier bool) (AckStatus, error) {
	if err := validatePlan(plan); err != nil {
		return AckStatus{}, err
	}
	p := plan.Publication
	keys, _ := NewKeyspace(s.keyPrefix, p.NamespaceID, p.QuotaPartition)
	now, resultErr := s.client.Time(ctx).Result()
	if resultErr != nil {
		return AckStatus{}, fmt.Errorf("read Redis time: %w", resultErr)
	}
	nowMillis := now.UnixMilli()
	if err := s.client.ZRemRangeByScore(ctx, keys.ReplicaIndex(), "-inf", strconv.FormatInt(nowMillis, 10)).Err(); err != nil {
		return AckStatus{}, fmt.Errorf("expire replica leases: %w", err)
	}
	active, resultErr := s.client.ZRange(ctx, keys.ReplicaIndex(), 0, -1).Result()
	if resultErr != nil {
		return AckStatus{}, fmt.Errorf("read active replicas: %w", resultErr)
	}
	required := active
	if s.requireFleetReplicas {
		fleet, err := s.liveFleetReplicas(ctx)
		if err != nil {
			return AckStatus{}, err
		}
		required = append(required, fleet...)
	}
	required = uniqueStrings(required)
	if s.requireFleetReplicas && len(required) == 0 {
		return AckStatus{}, ErrAcknowledgements
	}
	members := make([]any, len(required))
	for index := range required {
		members[index] = required[index]
	}
	if _, err := replaceRequiredReplicasScript.Run(
		ctx, s.client, []string{keys.PublicationRequiredReplicas(p.ID)}, members...,
	).Result(); err != nil {
		return AckStatus{}, fmt.Errorf("record required replicas: %w", err)
	}
	ackKey := keys.PublicationRoutingAcks(p.ID)
	if barrier {
		ackKey = keys.PublicationBarrierAcks(p.ID)
	}
	acknowledged, resultErr := s.client.SMembers(ctx, ackKey).Result()
	if resultErr != nil {
		return AckStatus{}, fmt.Errorf("read publication acknowledgements: %w", resultErr)
	}
	acked := make(map[string]struct{}, len(acknowledged))
	for _, replica := range acknowledged {
		acked[replica] = struct{}{}
	}
	status := AckStatus{Required: required}
	for _, replica := range status.Required {
		if _, exists := acked[replica]; !exists {
			status.Missing = append(status.Missing, replica)
		}
	}
	return status, nil
}

func (s *RedisStore) Activate(ctx context.Context, plan PublicationPlan) error {
	if err := validatePlan(plan); err != nil {
		return err
	}
	p := plan.Publication
	keys, _ := NewKeyspace(s.keyPrefix, p.NamespaceID, p.QuotaPartition)
	_, err := activatePublicationScript.Run(ctx, s.client, []string{
		keys.RuntimeEpoch(), keys.AccessGate(), keys.RoutingGate(), keys.Publication(p.ID),
		keys.ReplicaIndex(), keys.PublicationRequiredReplicas(p.ID),
		keys.PublicationBarrierAcks(p.ID), keys.PublicationRoutingAcks(p.ID), keys.PendingPublication(),
	}, p.RuntimeEpoch, plan.PriorAccessGate, plan.PriorRoutingGate, p.ID, p.Digest,
		p.DesiredRevision, p.Manifest.Digest, p.Routing.Digest,
		keys.RoutingSnapshot(p.DesiredRevision), boolDigit(plan.Restrictive())).Result()
	if err != nil {
		return classifyRedisPublicationError(err)
	}
	return s.registerNamespace(ctx, p.NamespaceID, p.QuotaPartition)
}

func (s *RedisStore) Compact(ctx context.Context, plan PublicationPlan, batchSize int) (bool, error) {
	if err := validatePlan(plan); err != nil {
		return false, err
	}
	if batchSize <= 0 || batchSize > 1000 {
		return false, fmt.Errorf("compaction batch size must be between 1 and 1000")
	}
	p := plan.Publication
	keys, _ := NewKeyspace(s.keyPrefix, p.NamespaceID, p.QuotaPartition)
	accessGate, routingGate, compactErr := s.readGatePair(ctx, keys)
	if compactErr != nil {
		return false, compactErr
	}
	if accessGate != p.ID || routingGate != p.ID {
		return false, ErrConflict
	}
	cursorText, compactErr := s.client.HGet(ctx, keys.Publication(p.ID), "compact_cursor").Result()
	if errors.Is(compactErr, redis.Nil) {
		return false, fmt.Errorf("%w: publication compaction cursor is absent", ErrStagedCorrupt)
	}
	if compactErr != nil {
		return false, fmt.Errorf("read publication compaction cursor: %w", compactErr)
	}
	cursor, compactErr := strconv.ParseInt(cursorText, 10, 64)
	if compactErr != nil || cursor < 0 {
		return false, fmt.Errorf("%w: publication compaction cursor is invalid", ErrStagedCorrupt)
	}
	members, compactErr := s.client.ZRange(ctx, keys.PublicationPointers(p.ID), cursor, cursor+int64(batchSize)-1).Result()
	if compactErr != nil {
		return false, fmt.Errorf("read publication pointer batch: %w", compactErr)
	}
	for _, member := range members {
		decoded, err := base64.RawURLEncoding.DecodeString(member)
		if err != nil {
			return false, fmt.Errorf("%w: pointer inventory member is invalid", ErrStagedCorrupt)
		}
		var reference pointerReference
		if err := decodeStrict(decoded, &reference); err != nil || reference.Kind == "" || reference.Key == "" {
			return false, fmt.Errorf("%w: pointer inventory reference is invalid", ErrStagedCorrupt)
		}
		if _, err := promotePointerScript.Run(ctx, s.client, []string{reference.Key}, p.ID).Result(); err != nil {
			return false, classifyRedisPublicationError(err)
		}
	}
	if len(members) > 0 {
		cursor += int64(len(members))
		if err := s.client.HSet(ctx, keys.Publication(p.ID), "compact_cursor", cursor).Err(); err != nil {
			return false, fmt.Errorf("advance publication compaction cursor: %w", err)
		}
	}
	total, compactErr := s.client.ZCard(ctx, keys.PublicationPointers(p.ID)).Result()
	if compactErr != nil {
		return false, fmt.Errorf("count publication pointers: %w", compactErr)
	}
	if cursor < total {
		return false, nil
	}
	if _, err := finishCompactionScript.Run(ctx, s.client, []string{
		keys.Publication(p.ID), keys.AccessGate(), keys.RoutingGate(),
	}, p.ID).Result(); err != nil {
		return false, classifyRedisPublicationError(err)
	}
	return true, nil
}

func (s *RedisStore) MarkApplied(ctx context.Context, plan PublicationPlan) error {
	if err := validatePlan(plan); err != nil {
		return err
	}
	p := plan.Publication
	keys, _ := NewKeyspace(s.keyPrefix, p.NamespaceID, p.QuotaPartition)
	_, err := markAppliedScript.Run(ctx, s.client, []string{
		keys.RuntimeEpoch(), keys.AccessGate(), keys.RoutingGate(),
		keys.Publication(p.ID), keys.AppliedRevision(),
	}, p.RuntimeEpoch, p.ID, p.DesiredRevision, p.NamespaceID, p.Digest, p.Routing.Digest).Result()
	return classifyRedisPublicationError(err)
}

func (s *RedisStore) ReconcileApplied(ctx context.Context, applied AppliedState) error {
	if strings.TrimSpace(applied.NamespaceID) == "" || strings.TrimSpace(applied.QuotaPartition) == "" ||
		applied.RuntimeEpoch == 0 || applied.DesiredRevision == 0 {
		return fmt.Errorf("applied namespace, partition, epoch, and revision are required")
	}
	keys, err := NewKeyspace(s.keyPrefix, applied.NamespaceID, applied.QuotaPartition)
	if err != nil {
		return err
	}
	accessValues, err := s.client.HGetAll(ctx, keys.AccessGate()).Result()
	if err != nil {
		return fmt.Errorf("read applied access gate: %w", err)
	}
	routingValues, err := s.client.HGetAll(ctx, keys.RoutingGate()).Result()
	if err != nil {
		return fmt.Errorf("read applied routing gate: %w", err)
	}
	accessGate, err := ParsePublicationGate(accessValues)
	if err != nil {
		return fmt.Errorf("%w: %w", ErrStagedCorrupt, err)
	}
	routingGate, err := ParsePublicationGate(routingValues)
	if err != nil {
		return fmt.Errorf("%w: %w", ErrStagedCorrupt, err)
	}
	if accessGate.PublicationID != routingGate.PublicationID || accessGate.Revision != routingGate.Revision ||
		accessGate.RuntimeEpoch != routingGate.RuntimeEpoch || accessGate.Revision != applied.DesiredRevision ||
		accessGate.RuntimeEpoch != applied.RuntimeEpoch {
		return fmt.Errorf("%w: applied PostgreSQL and Redis gates disagree", ErrConflict)
	}
	accessGate.SnapshotDigest = routingGate.SnapshotDigest
	accessGate.SnapshotKey = routingGate.SnapshotKey
	publication, err := s.loadPublication(ctx, keys, accessGate)
	if err != nil {
		return err
	}
	if applied.RoutingDigest != "" && applied.RoutingDigest != publication.Routing.Snapshot.Digest {
		return fmt.Errorf("%w: applied PostgreSQL routing digest disagrees with Redis", ErrConflict)
	}
	values, err := s.client.HGetAll(ctx, keys.Publication(publication.ID)).Result()
	if err != nil {
		return fmt.Errorf("read applied publication state: %w", err)
	}
	var stored storedPlan
	if err := decodeStrict([]byte(values["plan"]), &stored); err != nil {
		return fmt.Errorf("%w: applied publication plan is invalid", ErrStagedCorrupt)
	}
	plan := PublicationPlan{
		Publication: publication, Previous: stored.Previous, Barriers: stored.Barriers,
		Supersedes: stored.Supersedes, PriorAccessGate: stored.PriorAccessGate,
		PriorRoutingGate: stored.PriorRoutingGate,
	}
	for {
		complete, err := s.Compact(ctx, plan, 500)
		if err != nil {
			return err
		}
		if complete {
			break
		}
	}
	if err := s.MarkApplied(ctx, plan); err != nil {
		return err
	}
	return s.ClearAppliedBarriers(ctx, plan)
}

func (s *RedisStore) loadPublication(ctx context.Context, keys Keyspace, gate PublicationGate) (Publication, error) {
	manifestPayload, bytesErr := s.client.Get(ctx, keys.Manifest(gate.PublicationID)).Bytes()
	if bytesErr != nil {
		return Publication{}, stagedReadError("applied manifest", bytesErr)
	}
	var manifest Manifest
	if err := decodeStrict(manifestPayload, &manifest); err != nil || manifest.Digest != gate.ManifestDigest {
		return Publication{}, fmt.Errorf("%w: applied manifest is invalid", ErrStagedCorrupt)
	}
	if err := verifyManifest(manifest); err != nil {
		return Publication{}, err
	}
	publication := Publication{
		ID: gate.PublicationID, NamespaceID: manifest.NamespaceID, QuotaPartition: manifest.QuotaPartition,
		DesiredRevision: manifest.DesiredRevision, RuntimeEpoch: manifest.RuntimeEpoch,
		Digest: gate.PublicationDigest, Manifest: manifest,
	}
	for _, keyID := range sortedMapKeys(manifest.Access) {
		entry := manifest.Access[keyID]
		values, err := s.client.HGetAll(ctx, keys.AccessDocument(keyID, entry.Revision)).Result()
		if err != nil || values["digest"] != entry.Digest {
			return Publication{}, fmt.Errorf("%w: applied access document %s is invalid", ErrStagedCorrupt, keyID)
		}
		var projection accessprojection.Projection
		if err := decodeStrict([]byte(values["document"]), &projection); err != nil {
			return Publication{}, fmt.Errorf("%w: applied access document %s cannot decode", ErrStagedCorrupt, keyID)
		}
		publication.Access = append(publication.Access, AccessDocument{
			NamespaceID: manifest.NamespaceID, QuotaPartition: manifest.QuotaPartition,
			DesiredRevision: entry.Revision, KeyID: keyID, Projection: projection, Digest: entry.Digest,
		})
	}
	for _, identity := range sortedMapKeys(manifest.Credentials) {
		entry := manifest.Credentials[identity]
		kind, publicID, ok := strings.Cut(identity, ":")
		if !ok {
			return Publication{}, fmt.Errorf("%w: applied credential identity is invalid", ErrStagedCorrupt)
		}
		payload, err := s.client.Get(ctx, keys.CredentialDocument(kind, publicID, entry.Revision)).Bytes()
		if err != nil {
			return Publication{}, stagedReadError("applied credential document", err)
		}
		var document CredentialDocument
		if err := decodeStrict(payload, &document); err != nil || document.Digest != entry.Digest {
			return Publication{}, fmt.Errorf("%w: applied credential document is invalid", ErrStagedCorrupt)
		}
		publication.Credentials = append(publication.Credentials, document)
	}
	for _, credentialID := range sortedMapKeys(manifest.ProviderCredentials) {
		entry := manifest.ProviderCredentials[credentialID]
		document, err := s.loadProviderCredentialDocument(ctx, keys, gate.PublicationID, credentialID, entry)
		if err != nil {
			return Publication{}, err
		}
		publication.ProviderCredentials = append(publication.ProviderCredentials, document)
	}
	routingPayload, bytesErr := s.client.Get(ctx, keys.RoutingSnapshot(manifest.DesiredRevision)).Bytes()
	if bytesErr != nil {
		return Publication{}, stagedReadError("applied routing snapshot", bytesErr)
	}
	if err := decodeStrict(routingPayload, &publication.Routing); err != nil ||
		publication.Routing.Digest != manifest.RoutingDigest || publication.Routing.Digest != gate.SnapshotDigest {
		return Publication{}, fmt.Errorf("%w: applied routing snapshot is invalid", ErrStagedCorrupt)
	}
	if err := verifyPublication(publication); err != nil {
		return Publication{}, fmt.Errorf("%w: %w", ErrStagedCorrupt, err)
	}
	return publication, nil
}

func (s *RedisStore) ClearAppliedBarriers(ctx context.Context, plan PublicationPlan) error {
	if err := validatePlan(plan); err != nil {
		return err
	}
	p := plan.Publication
	keys, _ := NewKeyspace(s.keyPrefix, p.NamespaceID, p.QuotaPartition)
	applied, err := s.client.HGetAll(ctx, keys.AppliedRevision()).Result()
	if err != nil {
		return fmt.Errorf("read applied publication: %w", err)
	}
	if applied["publication_id"] != p.ID || applied["desired_revision"] != strconv.FormatUint(p.DesiredRevision, 10) {
		return fmt.Errorf("%w: barriers cannot clear before the publication is applied", ErrConflict)
	}
	publicationIDs, err := s.client.ZRangeByScore(ctx, keys.OpenPublications(), &redis.ZRangeBy{
		Min: "-inf", Max: strconv.FormatUint(p.DesiredRevision, 10),
	}).Result()
	if err != nil {
		return fmt.Errorf("read superseded publications: %w", err)
	}
	publicationIDs = uniqueStrings(append(publicationIDs, append(plan.Supersedes, p.ID)...))
	for _, publicationID := range publicationIDs {
		barrierKeys, err := s.client.SMembers(ctx, keys.PublicationBarriers(publicationID)).Result()
		if err != nil {
			return fmt.Errorf("read publication barriers: %w", err)
		}
		sort.Strings(barrierKeys)
		for _, barrierKey := range barrierKeys {
			if _, err := clearOneBarrierScript.Run(ctx, s.client, []string{barrierKey}, publicationID).Result(); err != nil {
				return fmt.Errorf("clear applied deny barrier: %w", err)
			}
		}
		if publicationID == p.ID {
			if err := s.client.HSet(ctx, keys.Publication(publicationID), "state", "finalized").Err(); err != nil {
				return fmt.Errorf("finalize publication: %w", err)
			}
		} else {
			if err := s.client.HSet(ctx, keys.Publication(publicationID), "state", "superseded").Err(); err != nil {
				return fmt.Errorf("supersede publication: %w", err)
			}
		}
		if err := s.client.ZRem(ctx, keys.OpenPublications(), publicationID).Err(); err != nil {
			return fmt.Errorf("close publication: %w", err)
		}
	}
	if _, err := clearPendingPublicationScript.Run(ctx, s.client,
		[]string{keys.PendingPublication()}, p.ID).Result(); err != nil {
		return fmt.Errorf("clear applied pending publication: %w", err)
	}
	return nil
}

func (s *RedisStore) Readiness(ctx context.Context, namespaceID, partition string) (Readiness, error) {
	keys, err := NewKeyspace(s.keyPrefix, namespaceID, partition)
	if err != nil {
		return Readiness{}, err
	}
	pipeline := s.client.Pipeline()
	epochCommand := pipeline.Get(ctx, keys.RuntimeEpoch())
	accessCommand := pipeline.HGetAll(ctx, keys.AccessGate())
	routingCommand := pipeline.HGetAll(ctx, keys.RoutingGate())
	appliedCommand := pipeline.HGetAll(ctx, keys.AppliedRevision())
	_, err = pipeline.Exec(ctx)
	if err != nil && !errors.Is(err, redis.Nil) {
		return Readiness{}, fmt.Errorf("read publication readiness: %w", err)
	}
	readiness := Readiness{AccessGate: accessCommand.Val()["publication_id"], RoutingGate: routingCommand.Val()["publication_id"]}
	readiness.RuntimeEpoch, _ = strconv.ParseUint(epochCommand.Val(), 10, 64)
	readiness.DesiredRevision, _ = strconv.ParseUint(accessCommand.Val()["revision"], 10, 64)
	readiness.AppliedRevision, _ = strconv.ParseUint(appliedCommand.Val()["desired_revision"], 10, 64)
	if readiness.DesiredRevision >= readiness.AppliedRevision {
		readiness.ProjectorLag = readiness.DesiredRevision - readiness.AppliedRevision
	}
	switch {
	case readiness.RuntimeEpoch == 0:
		readiness.Reason = "runtime_epoch_unpublished"
	case readiness.AccessGate == "" || readiness.RoutingGate == "":
		readiness.Reason = "publication_gate_unpublished"
	case readiness.AccessGate != readiness.RoutingGate:
		readiness.Reason = "publication_gate_mismatch"
	case accessCommand.Val()["runtime_epoch"] != epochCommand.Val() || routingCommand.Val()["runtime_epoch"] != epochCommand.Val():
		readiness.Reason = "runtime_epoch_mismatch"
	case appliedCommand.Val()["publication_id"] != readiness.AccessGate:
		readiness.Reason = "applied_publication_lagging"
	case readiness.AppliedRevision != readiness.DesiredRevision:
		readiness.Reason = "applied_revision_lagging"
	case accessCommand.Val()["publication_digest"] != appliedCommand.Val()["access_digest"]:
		readiness.Reason = "access_digest_mismatch"
	case routingCommand.Val()["snapshot_digest"] != appliedCommand.Val()["routing_digest"]:
		readiness.Reason = "routing_digest_mismatch"
	default:
		readiness.Ready = true
		readiness.Reason = "ready"
	}
	return readiness, nil
}
