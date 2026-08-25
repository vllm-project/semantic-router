package accessruntime

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

type RedisProjectionReader struct {
	client    redis.Cmdable
	keyPrefix string
}

type RedisProjectionReaderOptions struct {
	Client    redis.Cmdable
	KeyPrefix string
}

var _ ProjectionReader = (*RedisProjectionReader)(nil)

func NewRedisProjectionReader(options RedisProjectionReaderOptions) (*RedisProjectionReader, error) {
	if options.Client == nil {
		return nil, fmt.Errorf("redis projection client is required")
	}
	if _, err := quotaruntime.NewAccessProjectionKeyspaceWithPrefix(options.KeyPrefix, "validation"); err != nil {
		return nil, fmt.Errorf("redis projection key prefix: %w", err)
	}
	return &RedisProjectionReader{client: options.Client, keyPrefix: options.KeyPrefix}, nil
}

func (r *RedisProjectionReader) LocateCredential(
	ctx context.Context,
	kind accesscredential.Kind,
	publicID string,
) (CredentialLocation, error) {
	key, locateCredentialErr := quotaruntime.CredentialDirectoryKeyWithPrefix(r.keyPrefix, string(kind), publicID)
	if locateCredentialErr != nil {
		return CredentialLocation{}, fmt.Errorf("credential directory key: %w", locateCredentialErr)
	}
	values, locateCredentialErr := r.client.HGetAll(ctx, key).Result()
	if locateCredentialErr != nil {
		return CredentialLocation{}, fmt.Errorf("%w: read credential directory", ErrRuntimeUnavailable)
	}
	if len(values) == 0 {
		return CredentialLocation{}, ErrProjectionNotFound
	}
	required := []string{"publication_id", "state", "partition", "namespace_id", "kind", "public_id"}
	for _, field := range required {
		if strings.TrimSpace(values[field]) == "" {
			return CredentialLocation{}, fmt.Errorf("%w: credential directory field %s is missing", ErrRuntimeCorrupt, field)
		}
	}
	if values["state"] != string(accesspublisher.PointerStateActive) ||
		values["kind"] != string(kind) || values["public_id"] != publicID {
		return CredentialLocation{}, ErrProjectionNotFound
	}
	location := CredentialLocation{
		NamespaceID: values["namespace_id"], QuotaPartition: values["partition"],
		PublicationID: values["publication_id"],
	}
	if _, err := quotaruntime.NewAccessProjectionKeyspaceWithPrefix(r.keyPrefix, location.QuotaPartition); err != nil {
		return CredentialLocation{}, fmt.Errorf("%w: invalid credential partition", ErrRuntimeCorrupt)
	}
	accessGate, routingGate, locateCredentialErr := r.readPublicationGates(ctx, location.NamespaceID, location.QuotaPartition)
	if locateCredentialErr != nil {
		return CredentialLocation{}, locateCredentialErr
	}
	if accessGate.PublicationID != location.PublicationID || routingGate.PublicationID != location.PublicationID ||
		accessGate.RuntimeEpoch != routingGate.RuntimeEpoch || accessGate.Revision != routingGate.Revision ||
		routingGate.SnapshotDigest == "" || routingGate.Revision > math.MaxInt64 {
		return CredentialLocation{}, fmt.Errorf("%w: credential directory and publication gates disagree", ErrRuntimeCorrupt)
	}
	location.RuntimeEpoch = routingGate.RuntimeEpoch
	// #nosec G115 -- the PostgreSQL BIGINT bound is checked above.
	location.RoutingRevision = int64(routingGate.Revision)
	location.RoutingSnapshotHash = routingGate.SnapshotDigest
	return location, nil
}

func (r *RedisProjectionReader) ReadCredential(
	ctx context.Context,
	location CredentialLocation,
	kind accesscredential.Kind,
	publicID string,
) (accessprojection.CredentialProjection, error) {
	keys, err := quotaruntime.NewAccessProjectionKeyspaceWithPrefix(r.keyPrefix, location.QuotaPartition)
	if err != nil {
		return accessprojection.CredentialProjection{}, err
	}
	values, err := r.client.HGetAll(ctx, keys.Credential(string(kind), publicID)).Result()
	if err != nil {
		return accessprojection.CredentialProjection{}, fmt.Errorf("%w: read credential projection", ErrRuntimeUnavailable)
	}
	if len(values) == 0 {
		return accessprojection.CredentialProjection{}, ErrProjectionNotFound
	}
	required := []string{"publication_id", "state", "kind", "kid", "key_id", "secret_hmac", "pepper_version", "status", "not_before_ms"}
	for _, field := range required {
		if strings.TrimSpace(values[field]) == "" {
			return accessprojection.CredentialProjection{}, fmt.Errorf("%w: credential field %s is missing", ErrRuntimeCorrupt, field)
		}
	}
	if values["publication_id"] != location.PublicationID || values["state"] != string(accesspublisher.PointerStateActive) {
		return accessprojection.CredentialProjection{}, fmt.Errorf("%w: credential pointer does not match the active publication", ErrRuntimeCorrupt)
	}
	if values["kind"] != string(kind) {
		return accessprojection.CredentialProjection{}, fmt.Errorf("%w: credential kind mismatch", ErrRuntimeCorrupt)
	}
	digest, err := base64.RawURLEncoding.DecodeString(values["secret_hmac"])
	if err != nil || len(digest) != 32 {
		return accessprojection.CredentialProjection{}, fmt.Errorf("%w: credential verifier is invalid", ErrRuntimeCorrupt)
	}
	status := accesscontrol.CredentialStatus(values["status"])
	if status != accesscontrol.CredentialStatusActive && status != accesscontrol.CredentialStatusRetiring {
		return accessprojection.CredentialProjection{}, fmt.Errorf("%w: credential status is not publishable", ErrRuntimeCorrupt)
	}
	notBefore, err := parseProjectionMilliseconds(values["not_before_ms"])
	if err != nil {
		return accessprojection.CredentialProjection{}, fmt.Errorf("%w: credential not-before is invalid", ErrRuntimeCorrupt)
	}
	var expiresAt *time.Time
	if value := values["expires_at_ms"]; value != "" {
		parsed, parseErr := parseProjectionMilliseconds(value)
		if parseErr != nil || !parsed.After(notBefore) {
			return accessprojection.CredentialProjection{}, fmt.Errorf("%w: credential expiry is invalid", ErrRuntimeCorrupt)
		}
		expiresAt = &parsed
	}
	if publicID != values["kid"] {
		return accessprojection.CredentialProjection{}, fmt.Errorf("%w: credential public ID mismatch", ErrRuntimeCorrupt)
	}
	if status == accesscontrol.CredentialStatusRetiring && expiresAt == nil {
		return accessprojection.CredentialProjection{}, fmt.Errorf("%w: retiring credential is unbounded", ErrRuntimeCorrupt)
	}
	projection := accessprojection.CredentialProjection{
		Kind: string(kind), KID: publicID, KeyID: values["key_id"], SecretHMAC: digest,
		PepperVersion: values["pepper_version"], Status: string(status),
		NotBefore: notBefore, ExpiresAt: expiresAt,
	}
	if kind == accesscredential.KindDelegation {
		for _, field := range []string{"management_session_id", "principal_id", "delegation_epoch", "user_id", "audience"} {
			if strings.TrimSpace(values[field]) == "" {
				return accessprojection.CredentialProjection{}, fmt.Errorf("%w: delegated credential field %s is missing", ErrRuntimeCorrupt, field)
			}
		}
		delegationEpoch, parseErr := strconv.ParseUint(values["delegation_epoch"], 10, 64)
		if parseErr != nil || delegationEpoch == 0 {
			return accessprojection.CredentialProjection{}, fmt.Errorf("%w: delegated credential epoch is invalid", ErrRuntimeCorrupt)
		}
		projection.ManagementSessionID = values["management_session_id"]
		projection.PrincipalID = values["principal_id"]
		projection.DelegationEpoch = delegationEpoch
		projection.UserID = values["user_id"]
		projection.TeamID = values["team_id"]
		projection.Audience = values["audience"]
	} else if values["management_session_id"] != "" || values["principal_id"] != "" ||
		values["delegation_epoch"] != "" || values["user_id"] != "" ||
		values["team_id"] != "" || values["audience"] != "" {
		return accessprojection.CredentialProjection{}, fmt.Errorf("%w: API-key credential carries delegation context", ErrRuntimeCorrupt)
	}
	return projection, nil
}

func (r *RedisProjectionReader) ReadActivePolicy(
	ctx context.Context,
	location CredentialLocation,
	keyID string,
) (ActivePolicy, error) {
	keys, err := quotaruntime.NewAccessProjectionKeyspaceWithPrefix(r.keyPrefix, location.QuotaPartition)
	if err != nil {
		return ActivePolicy{}, err
	}
	values, err := r.client.HGetAll(ctx, keys.Active(keyID)).Result()
	if err != nil {
		return ActivePolicy{}, fmt.Errorf("%w: read active policy", ErrRuntimeUnavailable)
	}
	if len(values) == 0 {
		return ActivePolicy{}, ErrProjectionNotFound
	}
	if values["publication_id"] != location.PublicationID || values["state"] != string(accesspublisher.PointerStateActive) {
		return ActivePolicy{}, fmt.Errorf("%w: active policy does not match the active publication", ErrRuntimeCorrupt)
	}
	revision, err := strconv.ParseUint(values["revision"], 10, 64)
	if err != nil || revision == 0 {
		return ActivePolicy{}, fmt.Errorf("%w: active policy revision is invalid", ErrRuntimeCorrupt)
	}
	digest := values["digest"]
	if !validHexDigest(digest) {
		return ActivePolicy{}, fmt.Errorf("%w: active policy digest is invalid", ErrRuntimeCorrupt)
	}
	return ActivePolicy{
		KeyID: keyID, Revision: revision, Digest: digest,
		PublicationID: location.PublicationID, RuntimeEpoch: location.RuntimeEpoch,
		RoutingRevision: location.RoutingRevision, RoutingSnapshotHash: location.RoutingSnapshotHash,
	}, nil
}

func (r *RedisProjectionReader) ReadPolicy(
	ctx context.Context,
	location CredentialLocation,
	active ActivePolicy,
) (accessprojection.Projection, error) {
	keys, err := quotaruntime.NewAccessProjectionKeyspaceWithPrefix(r.keyPrefix, location.QuotaPartition)
	if err != nil {
		return accessprojection.Projection{}, err
	}
	values, err := r.client.HGetAll(ctx, keys.Policy(active.KeyID, strconv.FormatUint(active.Revision, 10))).Result()
	if err != nil {
		return accessprojection.Projection{}, fmt.Errorf("%w: read policy projection", ErrRuntimeUnavailable)
	}
	if len(values) == 0 {
		return accessprojection.Projection{}, ErrProjectionNotFound
	}
	if values["publication_id"] != active.PublicationID || values["digest"] != active.Digest || values["document"] == "" {
		return accessprojection.Projection{}, fmt.Errorf("%w: policy pointer and document disagree", ErrRuntimeCorrupt)
	}
	var projection accessprojection.Projection
	decoder := json.NewDecoder(bytes.NewBufferString(values["document"]))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&projection); err != nil {
		return accessprojection.Projection{}, fmt.Errorf("%w: decode policy projection", ErrRuntimeCorrupt)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return accessprojection.Projection{}, fmt.Errorf("%w: policy projection has trailing data", ErrRuntimeCorrupt)
	}
	return projection, nil
}

// ReadAppliedPolicy reads the same active pointer and immutable document used
// by inference without going through the public credential directory. The
// namespace publication gates remain the authority boundary, so a caller
// cannot combine a key with another namespace or quota partition.
func (r *RedisProjectionReader) ReadAppliedPolicy(
	ctx context.Context,
	namespaceID string,
	partition string,
	keyID string,
) (AppliedPolicy, error) {
	if strings.TrimSpace(namespaceID) == "" || strings.TrimSpace(keyID) == "" {
		return AppliedPolicy{}, ErrProjectionNotFound
	}
	accessGate, routingGate, err := r.readPublicationGates(ctx, namespaceID, partition)
	if err != nil {
		return AppliedPolicy{}, err
	}
	if accessGate.PublicationID == "" || accessGate.PublicationID != routingGate.PublicationID ||
		accessGate.RuntimeEpoch != routingGate.RuntimeEpoch || accessGate.Revision != routingGate.Revision ||
		routingGate.SnapshotDigest == "" || routingGate.Revision > math.MaxInt64 {
		return AppliedPolicy{}, fmt.Errorf("%w: publication gates disagree", ErrRuntimeCorrupt)
	}
	// #nosec G115 -- the PostgreSQL BIGINT bound is checked above.
	routingRevision := int64(routingGate.Revision)
	location := CredentialLocation{
		NamespaceID: namespaceID, QuotaPartition: partition,
		PublicationID: accessGate.PublicationID, RuntimeEpoch: accessGate.RuntimeEpoch,
		RoutingRevision: routingRevision, RoutingSnapshotHash: routingGate.SnapshotDigest,
	}
	active, err := r.ReadActivePolicy(ctx, location, keyID)
	if err != nil {
		return AppliedPolicy{}, err
	}
	projection, err := r.ReadPolicy(ctx, location, active)
	if err != nil {
		return AppliedPolicy{}, err
	}
	if projection.NamespaceID != namespaceID || projection.QuotaPartition != partition ||
		projection.KeyID != keyID || projection.Revision != active.Revision {
		return AppliedPolicy{}, fmt.Errorf("%w: applied policy identity mismatch", ErrRuntimeCorrupt)
	}
	if err := projection.VerifyDigest(active.Digest); err != nil {
		return AppliedPolicy{}, fmt.Errorf("%w: %w", ErrRuntimeCorrupt, err)
	}
	return AppliedPolicy{Active: active, Projection: projection}, nil
}

func (r *RedisProjectionReader) readPublicationGates(
	ctx context.Context,
	namespaceID string,
	partition string,
) (accesspublisher.PublicationGate, accesspublisher.PublicationGate, error) {
	keys, err := accesspublisher.NewKeyspace(r.keyPrefix, namespaceID, partition)
	if err != nil {
		return accesspublisher.PublicationGate{}, accesspublisher.PublicationGate{},
			fmt.Errorf("%w: invalid publication keyspace", ErrRuntimeCorrupt)
	}
	accessValues, err := r.client.HGetAll(ctx, keys.AccessGate()).Result()
	if err != nil {
		return accesspublisher.PublicationGate{}, accesspublisher.PublicationGate{},
			fmt.Errorf("%w: read publication gates", ErrRuntimeUnavailable)
	}
	routingValues, err := r.client.HGetAll(ctx, keys.RoutingGate()).Result()
	if err != nil {
		return accesspublisher.PublicationGate{}, accesspublisher.PublicationGate{},
			fmt.Errorf("%w: read publication gates", ErrRuntimeUnavailable)
	}
	accessGate, err := accesspublisher.ParsePublicationGate(accessValues)
	if err != nil {
		return accesspublisher.PublicationGate{}, accesspublisher.PublicationGate{},
			fmt.Errorf("%w: invalid access publication gate", ErrRuntimeCorrupt)
	}
	routingGate, err := accesspublisher.ParsePublicationGate(routingValues)
	if err != nil {
		return accesspublisher.PublicationGate{}, accesspublisher.PublicationGate{},
			fmt.Errorf("%w: invalid routing publication gate", ErrRuntimeCorrupt)
	}
	return accessGate, routingGate, nil
}

func parseProjectionMilliseconds(value string) (time.Time, error) {
	milliseconds, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return time.Time{}, err
	}
	return time.UnixMilli(milliseconds).UTC(), nil
}

func validHexDigest(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, character := range value {
		if (character < '0' || character > '9') && (character < 'a' || character > 'f') {
			return false
		}
	}
	return true
}
