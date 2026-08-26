package delegationmanagement

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
)

type RedisPublicationWaiter struct {
	store  publicationRuntimeStore
	reader publicationProjectionReader
}

type publicationRuntimeStore interface {
	Readiness(context.Context, string, string) (accesspublisher.Readiness, error)
	ActiveReplicaAcknowledgements(
		context.Context, string, string, accesspublisher.ActiveGeneration,
	) (accesspublisher.ActiveReplicaStatus, error)
}

type publicationProjectionReader interface {
	accessruntime.ProjectionReader
	LocateCredentialCoherent(context.Context, accesscredential.Kind, string) (accessruntime.CredentialLocation, error)
}

type activeCredentialPublication struct {
	location   accessruntime.CredentialLocation
	credential accessprojection.CredentialProjection
	projection accessprojection.Projection
}

func NewRedisPublicationWaiter(client redis.UniversalClient, keyPrefix string) (*RedisPublicationWaiter, error) {
	store, err := accesspublisher.NewRedisStore(accesspublisher.RedisStoreOptions{Client: client, KeyPrefix: keyPrefix})
	if err != nil {
		return nil, err
	}
	reader, err := accessruntime.NewRedisProjectionReader(accessruntime.RedisProjectionReaderOptions{Client: client, KeyPrefix: keyPrefix})
	if err != nil {
		return nil, err
	}
	return &RedisPublicationWaiter{store: store, reader: reader}, nil
}

func (waiter *RedisPublicationWaiter) WaitActive(ctx context.Context, session Session, revision uint64) error {
	if waiter == nil || waiter.store == nil || waiter.reader == nil || revision == 0 {
		return ErrUnavailable
	}
	return waiter.wait(ctx, func(ctx context.Context) (bool, error) {
		ready, err := waiter.store.Readiness(ctx, session.NamespaceID, session.QuotaPartition)
		if err != nil || !ready.Ready || ready.AppliedRevision < revision {
			return false, err
		}
		publication, available, err := waiter.activeCredential(
			ctx, accesscredential.KindDelegation, session.PublicID, session.APIKeyID,
		)
		if err != nil || !available {
			return false, err
		}
		return publication.location.NamespaceID == session.NamespaceID &&
			publication.location.QuotaPartition == session.QuotaPartition &&
			publication.credential.ManagementSessionID == session.ManagementSessionID &&
			publication.credential.PrincipalID == session.PrincipalID &&
			publication.credential.DelegationEpoch == session.DelegationEpoch &&
			publication.credential.UserID == session.UserID &&
			publication.credential.TeamID == session.TeamID &&
			publication.credential.Audience == session.Audience, nil
	})
}

func (waiter *RedisPublicationWaiter) WaitApplied(ctx context.Context, namespaceID, partition string, revision uint64) error {
	if waiter == nil || waiter.store == nil || revision == 0 {
		return ErrUnavailable
	}
	return waiter.wait(ctx, func(ctx context.Context) (bool, error) {
		ready, err := waiter.store.Readiness(ctx, namespaceID, partition)
		if err != nil || !ready.Ready || ready.AppliedRevision < revision {
			return false, err
		}
		return waiter.activeReplicasReady(ctx, namespaceID, partition, accesspublisher.ActiveGeneration{
			PublicationID:         ready.RoutingGate,
			Revision:              ready.DesiredRevision,
			RuntimeEpoch:          ready.RuntimeEpoch,
			RoutingSnapshotDigest: ready.RoutingDigest,
		})
	})
}

// WaitAPIKeyActive waits for the exact one-time credential being returned by
// Management to be readable through the same publication gates and immutable
// policy projection used by inference authentication.
func (waiter *RedisPublicationWaiter) WaitAPIKeyActive(
	ctx context.Context,
	namespaceID string,
	keyID string,
	publicID string,
) error {
	if waiter == nil || waiter.store == nil || waiter.reader == nil || namespaceID == "" || keyID == "" || publicID == "" {
		return ErrUnavailable
	}
	return waiter.wait(ctx, func(ctx context.Context) (bool, error) {
		publication, available, err := waiter.activeCredential(
			ctx, accesscredential.KindAPIKey, publicID, keyID,
		)
		if err != nil || !available {
			return false, err
		}
		if publication.location.NamespaceID != namespaceID || publication.projection.NamespaceID != namespaceID {
			return false, fmt.Errorf("%w: API key publication namespace mismatch", accessruntime.ErrRuntimeCorrupt)
		}
		return true, nil
	})
}

func (waiter *RedisPublicationWaiter) activeCredential(
	ctx context.Context,
	kind accesscredential.Kind,
	publicID string,
	keyID string,
) (activeCredentialPublication, bool, error) {
	location, err := waiter.reader.LocateCredentialCoherent(ctx, kind, publicID)
	if err != nil {
		available, transitionErr := publicationTransition(err)
		return activeCredentialPublication{}, available, transitionErr
	}
	credential, err := waiter.reader.ReadCredential(ctx, location, kind, publicID)
	if err != nil {
		available, transitionErr := waiter.publicationReadTransition(ctx, kind, location, publicID, err)
		return activeCredentialPublication{}, available, transitionErr
	}
	active, err := waiter.reader.ReadActivePolicy(ctx, location, keyID)
	if err != nil {
		available, transitionErr := waiter.publicationReadTransition(ctx, kind, location, publicID, err)
		return activeCredentialPublication{}, available, transitionErr
	}
	projection, err := waiter.reader.ReadPolicy(ctx, location, active)
	if err != nil {
		available, transitionErr := waiter.publicationReadTransition(ctx, kind, location, publicID, err)
		return activeCredentialPublication{}, available, transitionErr
	}
	if credential.KeyID != keyID || projection.NamespaceID != location.NamespaceID ||
		projection.QuotaPartition != location.QuotaPartition || projection.KeyID != keyID ||
		projection.Revision != active.Revision {
		return activeCredentialPublication{}, false,
			fmt.Errorf("%w: credential publication identity mismatch", accessruntime.ErrRuntimeCorrupt)
	}
	if err := projection.VerifyDigest(active.Digest); err != nil {
		return activeCredentialPublication{}, false, fmt.Errorf("%w: %w", accessruntime.ErrRuntimeCorrupt, err)
	}
	if location.RoutingRevision <= 0 {
		return activeCredentialPublication{}, false,
			fmt.Errorf("%w: credential routing revision is invalid", accessruntime.ErrRuntimeCorrupt)
	}
	ready, err := waiter.activeReplicasReady(
		ctx, location.NamespaceID, location.QuotaPartition, accesspublisher.ActiveGeneration{
			PublicationID:         location.PublicationID,
			Revision:              uint64(location.RoutingRevision),
			RuntimeEpoch:          location.RuntimeEpoch,
			RoutingSnapshotDigest: location.RoutingDocumentDigest,
		},
	)
	if err != nil || !ready {
		return activeCredentialPublication{}, false, err
	}
	return activeCredentialPublication{
		location: location, credential: credential, projection: projection,
	}, true, nil
}

func (waiter *RedisPublicationWaiter) activeReplicasReady(
	ctx context.Context,
	namespaceID string,
	partition string,
	generation accesspublisher.ActiveGeneration,
) (bool, error) {
	replicas, err := waiter.store.ActiveReplicaAcknowledgements(
		ctx, namespaceID, partition, generation,
	)
	if errors.Is(err, accesspublisher.ErrPublicationChanged) ||
		errors.Is(err, accesspublisher.ErrAcknowledgements) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return replicas.Complete(), nil
}

// publicationReadTransition distinguishes a corrupt active publication from a
// valid publication advancing between the waiter's independent reads. A newer
// gate is retried; an inconsistency under the same gate fails immediately.
func (waiter *RedisPublicationWaiter) publicationReadTransition(
	ctx context.Context,
	kind accesscredential.Kind,
	location accessruntime.CredentialLocation,
	publicID string,
	readErr error,
) (bool, error) {
	if !errors.Is(readErr, accessruntime.ErrProjectionNotFound) &&
		!errors.Is(readErr, accessruntime.ErrPublicationPending) &&
		!errors.Is(readErr, accessruntime.ErrRuntimeCorrupt) {
		return false, readErr
	}
	refreshed, err := waiter.reader.LocateCredentialCoherent(ctx, kind, publicID)
	if err != nil {
		if errors.Is(err, accessruntime.ErrProjectionNotFound) ||
			errors.Is(err, accessruntime.ErrPublicationPending) {
			return false, nil
		}
		return false, err
	}
	if publicationAdvanced(location, refreshed) {
		return false, nil
	}
	return false, readErr
}

func publicationAdvanced(before, after accessruntime.CredentialLocation) bool {
	if before.NamespaceID != after.NamespaceID || before.QuotaPartition != after.QuotaPartition ||
		before.PublicationID == after.PublicationID {
		return false
	}
	return after.RuntimeEpoch > before.RuntimeEpoch ||
		(after.RuntimeEpoch == before.RuntimeEpoch && after.RoutingRevision > before.RoutingRevision)
}

func publicationTransition(err error) (bool, error) {
	if errors.Is(err, accessruntime.ErrProjectionNotFound) ||
		errors.Is(err, accessruntime.ErrPublicationPending) {
		return false, nil
	}
	return false, err
}

func (waiter *RedisPublicationWaiter) wait(ctx context.Context, condition func(context.Context) (bool, error)) error {
	ticker := time.NewTicker(25 * time.Millisecond)
	defer ticker.Stop()
	for {
		ready, err := condition(ctx)
		if err != nil {
			return fmt.Errorf("read publication state: %w", err)
		}
		if ready {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

var _ PublicationWaiter = (*RedisPublicationWaiter)(nil)
