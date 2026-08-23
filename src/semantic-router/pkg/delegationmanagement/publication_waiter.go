package delegationmanagement

import (
	"context"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
)

type RedisPublicationWaiter struct {
	store  *accesspublisher.RedisStore
	reader *accessruntime.RedisProjectionReader
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
		location, err := waiter.reader.LocateCredential(ctx, accesscredential.KindDelegation, session.PublicID)
		if err != nil {
			return false, nil
		}
		credential, err := waiter.reader.ReadCredential(ctx, location, accesscredential.KindDelegation, session.PublicID)
		if err != nil {
			return false, nil
		}
		return location.NamespaceID == session.NamespaceID && location.QuotaPartition == session.QuotaPartition &&
			credential.KeyID == session.APIKeyID && credential.ManagementSessionID == session.ManagementSessionID &&
			credential.PrincipalID == session.PrincipalID && credential.DelegationEpoch == session.DelegationEpoch &&
			credential.UserID == session.UserID && credential.TeamID == session.TeamID &&
			credential.Audience == session.Audience, nil
	})
}

func (waiter *RedisPublicationWaiter) WaitApplied(ctx context.Context, namespaceID, partition string, revision uint64) error {
	if waiter == nil || waiter.store == nil || revision == 0 {
		return ErrUnavailable
	}
	return waiter.wait(ctx, func(ctx context.Context) (bool, error) {
		ready, err := waiter.store.Readiness(ctx, namespaceID, partition)
		return err == nil && ready.Ready && ready.AppliedRevision >= revision, err
	})
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
