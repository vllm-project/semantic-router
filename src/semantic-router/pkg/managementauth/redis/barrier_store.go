// Package redis implements the globally shared Management revocation barrier
// plane on Redis or Valkey. Barriers live in an atomically selected generation
// so a cold rebuild never exposes a partially reconstructed allow state.
package redis

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/google/uuid"
	redisclient "github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

const (
	readyField              = "__ready"
	defaultLockTTL          = 30 * time.Second
	defaultOldGenerationTTL = time.Minute
)

var keyPrefixPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9:_.-]{0,127}$`)

var (
	ErrNotReady       = errors.New("management revocation barriers are not ready")
	ErrRebuildBusy    = errors.New("management revocation barrier rebuild is already in progress")
	ErrInvalidBarrier = errors.New("management revocation barrier is invalid")
)

type Options struct {
	Client           *redisclient.Client
	KeyPrefix        string
	Loader           managementauth.RevocationSnapshotLoader
	LockTTL          time.Duration
	OldGenerationTTL time.Duration
	NewGenerationID  func() (string, error)
}

// Store implements managementauth.RevocationBarrierStore. It never treats a
// missing generation/ready marker as an empty allow set.
type Store struct {
	client           *redisclient.Client
	prefix           string
	loader           managementauth.RevocationSnapshotLoader
	lockTTL          time.Duration
	oldGenerationTTL time.Duration
	newGenerationID  func() (string, error)
}

func New(options Options) (*Store, error) {
	if options.Client == nil || options.Loader == nil || !keyPrefixPattern.MatchString(options.KeyPrefix) {
		return nil, errors.New("management revocation barriers require a client, snapshot loader, and canonical key prefix")
	}
	lockTTL := options.LockTTL
	if lockTTL == 0 {
		lockTTL = defaultLockTTL
	}
	oldTTL := options.OldGenerationTTL
	if oldTTL == 0 {
		oldTTL = defaultOldGenerationTTL
	}
	if lockTTL < time.Second || lockTTL > 5*time.Minute || oldTTL < time.Second || oldTTL > time.Hour {
		return nil, errors.New("management revocation barrier lock and generation TTLs are out of bounds")
	}
	generator := options.NewGenerationID
	if generator == nil {
		generator = func() (string, error) { return uuid.NewString(), nil }
	}
	return &Store{
		client: options.Client, prefix: options.KeyPrefix, loader: options.Loader,
		lockTTL: lockTTL, oldGenerationTTL: oldTTL, newGenerationID: generator,
	}, nil
}

func (store *Store) Ready(ctx context.Context) error {
	if store == nil || store.client == nil {
		return ErrNotReady
	}
	if err := store.client.Ping(ctx).Err(); err != nil {
		return fmt.Errorf("check Management revocation Valkey: %w", err)
	}
	generation, err := store.client.Get(ctx, store.activeKey()).Result()
	if err != nil || !canonicalGeneration(generation) {
		return ErrNotReady
	}
	ready, err := store.client.HGet(ctx, store.generationKey(generation), readyField).Result()
	if err != nil || ready != "1" {
		return ErrNotReady
	}
	return nil
}

func (store *Store) Check(ctx context.Context, check managementauth.BarrierCheck) (managementauth.BarrierState, error) {
	if store == nil || store.client == nil || validateCheck(check) != nil {
		return managementauth.BarrierState{}, ErrInvalidBarrier
	}
	fields := []any{
		store.generationPrefix(), readyField,
		barrierField(managementauth.BarrierClusterSessionPolicy, "singleton"),
		barrierField(managementauth.BarrierNamespaceSecurityPolicy, check.NamespaceID),
		barrierField(managementauth.BarrierManagementSession, check.SessionID),
		barrierField(managementauth.BarrierManagementPrincipal, check.PrincipalID),
		barrierField(managementauth.BarrierAuthenticationSource, authSourceBarrierID(check.AuthSourceKind, check.AuthSourceID)),
	}
	result, err := checkScript.Run(ctx, store.client, []string{store.activeKey()}, fields...).Slice()
	if err != nil {
		if errors.Is(err, redisclient.Nil) {
			return managementauth.BarrierState{Ready: false}, nil
		}
		return managementauth.BarrierState{}, fmt.Errorf("check Management revocation barriers: %w", err)
	}
	if len(result) != 6 || stringValue(result[0]) != "1" {
		return managementauth.BarrierState{Ready: false}, nil
	}
	return managementauth.BarrierState{
		Ready:            true,
		ClusterDenied:    stringValue(result[1]) == "1",
		NamespaceDenied:  stringValue(result[2]) == "1",
		SessionDenied:    stringValue(result[3]) == "1",
		PrincipalDenied:  stringValue(result[4]) == "1",
		AuthSourceDenied: stringValue(result[5]) == "1",
	}, nil
}

func (store *Store) CheckDelegation(
	ctx context.Context,
	check managementauth.DelegationBarrierCheck,
) (managementauth.DelegationBarrierState, error) {
	if store == nil || store.client == nil || validateDelegationCheck(check) != nil {
		return managementauth.DelegationBarrierState{}, ErrInvalidBarrier
	}
	result, err := delegationCheckScript.Run(ctx, store.client, []string{store.activeKey()},
		store.generationPrefix(), readyField,
		barrierField(managementauth.BarrierManagementSession, check.SessionID),
		barrierField(managementauth.BarrierManagementPrincipal, check.PrincipalID),
	).Slice()
	if err != nil {
		if errors.Is(err, redisclient.Nil) {
			return managementauth.DelegationBarrierState{Ready: false}, nil
		}
		return managementauth.DelegationBarrierState{}, fmt.Errorf("check delegated inference revocation barriers: %w", err)
	}
	if len(result) != 3 || stringValue(result[0]) != "1" {
		return managementauth.DelegationBarrierState{Ready: false}, nil
	}
	return managementauth.DelegationBarrierState{
		Ready:           true,
		SessionDenied:   stringValue(result[1]) == "1",
		PrincipalDenied: stringValue(result[2]) == "1",
	}, nil
}

func (store *Store) InstallDeny(ctx context.Context, kind managementauth.BarrierKind, id string) error {
	return store.mutate(ctx, kind, id, true)
}

// RemoveDeny is used only after the authoritative lifecycle mutation has
// revoked every session covered by the old barrier. It never reconstructs an
// allow state from a cache miss.
func (store *Store) RemoveDeny(ctx context.Context, kind managementauth.BarrierKind, id string) error {
	return store.mutate(ctx, kind, id, false)
}

func (store *Store) mutate(ctx context.Context, kind managementauth.BarrierKind, id string, install bool) error {
	if store == nil || store.client == nil || validateBarrier(managementauth.RevocationBarrier{Kind: kind, ID: id}) != nil {
		return ErrInvalidBarrier
	}
	token, err := store.acquireLock(ctx)
	if err != nil {
		return err
	}
	defer store.releaseLock(context.WithoutCancel(ctx), token)
	generation, err := store.client.Get(ctx, store.activeKey()).Result()
	if err != nil || !canonicalGeneration(generation) {
		return ErrNotReady
	}
	hash := store.generationKey(generation)
	if ready, readyErr := store.client.HGet(ctx, hash, readyField).Result(); readyErr != nil || ready != "1" {
		return ErrNotReady
	}
	field := barrierField(kind, id)
	if install {
		err = store.client.HSet(ctx, hash, field, "1").Err()
	} else {
		err = store.client.HDel(ctx, hash, field).Err()
	}
	if err != nil {
		return fmt.Errorf("mutate Management revocation barrier: %w", err)
	}
	return nil
}

// Rebuild reconstructs all durable lifecycle barriers under a new generation
// and swaps it atomically. Install/remove use the same distributed lock, so a
// committed mutation is represented either by the snapshot or by the mutation
// that runs immediately after the swap.
func (store *Store) Rebuild(ctx context.Context) error {
	if store == nil || store.client == nil || store.loader == nil {
		return ErrNotReady
	}
	token, rebuildErr := store.acquireLock(ctx)
	if rebuildErr != nil {
		return rebuildErr
	}
	defer store.releaseLock(context.WithoutCancel(ctx), token)
	barriers, rebuildErr := store.loader.LoadRevocationBarriers(ctx)
	if rebuildErr != nil {
		return fmt.Errorf("load durable Management revocations: %w", rebuildErr)
	}
	generation, rebuildErr := store.newGenerationID()
	if rebuildErr != nil || !canonicalGeneration(generation) {
		return errors.New("generate Management revocation generation")
	}
	hash := store.generationKey(generation)
	values := make(map[string]any, len(barriers)+1)
	values[readyField] = "1"
	for _, barrier := range barriers {
		if err := validateBarrier(barrier); err != nil {
			return err
		}
		field := barrierField(barrier.Kind, barrier.ID)
		if _, duplicate := values[field]; duplicate {
			return fmt.Errorf("%w: duplicate %s", ErrInvalidBarrier, field)
		}
		values[field] = "1"
	}
	if err := store.client.HSet(ctx, hash, values).Err(); err != nil {
		return fmt.Errorf("stage Management revocation generation: %w", err)
	}
	swapped, rebuildErr := swapScript.Run(ctx, store.client,
		[]string{store.lockKey(), store.activeKey()},
		token, generation, store.generationPrefix(), int64(store.oldGenerationTTL/time.Millisecond),
	).Int64()
	if rebuildErr != nil || swapped != 1 {
		_ = store.client.Del(context.WithoutCancel(ctx), hash).Err()
		if rebuildErr != nil {
			return fmt.Errorf("activate Management revocation generation: %w", rebuildErr)
		}
		return ErrRebuildBusy
	}
	return nil
}

func (store *Store) acquireLock(ctx context.Context) (string, error) {
	token := uuid.NewString()
	deadline := time.NewTimer(2 * time.Second)
	defer deadline.Stop()
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()
	for {
		acquired, err := store.client.SetNX(ctx, store.lockKey(), token, store.lockTTL).Result()
		if err != nil {
			return "", fmt.Errorf("acquire Management revocation lock: %w", err)
		}
		if acquired {
			return token, nil
		}
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-deadline.C:
			return "", ErrRebuildBusy
		case <-ticker.C:
		}
	}
}

func (store *Store) releaseLock(ctx context.Context, token string) {
	_, _ = releaseLockScript.Run(ctx, store.client, []string{store.lockKey()}, token).Result()
}

func (store *Store) activeKey() string        { return store.prefix + ":management:revocations:active" }
func (store *Store) lockKey() string          { return store.prefix + ":management:revocations:lock" }
func (store *Store) generationPrefix() string { return store.prefix + ":management:revocations:g:" }
func (store *Store) generationKey(generation string) string {
	return store.generationPrefix() + generation
}

func validateCheck(check managementauth.BarrierCheck) error {
	for _, value := range []string{check.SessionID, check.PrincipalID, check.AuthSourceID} {
		if _, err := uuid.Parse(value); err != nil {
			return ErrInvalidBarrier
		}
	}
	if check.NamespaceID != "" {
		if _, err := uuid.Parse(check.NamespaceID); err != nil {
			return ErrInvalidBarrier
		}
	}
	switch check.AuthSourceKind {
	case managementauth.AuthSourceIssuer, managementauth.AuthSourceServiceCredential, managementauth.AuthSourceMTLS:
		return nil
	default:
		return ErrInvalidBarrier
	}
}

func validateDelegationCheck(check managementauth.DelegationBarrierCheck) error {
	for _, value := range []string{check.SessionID, check.PrincipalID} {
		parsed, err := uuid.Parse(value)
		if err != nil || parsed.String() != value {
			return ErrInvalidBarrier
		}
	}
	return nil
}

func validateBarrier(barrier managementauth.RevocationBarrier) error {
	switch barrier.Kind {
	case managementauth.BarrierClusterSessionPolicy:
		if barrier.ID == "singleton" {
			return nil
		}
	case managementauth.BarrierNamespaceSecurityPolicy, managementauth.BarrierManagementSession,
		managementauth.BarrierManagementPrincipal:
		if _, err := uuid.Parse(barrier.ID); err == nil {
			return nil
		}
	case managementauth.BarrierAuthenticationSource:
		kind, id, found := strings.Cut(barrier.ID, ":")
		if found && id != "" {
			if _, err := uuid.Parse(id); err == nil {
				switch managementauth.AuthSourceKind(kind) {
				case managementauth.AuthSourceIssuer, managementauth.AuthSourceServiceCredential, managementauth.AuthSourceMTLS:
					return nil
				}
			}
		}
	}
	return ErrInvalidBarrier
}

func authSourceBarrierID(kind managementauth.AuthSourceKind, id string) string {
	return string(kind) + ":" + id
}

func barrierField(kind managementauth.BarrierKind, id string) string { return string(kind) + ":" + id }

func canonicalGeneration(value string) bool {
	parsed, err := uuid.Parse(value)
	return err == nil && parsed.String() == value
}

func stringValue(value any) string {
	switch typed := value.(type) {
	case string:
		return typed
	case []byte:
		return string(typed)
	default:
		return ""
	}
}

var checkScript = redisclient.NewScript(`
local generation = redis.call('GET', KEYS[1])
if not generation then return nil end
local hash = ARGV[1] .. generation
return redis.call('HMGET', hash, ARGV[2], ARGV[3], ARGV[4], ARGV[5], ARGV[6], ARGV[7])
`)

var delegationCheckScript = redisclient.NewScript(`
local generation = redis.call('GET', KEYS[1])
if not generation then return nil end
local hash = ARGV[1] .. generation
return redis.call('HMGET', hash, ARGV[2], ARGV[3], ARGV[4])
`)

var releaseLockScript = redisclient.NewScript(`
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
`)

var swapScript = redisclient.NewScript(`
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
local old = redis.call('GET', KEYS[2])
redis.call('SET', KEYS[2], ARGV[2])
if old and old ~= ARGV[2] then
  redis.call('PEXPIRE', ARGV[3] .. old, ARGV[4])
end
return 1
`)
