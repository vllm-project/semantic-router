package sessiontools

import (
	"context"
	"maps"
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type memorySnapshot struct {
	states       map[string]State
	keyQuota     map[string]QuotaKey
	quotaMembers map[QuotaKey]map[string]struct{}
	totalCount   int
}

func assertMemoryMembership(t *testing.T, store *MemoryStore, key string, quota QuotaKey) {
	t.Helper()
	snapshot := snapshotMemory(store)
	require.Equal(t, quota, snapshot.keyQuota[key])
	require.Contains(t, snapshot.quotaMembers[quota], key)
	require.Equal(t, 1, snapshot.totalCount)
	require.Len(t, snapshot.keyQuota, 1)
	require.Len(t, snapshot.quotaMembers, 1)
	require.Len(t, snapshot.quotaMembers[quota], 1)
}

// Inspect without Load: sliding expiry is itself a mutation. Tests must catch
// changes to expired entries as well as live state and admission bookkeeping.
func snapshotMemory(store *MemoryStore) memorySnapshot {
	store.admissionMu.Lock()
	defer store.admissionMu.Unlock()
	result := memorySnapshot{
		states: make(map[string]State), keyQuota: maps.Clone(store.keyQuota),
		quotaMembers: make(map[QuotaKey]map[string]struct{}), totalCount: store.totalCount,
	}
	for quota, members := range store.quotaMembers {
		result.quotaMembers[quota] = maps.Clone(members)
	}
	for _, shard := range store.shards {
		shard.mu.Lock()
		for key, entry := range shard.entries {
			result.states[key] = entry.state.Clone()
		}
		shard.mu.Unlock()
	}
	return result
}

func TestMemoryStoreRevisionExhaustionNeverMutatesOrEvicts(t *testing.T) {
	for _, operation := range []string{"update", "new key at quota capacity", "expired recreation"} {
		t.Run(operation, func(t *testing.T) {
			clock := newSyntheticClock(selectionFixture().Now)
			store := newTestStore(t, clock, 1, 1, 60)
			quota := QuotaKey{Principal: "principal", Namespace: "recipe"}
			ctx := context.Background()
			ok, err := store.CompareAndSwap(ctx, "existing", 0, newTestState(0), time.Minute, quota)
			require.NoError(t, err)
			require.True(t, ok)
			store.nextRevision.Store(math.MaxUint64)
			key, revision := "existing", uint64(1)
			switch operation {
			case "new key at quota capacity":
				key, revision = "new", 0
			case "expired recreation":
				clock.Advance(time.Minute)
				revision = 0
			}
			before := snapshotMemory(store)
			ok, err = store.CompareAndSwap(ctx, key, revision, newTestState(0), time.Minute, quota)
			require.False(t, ok)
			require.ErrorIs(t, err, ErrRevisionExhausted)
			require.Equal(t, before, snapshotMemory(store))
			require.Equal(t, uint64(math.MaxUint64), store.nextRevision.Load())
		})
	}
}

func TestMemoryStoreLastRevisionCreationAndUpdate(t *testing.T) {
	for _, update := range []bool{false, true} {
		store := newTestStore(t, newSyntheticClock(selectionFixture().Now), 2, 2, 60)
		ctx, quota := context.Background(), QuotaKey{Principal: "principal", Namespace: "recipe"}
		revision := uint64(0)
		if update {
			ok, err := store.CompareAndSwap(ctx, "key", 0, newTestState(0), time.Minute, quota)
			require.NoError(t, err)
			require.True(t, ok)
			revision = 1
		}
		store.nextRevision.Store(math.MaxUint64 - 1)
		ok, err := store.CompareAndSwap(ctx, "key", revision, newTestState(0), time.Minute, quota)
		require.NoError(t, err)
		require.True(t, ok)
		require.Equal(t, uint64(math.MaxUint64), snapshotMemory(store).states["key"].Revision)
		ok, err = store.CompareAndSwap(ctx, "key", math.MaxUint64, newTestState(0), time.Minute, quota)
		require.False(t, ok)
		require.ErrorIs(t, err, ErrRevisionExhausted)
	}
}

func TestMemoryStoreActualEncodedByteBoundary(t *testing.T) {
	input := selectionFixture()
	state := mergeOK(t, input).State
	state.Revision = 1
	limit, err := state.encodedSize()
	require.NoError(t, err)
	clock := newSyntheticClock(input.Now)
	store := newTestStore(t, clock, 1, 1, 3600)
	store.maxStateBytes = limit
	ctx, quota := context.Background(), QuotaKey{Principal: "principal", Namespace: "recipe"}
	ok, err := store.CompareAndSwap(ctx, "key", 0, state, input.TTL, quota)
	require.NoError(t, err)
	require.True(t, ok)
	before := snapshotMemory(store)
	state.StrategyID += "x" // Exactly one additional serialized byte.
	for _, key := range []string{"key", "new"} {
		revision := uint64(0)
		if key == "key" {
			revision = 1
		}
		ok, err = store.CompareAndSwap(ctx, key, revision, state, input.TTL, quota)
		require.False(t, ok)
		require.ErrorIs(t, err, ErrStateTooLarge)
		require.Equal(t, before, snapshotMemory(store))
	}
	clock.Advance(time.Nanosecond) // Sliding TTL now needs fractional seconds.
	_, err = store.Load(ctx, "key")
	require.ErrorIs(t, err, ErrStateTooLarge)
	require.Equal(t, before, snapshotMemory(store))
}

func TestMemoryStoreDelayedCleanupPreservesRecreatedExpiredSuccessor(t *testing.T) {
	clock := newSyntheticClock(selectionFixture().Now)
	store := newTestStore(t, clock, 2, 1, 60)
	ctx := context.Background()
	oldQuota := QuotaKey{Principal: "old", Namespace: "recipe"}
	newQuota := QuotaKey{Principal: "new", Namespace: "recipe"}
	ok, err := store.CompareAndSwap(ctx, "key", 0, newTestState(0), time.Minute, oldQuota)
	require.NoError(t, err)
	require.True(t, ok)
	observed := snapshotMemory(store).states["key"]
	clock.Advance(time.Minute)
	ok, err = store.CompareAndSwap(ctx, "key", 0, newTestState(0), time.Minute, newQuota)
	require.NoError(t, err)
	require.True(t, ok)
	clock.Advance(time.Minute) // A TTL-only cleanup check would now erase the successor.
	before := snapshotMemory(store)
	store.deleteIfExpiredLocked("key", observed.Revision, observed.ExpiresAt)
	require.Equal(t, before, snapshotMemory(store))
	require.Equal(t, newQuota, before.keyQuota["key"])
	require.NotContains(t, before.quotaMembers, oldQuota)
}

func TestMemoryStoreCanceledOperationsDoNotMutate(t *testing.T) {
	clock := newSyntheticClock(selectionFixture().Now)
	store := newTestStore(t, clock, 2, 2, 60)
	quota := QuotaKey{Principal: "principal", Namespace: "recipe"}
	ok, err := store.CompareAndSwap(context.Background(), "key", 0, newTestState(0), time.Minute, quota)
	require.NoError(t, err)
	require.True(t, ok)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	before := snapshotMemory(store)
	clock.Advance(time.Second)
	_, err = store.Load(ctx, "key")
	require.ErrorIs(t, err, context.Canceled)
	require.ErrorIs(t, store.Delete(ctx, "key"), context.Canceled)
	for _, revision := range []uint64{0, 1} {
		ok, err = store.CompareAndSwap(ctx, "key", revision, newTestState(0), time.Minute, quota)
		require.False(t, ok)
		require.ErrorIs(t, err, context.Canceled)
	}
	require.Equal(t, before, snapshotMemory(store))
	require.Equal(t, uint64(1), store.nextRevision.Load())
}

func TestManagerMemoryStoreLifecycle(t *testing.T) {
	input := selectionFixture()
	clock := newSyntheticClock(input.Now)
	store := newTestStore(t, clock, 2, 2, 60)
	manager, request := managerFixture(t, store)
	manager.options.TTL, manager.options.Clock = time.Minute, clock.Now
	request.Selection.Ranked = request.Selection.Ranked[:1]
	ctx := context.Background()
	first, err := manager.Update(ctx, request)
	require.NoError(t, err)
	require.Equal(t, OutcomeSeeded, first.Receipt.Outcome)
	first.Tools[0].Name = "caller-mutation"
	request.Selection.Ranked = input.Ranked
	second, err := manager.Update(ctx, request)
	require.NoError(t, err)
	require.Equal(t, "a", second.Tools[0].Name)
	require.Len(t, second.Tools, 2)
	stored := snapshotMemory(store).states[request.Key]
	require.Equal(t, uint64(2), stored.Turn)
	clock.Advance(time.Minute)
	third, err := manager.Update(ctx, request)
	require.NoError(t, err)
	require.Equal(t, OutcomeSeeded, third.Receipt.Outcome)
	current := snapshotMemory(store)
	require.Equal(t, uint64(1), current.states[request.Key].Turn)
	require.Greater(t, current.states[request.Key].Revision, stored.Revision)
	require.Equal(t, 1, current.totalCount)
	require.Len(t, current.quotaMembers[request.Quota], 1)
	invalid := current.states[request.Key]
	invalid.SchemaVersion--
	ok, err := store.CompareAndSwap(ctx, request.Key, invalid.Revision, invalid, time.Minute, request.Quota)
	require.NoError(t, err)
	require.True(t, ok)
	reset, err := manager.Update(ctx, request)
	require.NoError(t, err)
	require.Equal(t, ReasonInvalidState, reset.Receipt.Reason)
	assertMemoryMembership(t, store, request.Key, request.Quota)
}
