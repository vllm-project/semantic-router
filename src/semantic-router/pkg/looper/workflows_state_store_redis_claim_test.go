package looper

import (
	"context"
	"encoding/json"
	"errors"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func redisClaimPair(t *testing.T, ttl time.Duration) (*workflowRedisToolStateStore, *workflowRedisToolStateStore) {
	t.Helper()
	mr := miniredis.RunT(t)
	cfg := config.WorkflowStateRedisConfig{
		Address:   mr.Addr(),
		KeyPrefix: "test-redis-claim:",
	}
	stale := newWorkflowRedisToolStateStore(cfg, ttl)
	live := newWorkflowRedisToolStateStore(cfg, ttl)
	t.Cleanup(func() {
		_ = stale.Close()
		_ = live.Close()
	})
	return stale, live
}

func putNamedWorkflowState(t *testing.T, s workflowToolStateStore, id, decision string, iteration int, createdAt time.Time) {
	t.Helper()
	state := makeTestState(id)
	state.DecisionName = decision
	state.Iteration = iteration
	if !createdAt.IsZero() {
		state.CreatedAt = createdAt
	}
	if _, err := s.Put(context.Background(), state); err != nil {
		t.Fatalf("Put %s: %v", decision, err)
	}
}

func advanceRedisSnapshot(t *testing.T, s *workflowRedisToolStateStore, id, decision string, iteration int) {
	t.Helper()
	ctx := context.Background()
	claim, ok, err := s.Claim(ctx, config.DefaultRecipeName, id)
	if err != nil || !ok || claim == nil {
		t.Fatalf("concurrent Claim: ok=%v err=%v", ok, err)
	}
	next := makeTestState(id)
	next.DecisionName = decision
	next.Iteration = iteration
	if replaceErr := s.Replace(ctx, config.DefaultRecipeName, id, claim.Token, next); replaceErr != nil {
		t.Fatalf("concurrent Replace: %v", replaceErr)
	}
}

func durableRedisState(t *testing.T, s *workflowRedisToolStateStore, id string) *workflowPendingToolState {
	t.Helper()
	data, err := s.client.Get(context.Background(), s.key(config.DefaultRecipeName, id)).Bytes()
	if err != nil {
		t.Fatalf("read durable state: %v", err)
	}
	var stored workflowPendingToolState
	if unmarshalErr := json.Unmarshal(data, &stored); unmarshalErr != nil {
		t.Fatalf("parse durable state: %v", unmarshalErr)
	}
	return &stored
}

func runRedisClaimScript(
	t *testing.T,
	s *workflowRedisToolStateStore,
	id, token string,
	claimed, expected []byte,
	mode string,
) interface{} {
	t.Helper()
	result, err := workflowRedisClaimScript.Run(
		context.Background(),
		s.client,
		[]string{s.key(config.DefaultRecipeName, id), s.claimKey(config.DefaultRecipeName, id)},
		token,
		claimed,
		strconv.FormatInt(currentWorkflowStateClaimLease().Milliseconds(), 10),
		expected,
		mode,
		workflowRedisClaimConflict,
	).Result()
	if err != nil && !errors.Is(err, redis.Nil) {
		t.Fatalf("claim script: %v", err)
	}
	return result
}

func TestRedisClaimScript_RefusesStaleSnapshot(t *testing.T) {
	stale, live := redisClaimPair(t, time.Hour)
	const id = "stale-script"
	putNamedWorkflowState(t, stale, id, "S0", 1, time.Time{})
	expected, err := stale.client.Get(context.Background(), stale.key(config.DefaultRecipeName, id)).Bytes()
	if err != nil {
		t.Fatalf("GET S0: %v", err)
	}
	var s0 workflowPendingToolState
	if unmarshalErr := json.Unmarshal(expected, &s0); unmarshalErr != nil {
		t.Fatalf("parse S0: %v", unmarshalErr)
	}
	claimed, err := marshalWorkflowStateWithClaim(&s0, "stale-token", time.Now().UTC())
	if err != nil {
		t.Fatalf("marshal claimed S0: %v", err)
	}

	advanceRedisSnapshot(t, live, id, "S1", 2)

	result := runRedisClaimScript(t, stale, id, "stale-token", claimed, expected, workflowRedisClaimModeClaim)
	if !workflowRedisResultIsConflict(result) {
		t.Fatalf("stale claim script result=%v (%T), want conflict", result, result)
	}
	stored := durableRedisState(t, stale, id)
	if stored.DecisionName != "S1" || stored.Iteration != 2 {
		t.Fatalf("durable state rolled back: %+v", stored)
	}
}

func TestRedisClaimScript_ExpireDeletesOnlyMatchingSnapshot(t *testing.T) {
	stale, live := redisClaimPair(t, 50*time.Millisecond)
	const id = "expire-script"
	putNamedWorkflowState(t, stale, id, "S0", 1, time.Now().UTC().Add(-time.Hour))
	expected, err := stale.client.Get(context.Background(), stale.key(config.DefaultRecipeName, id)).Bytes()
	if err != nil {
		t.Fatalf("GET S0: %v", err)
	}

	putNamedWorkflowState(t, live, id, "S1", 2, time.Now().UTC())

	result := runRedisClaimScript(t, stale, id, "", nil, expected, workflowRedisClaimModeExpire)
	if !workflowRedisResultIsConflict(result) {
		t.Fatalf("expire script result=%v (%T), want conflict", result, result)
	}
	stored := durableRedisState(t, stale, id)
	if stored.DecisionName != "S1" || stored.Iteration != 2 {
		t.Fatalf("expire deleted or rolled back newer snapshot: %+v", stored)
	}

	matched, err := stale.client.Get(context.Background(), stale.key(config.DefaultRecipeName, id)).Bytes()
	if err != nil {
		t.Fatalf("GET S1: %v", err)
	}
	if result := runRedisClaimScript(t, stale, id, "", nil, matched, workflowRedisClaimModeExpire); result != nil {
		t.Fatalf("matching expire result=%v, want false/nil", result)
	}
	if _, err := stale.client.Get(context.Background(), stale.key(config.DefaultRecipeName, id)).Bytes(); !errors.Is(err, redis.Nil) {
		t.Fatalf("matching expire left the key: %v", err)
	}
}

func TestRedisStateStore_ClaimDoesNotRollBackConcurrentResume(t *testing.T) {
	stale, live := redisClaimPair(t, time.Hour)
	const id = "concurrent-resume"
	putNamedWorkflowState(t, stale, id, "S0", 1, time.Time{})

	var once sync.Once
	stale.testClaimInterleave = func() {
		once.Do(func() {
			advanceRedisSnapshot(t, live, id, "S1", 2)
		})
	}

	ctx := context.Background()
	claim, ok, err := stale.Claim(ctx, config.DefaultRecipeName, id)
	if err != nil || !ok || claim == nil || claim.State == nil {
		t.Fatalf("Claim after concurrent resume: ok=%v err=%v", ok, err)
	}
	if claim.State.DecisionName != "S1" || claim.State.Iteration != 2 {
		t.Fatalf("Claim returned older snapshot: %+v", claim.State)
	}
	stored := durableRedisState(t, stale, id)
	if stored.DecisionName != "S1" || stored.Iteration != 2 {
		t.Fatalf("Claim persisted older snapshot %q while returning %q", stored.DecisionName, claim.State.DecisionName)
	}

	claim.State.DecisionName = "mutated-resume"
	claim.State.Iteration = 9
	if releaseErr := stale.Release(ctx, config.DefaultRecipeName, id, claim.Token); releaseErr != nil {
		t.Fatalf("Release after failed resume: %v", releaseErr)
	}

	retried, ok, err := live.Claim(ctx, config.DefaultRecipeName, id)
	if err != nil || !ok || retried == nil || retried.State == nil {
		t.Fatalf("retry Claim: ok=%v err=%v", ok, err)
	}
	if retried.State.DecisionName != "S1" || retried.State.Iteration != 2 {
		t.Fatalf("failed resume rolled durable state back: %+v", retried.State)
	}
	if commitErr := live.Commit(ctx, config.DefaultRecipeName, id, retried.Token); commitErr != nil {
		t.Fatalf("Commit retried claim: %v", commitErr)
	}
}

func TestRedisStateStore_ExpiredClaimDoesNotDeleteNewerSnapshot(t *testing.T) {
	stale, live := redisClaimPair(t, 50*time.Millisecond)
	const id = "expire-interleave"
	putNamedWorkflowState(t, stale, id, "S0", 1, time.Now().UTC().Add(-time.Hour))

	var once sync.Once
	stale.testClaimInterleave = func() {
		once.Do(func() {
			putNamedWorkflowState(t, live, id, "S1", 2, time.Now().UTC())
		})
	}

	ctx := context.Background()
	claim, ok, err := stale.Claim(ctx, config.DefaultRecipeName, id)
	if err != nil || !ok || claim == nil || claim.State == nil {
		t.Fatalf("Claim after newer Put: ok=%v err=%v", ok, err)
	}
	if claim.State.DecisionName != "S1" || claim.State.Iteration != 2 {
		t.Fatalf("Claim returned expired snapshot: %+v", claim.State)
	}
	stored := durableRedisState(t, stale, id)
	if stored.DecisionName != "S1" || stored.Iteration != 2 {
		t.Fatalf("pre-script expire deleted newer snapshot: %+v", stored)
	}
	if releaseErr := stale.Release(ctx, config.DefaultRecipeName, id, claim.Token); releaseErr != nil {
		t.Fatalf("Release: %v", releaseErr)
	}
}

func TestRedisStateStore_ExpiredUnclaimedSnapshotIsDeleted(t *testing.T) {
	s, _ := redisClaimPair(t, 50*time.Millisecond)
	const id = "expire-unclaimed"
	putNamedWorkflowState(t, s, id, "S0", 1, time.Now().UTC().Add(-time.Hour))

	claim, ok, err := s.Claim(context.Background(), config.DefaultRecipeName, id)
	if err != nil {
		t.Fatalf("Claim expired snapshot: %v", err)
	}
	if ok || claim != nil {
		t.Fatal("Claim returned an expired snapshot")
	}
	if _, err := s.client.Get(context.Background(), s.key(config.DefaultRecipeName, id)).Bytes(); !errors.Is(err, redis.Nil) {
		t.Fatalf("expired snapshot still stored: %v", err)
	}
}

func TestRedisStateStore_ExpiredCreatedAtDoesNotDeleteLiveClaim(t *testing.T) {
	s, _ := redisClaimPair(t, 50*time.Millisecond)
	const id = "expire-live"
	putNamedWorkflowState(t, s, id, "S0", 1, time.Time{})

	ctx := context.Background()
	claim, ok, err := s.Claim(ctx, config.DefaultRecipeName, id)
	if err != nil || !ok || claim == nil {
		t.Fatalf("Claim: ok=%v err=%v", ok, err)
	}
	time.Sleep(80 * time.Millisecond)

	busy, held, heldErr := s.Claim(ctx, config.DefaultRecipeName, id)
	if heldErr != nil {
		t.Fatalf("Claim while held: %v", heldErr)
	}
	if held || busy != nil {
		t.Fatal("expired CreatedAt deleted or released a live claim")
	}
	if _, getErr := s.client.Get(ctx, s.key(config.DefaultRecipeName, id)).Bytes(); getErr != nil {
		t.Fatalf("live claim key missing after expired Claim: %v", getErr)
	}
	if releaseErr := s.Release(ctx, config.DefaultRecipeName, id, claim.Token); releaseErr != nil {
		t.Fatalf("Release: %v", releaseErr)
	}
	got, released, releasedErr := s.Claim(ctx, config.DefaultRecipeName, id)
	if releasedErr != nil {
		t.Fatalf("Claim after release: %v", releasedErr)
	}
	if released || got != nil {
		t.Fatal("released snapshot stayed claimable after CreatedAt TTL")
	}
	if _, getErr := s.client.Get(ctx, s.key(config.DefaultRecipeName, id)).Bytes(); !errors.Is(getErr, redis.Nil) {
		t.Fatalf("released expired snapshot still stored: %v", getErr)
	}
}

func TestRedisStateStore_ConcurrentResumesKeepSingleGeneration(t *testing.T) {
	stale, live := redisClaimPair(t, time.Hour)
	ctx := context.Background()

	for i := 0; i < 32; i++ {
		id := "generation-" + strconv.Itoa(i)
		putNamedWorkflowState(t, stale, id, "S0", 0, time.Time{})

		var (
			wg      sync.WaitGroup
			wins    [2]atomic.Bool
			writers = [2]*workflowRedisToolStateStore{stale, live}
			names   = [2]string{"S1", "S2"}
		)
		for worker := 0; worker < 2; worker++ {
			wg.Add(1)
			go func(worker int) {
				defer wg.Done()
				claim, ok, err := writers[worker].Claim(ctx, config.DefaultRecipeName, id)
				if err != nil || !ok || claim == nil {
					return
				}
				next := makeTestState(id)
				next.DecisionName = names[worker]
				next.Iteration = worker + 1
				if replaceErr := writers[worker].Replace(ctx, config.DefaultRecipeName, id, claim.Token, next); replaceErr != nil {
					t.Errorf("worker %d Replace: %v", worker, replaceErr)
					return
				}
				wins[worker].Store(true)
			}(worker)
		}
		wg.Wait()

		got, ok, err := consumeWorkflowState(stale, id)
		if err != nil || !ok || got == nil {
			t.Fatalf("consume generation %d: ok=%v err=%v", i, ok, err)
		}
		switch {
		case wins[0].Load() && got.DecisionName == "S1" && got.Iteration == 1:
		case wins[1].Load() && got.DecisionName == "S2" && got.Iteration == 2:
		case !wins[0].Load() && !wins[1].Load() && got.DecisionName == "S0":
		default:
			t.Fatalf("generation %d durable=%+v wins=[%v %v]", i, got, wins[0].Load(), wins[1].Load())
		}
	}
}
