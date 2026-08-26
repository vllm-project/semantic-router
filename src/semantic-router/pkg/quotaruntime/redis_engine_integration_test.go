package quotaruntime

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

func TestRedisEngineAdmissionFinalizationUnknownAndMeters(t *testing.T) {
	client, partition := integrationRedis(t)
	engine, testRedisEngineAdmissionFinalizationUnknownAndMetersErr := NewRedisEngine(client, RedisEngineOptions{FinalizationMarkerTTL: time.Hour})
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil {
		t.Fatalf("NewRedisEngine() error = %v", testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	rules := []RuleBinding{
		tokenRule(t, "binding-user", "tokens", "10", time.Minute, 0),
		costRule(t, "binding-user", "cost", "5", time.Minute, 1),
		requestRule(t, "binding-user", "requests", "2", time.Minute, 2),
		concurrencyRule(t, "binding-user", "concurrency", "1", 3),
	}
	preconditions, denyKey, credentialKey := seedAccessProjection(t, client, partition)
	first := AdmissionRequest{
		Partition: partition, AdmissionID: "admission-a", Digest: "request-a",
		LeaseDuration: time.Minute, Preconditions: preconditions, Rules: rules,
	}
	blocked := assertAdmissionGuardsAndIdempotency(t, client, engine, first, rules, denyKey, credentialKey)
	tokenIdentity, _ := rules[0].Counter()
	costIdentity, _ := rules[1].Counter()
	assertKnownFinalization(t, client, engine, partition, first, rules, tokenIdentity, costIdentity)
	assertUnknownFinalization(t, client, engine, partition, first, blocked, rules, tokenIdentity, costIdentity)
}

func TestRedisReconciliationClearsExpiredSlidingUnknownAcrossReplicas(t *testing.T) {
	client, partition := integrationRedis(t)
	first, err := NewRedisEngine(client, RedisEngineOptions{FinalizationMarkerTTL: time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewRedisEngine(client, RedisEngineOptions{FinalizationMarkerTTL: time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	rules := []RuleBinding{tokenRule(t, "binding-expired", "tokens", "1000", time.Minute, 0)}
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	identity, err := rules[0].Counter()
	if err != nil {
		t.Fatal(err)
	}

	finalizeSlidingUsage(t, first, AdmissionRequest{
		Partition: partition, AdmissionID: "admission-current", Digest: "request-current",
		LeaseDuration: time.Minute, Preconditions: preconditions, Rules: rules,
	}, "dispatch-current", "usage-current", "", identity, ActualEvidence{
		State: ActualEvidenceKnown, Amount: quotaInteger(t, "7"),
	})
	finalizeSlidingUsage(t, first, AdmissionRequest{
		Partition: partition, AdmissionID: "admission-expired", Digest: "request-expired",
		LeaseDuration: time.Minute, Preconditions: preconditions, Rules: rules,
	}, "dispatch-expired", "usage-expired", "fence-expired", identity, ActualEvidence{
		State: ActualEvidenceUnknown, Reason: "provider_usage_unavailable",
	})

	reconciliation := ReconciliationRequest{
		Partition: partition, FenceID: "fence-expired", AdmissionID: "admission-expired",
		ReconciliationID: "reconciliation-expired", PlanDigest: strings.Repeat("c", 64),
		Event: `{"admissionId":"admission-expired","kind":"correction"}`,
		Corrections: []CounterCorrection{{
			BindingID: "binding-expired", RuleID: "tokens", Metric: quota.MetricTotalTokens,
			Algorithm: quota.AlgorithmSlidingLog, Enforcement: quota.EnforcementEnforce,
			Amount: "5", CounterIncompleteCount: "1", ChargeAt: time.Now().UTC().Add(-time.Hour),
			Window: time.Minute, Charge: true, Known: true,
		}},
	}
	assertConcurrentReconciliation(t, []*RedisEngine{first, second}, reconciliation)

	meters, err := first.ReadMeters(context.Background(), MeterReadRequest{Partition: partition, Rules: rules})
	if err != nil {
		t.Fatal(err)
	}
	corrected := meterByRule(t, meters, "tokens")
	if corrected.Used != "7" || corrected.KnownDispatches != "1" ||
		corrected.IncompleteDispatches != "0" || corrected.CapacityState != quota.CapacityFenced ||
		len(corrected.ActiveFenceIDs) != 1 || corrected.ActiveFenceIDs[0] != "fence-expired" {
		t.Fatalf("corrected expired-window meter = %+v", corrected)
	}

	removal := FenceRemovalRequest{
		Partition: partition, FenceID: "fence-expired", ReconciliationID: "reconciliation-expired",
		PlanDigest: strings.Repeat("c", 64), Counters: []FenceCounter{{
			BindingID: "binding-expired", RuleID: "tokens", Metric: quota.MetricTotalTokens,
		}},
	}
	assertConcurrentFenceRemoval(t, []*RedisEngine{first, second}, removal)
	meters, err = second.ReadMeters(context.Background(), MeterReadRequest{Partition: partition, Rules: rules})
	if err != nil {
		t.Fatal(err)
	}
	released := meterByRule(t, meters, "tokens")
	if released.Used != "7" || released.KnownDispatches != "1" ||
		released.IncompleteDispatches != "0" || released.CapacityState != quota.CapacityAvailable ||
		len(released.ActiveFenceIDs) != 0 {
		t.Fatalf("released expired-window meter = %+v", released)
	}
}

func TestRedisFenceReleasePreservesLastFenceWhenUsageIsIncomplete(t *testing.T) {
	client, partition := integrationRedis(t)
	engine, err := NewRedisEngine(client, RedisEngineOptions{FinalizationMarkerTTL: time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	rules := []RuleBinding{tokenRule(t, "binding-guard", "tokens", "1000", time.Minute, 0)}
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	identity, err := rules[0].Counter()
	if err != nil {
		t.Fatal(err)
	}
	finalizeSlidingUsage(t, engine, AdmissionRequest{
		Partition: partition, AdmissionID: "admission-guard", Digest: "request-guard",
		LeaseDuration: time.Minute, Preconditions: preconditions, Rules: rules,
	}, "dispatch-guard", "usage-guard", "fence-guard", identity, ActualEvidence{
		State: ActualEvidenceUnknown, Reason: "provider_usage_unavailable",
	})
	keys, err := newPartitionKeys(partition)
	if err != nil {
		t.Fatal(err)
	}
	digest := strings.Repeat("d", 64)
	if writeErr := client.HSet(context.Background(), keys.fence("fence-guard"),
		"state", "corrected", "reconciliation_id", "reconciliation-guard",
		"reconciliation_digest", digest).Err(); writeErr != nil {
		t.Fatal(writeErr)
	}
	_, err = engine.RemoveReconciledFence(context.Background(), FenceRemovalRequest{
		Partition: partition, FenceID: "fence-guard", ReconciliationID: "reconciliation-guard",
		PlanDigest: digest, Counters: []FenceCounter{{
			BindingID: "binding-guard", RuleID: "tokens", Metric: quota.MetricTotalTokens,
		}},
	})
	if !errors.Is(err, ErrRuntimeCorrupt) {
		t.Fatalf("RemoveReconciledFence() error = %v, want %v", err, ErrRuntimeCorrupt)
	}
	meters, err := engine.ReadMeters(context.Background(), MeterReadRequest{Partition: partition, Rules: rules})
	if err != nil {
		t.Fatal(err)
	}
	guarded := meterByRule(t, meters, "tokens")
	if guarded.IncompleteDispatches != "1" || guarded.CapacityState != quota.CapacityFenced ||
		len(guarded.ActiveFenceIDs) != 1 || guarded.ActiveFenceIDs[0] != "fence-guard" {
		t.Fatalf("failed release mutated fenced meter = %+v", guarded)
	}
}

func finalizeSlidingUsage(
	t *testing.T,
	engine *RedisEngine,
	request AdmissionRequest,
	dispatchID, finalizationDigest, fenceID string,
	identity quota.CounterIdentity,
	evidence ActualEvidence,
) {
	t.Helper()
	admission, err := engine.Admit(context.Background(), request)
	if err != nil || !admission.Allowed() {
		t.Fatalf("Admit(%s) = %+v, %v", request.AdmissionID, admission, err)
	}
	if _, err := engine.JournalDispatch(context.Background(), DispatchJournalRequest{
		Partition: request.Partition, AdmissionID: request.AdmissionID, AdmissionDigest: request.Digest,
		DispatchID: dispatchID, Digest: "journal-" + dispatchID,
	}); err != nil {
		t.Fatalf("JournalDispatch(%s): %v", request.AdmissionID, err)
	}
	state := usageledger.EvidenceKnown
	if evidence.State == ActualEvidenceUnknown {
		state = usageledger.EvidenceUnknown
	}
	if _, err := engine.Finalize(context.Background(), FinalizationRequest{
		Partition: request.Partition, AdmissionID: request.AdmissionID, AdmissionDigest: request.Digest,
		FinalizationDigest: finalizationDigest, DispatchCount: 1, FenceID: fenceID,
		Event: `{"kind":"terminal"}`, EventEvidenceState: state,
		Rules: request.Rules, Evidence: map[quota.CounterIdentity]ActualEvidence{identity: evidence},
	}); err != nil {
		t.Fatalf("Finalize(%s): %v", request.AdmissionID, err)
	}
}

type concurrentMutationResult struct {
	idempotent bool
	err        error
}

func assertConcurrentReconciliation(
	t *testing.T,
	engines []*RedisEngine,
	request ReconciliationRequest,
) {
	t.Helper()
	results := make(chan concurrentMutationResult, len(engines))
	for _, engine := range engines {
		go func(engine *RedisEngine) {
			result, err := engine.ApplyReconciliation(context.Background(), request)
			results <- concurrentMutationResult{idempotent: result.Idempotent, err: err}
		}(engine)
	}
	assertOneNewMutation(t, results, len(engines), "ApplyReconciliation")
}

func assertConcurrentFenceRemoval(
	t *testing.T,
	engines []*RedisEngine,
	request FenceRemovalRequest,
) {
	t.Helper()
	results := make(chan concurrentMutationResult, len(engines))
	for _, engine := range engines {
		go func(engine *RedisEngine) {
			result, err := engine.RemoveReconciledFence(context.Background(), request)
			results <- concurrentMutationResult{idempotent: result.Idempotent, err: err}
		}(engine)
	}
	assertOneNewMutation(t, results, len(engines), "RemoveReconciledFence")
}

func assertOneNewMutation(
	t *testing.T,
	results <-chan concurrentMutationResult,
	count int,
	operation string,
) {
	t.Helper()
	newMutations := 0
	for index := 0; index < count; index++ {
		result := <-results
		if result.err != nil {
			t.Fatalf("%s concurrent call: %v", operation, result.err)
		}
		if !result.idempotent {
			newMutations++
		}
	}
	if newMutations != 1 {
		t.Fatalf("%s new mutations = %d, want 1", operation, newMutations)
	}
}

func assertAdmissionGuardsAndIdempotency(
	t *testing.T,
	client *redis.Client,
	engine *RedisEngine,
	first AdmissionRequest,
	rules []RuleBinding,
	denyKey, credentialKey string,
) AdmissionRequest {
	t.Helper()
	if err := client.HSet(context.Background(), credentialKey, "status", "revoked").Err(); err != nil {
		t.Fatalf("revoke credential projection: %v", err)
	}
	denied, testRedisEngineAdmissionFinalizationUnknownAndMetersErr := engine.Admit(context.Background(), first)
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || denied.Disposition != AdmissionUnauthenticated {
		t.Fatalf("revoked credential Admit() = (%+v, %v), want unauthenticated", denied, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	if err := client.HSet(context.Background(), credentialKey, "status", "active").Err(); err != nil {
		t.Fatalf("restore credential projection: %v", err)
	}
	if err := client.Set(context.Background(), denyKey, "1", 0).Err(); err != nil {
		t.Fatalf("seed deny barrier: %v", err)
	}
	denied, testRedisEngineAdmissionFinalizationUnknownAndMetersErr = engine.Admit(context.Background(), first)
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || denied.Disposition != AdmissionForbidden {
		t.Fatalf("guarded Admit() = (%+v, %v), want forbidden", denied, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	if err := client.Del(context.Background(), denyKey).Err(); err != nil {
		t.Fatalf("clear deny barrier: %v", err)
	}
	result, testRedisEngineAdmissionFinalizationUnknownAndMetersErr := engine.Admit(context.Background(), first)
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || !result.Allowed() || result.Idempotent {
		t.Fatalf("first Admit() = (%+v, %v), want new allow", result, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	duplicate, testRedisEngineAdmissionFinalizationUnknownAndMetersErr := engine.Admit(context.Background(), first)
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || !duplicate.Allowed() || !duplicate.Idempotent {
		t.Fatalf("duplicate Admit() = (%+v, %v), want idempotent allow", duplicate, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	admissionMeters, testRedisEngineAdmissionFinalizationUnknownAndMetersErr := engine.ReadMeters(context.Background(), MeterReadRequest{
		Partition: first.Partition, Rules: rules,
	})
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil {
		t.Fatalf("ReadMeters() after admission error = %v", testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	if token := meterByRule(t, admissionMeters, "tokens"); token.Used != "0" || token.KnownDispatches != "0" {
		t.Fatalf("token meter was pre-debited at admission: %+v", token)
	}
	if cost := meterByRule(t, admissionMeters, "cost"); cost.Used != "0" || cost.KnownDispatches != "0" {
		t.Fatalf("cost meter was pre-debited at admission: %+v", cost)
	}
	if request := meterByRule(t, admissionMeters, "requests"); request.Used != "1" {
		t.Fatalf("duplicate admission request meter = %+v, want exactly one", request)
	}
	reordered := first
	reordered.Preconditions = append([]AdmissionPrecondition(nil), first.Preconditions...)
	for left, right := 0, len(reordered.Preconditions)-1; left < right; left, right = left+1, right-1 {
		reordered.Preconditions[left], reordered.Preconditions[right] = reordered.Preconditions[right], reordered.Preconditions[left]
	}
	if reorderedResult, err := engine.Admit(context.Background(), reordered); err != nil ||
		!reorderedResult.Allowed() || !reorderedResult.Idempotent {
		t.Fatalf("reordered Admit() = (%+v, %v), want canonical idempotent allow", reorderedResult, err)
	}
	changedPlan := first
	changedPlan.Rules = append([]RuleBinding(nil), first.Rules...)
	changedPlan.Rules[2] = requestRule(t, "binding-user", "requests", "3", time.Minute, 2)
	if _, err := engine.Admit(context.Background(), changedPlan); !errors.Is(err, ErrConflict) {
		t.Fatalf("changed-plan Admit() error = %v, want %v", err, ErrConflict)
	}

	blocked := first
	blocked.AdmissionID = "admission-b"
	blocked.Digest = "request-b"
	result, testRedisEngineAdmissionFinalizationUnknownAndMetersErr = engine.Admit(context.Background(), blocked)
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || result.Disposition != AdmissionRateLimited || result.Limiting == nil ||
		result.Limiting.RuleID != "concurrency" || result.ResetAt == nil {
		t.Fatalf("concurrent Admit() = (%+v, %v), want concurrency denial", result, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}

	release, testRedisEngineAdmissionFinalizationUnknownAndMetersErr := engine.ReleaseConcurrency(context.Background(), ConcurrencyReleaseRequest{
		Partition: first.Partition, AdmissionID: first.AdmissionID, AdmissionDigest: first.Digest, Rules: rules,
	})
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || release.Idempotent {
		t.Fatalf("ReleaseConcurrency() = (%+v, %v), want new release", release, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	release, testRedisEngineAdmissionFinalizationUnknownAndMetersErr = engine.ReleaseConcurrency(context.Background(), ConcurrencyReleaseRequest{
		Partition: first.Partition, AdmissionID: first.AdmissionID, AdmissionDigest: first.Digest, Rules: rules,
	})
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || !release.Idempotent {
		t.Fatalf("duplicate ReleaseConcurrency() = (%+v, %v), want idempotent", release, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	result, testRedisEngineAdmissionFinalizationUnknownAndMetersErr = engine.Admit(context.Background(), blocked)
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || !result.Allowed() {
		t.Fatalf("Admit() after release = (%+v, %v), want allow", result, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	return blocked
}

func assertKnownFinalization(
	t *testing.T,
	client *redis.Client,
	engine *RedisEngine,
	partition string,
	first AdmissionRequest,
	rules []RuleBinding,
	tokenIdentity, costIdentity quota.CounterIdentity,
) {
	t.Helper()
	journal := DispatchJournalRequest{
		Partition: partition, AdmissionID: first.AdmissionID, AdmissionDigest: first.Digest,
		DispatchID: "dispatch-a", Ordinal: 0, Digest: "dispatch-digest-a",
	}
	mutation, testRedisEngineAdmissionFinalizationUnknownAndMetersErr := engine.JournalDispatch(context.Background(), journal)
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || mutation.Idempotent {
		t.Fatalf("JournalDispatch() = (%+v, %v), want new journal", mutation, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	mutation, testRedisEngineAdmissionFinalizationUnknownAndMetersErr = engine.JournalDispatch(context.Background(), journal)
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || !mutation.Idempotent {
		t.Fatalf("duplicate JournalDispatch() = (%+v, %v), want idempotent", mutation, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}

	finalization := FinalizationRequest{
		Partition: partition, AdmissionID: first.AdmissionID, AdmissionDigest: first.Digest,
		FinalizationDigest: "usage-a", DispatchCount: 1,
		Event: `{"admissionId":"admission-a"}`, EventEvidenceState: "known", Rules: rules,
		Evidence: map[quota.CounterIdentity]ActualEvidence{
			tokenIdentity: {State: ActualEvidenceKnown, Amount: quotaInteger(t, "12")},
			costIdentity:  {State: ActualEvidenceKnown, Amount: currencyDecimal(t, "2.5").ScaledInteger()},
		},
	}
	finalized, testRedisEngineAdmissionFinalizationUnknownAndMetersErr := engine.Finalize(context.Background(), finalization)
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || finalized.Idempotent || finalized.EvidenceState != "known" {
		t.Fatalf("Finalize() = (%+v, %v), want new known finalization", finalized, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	partitionKeys, _ := newPartitionKeys(partition)
	for _, key := range []string{
		partitionKeys.pending(first.AdmissionID),
		partitionKeys.dispatches(first.AdmissionID),
		partitionKeys.terminal(first.AdmissionID),
	} {
		ttl, ttlErr := client.PTTL(context.Background(), key).Result()
		if ttlErr != nil || ttl <= 0 || ttl > time.Hour {
			t.Fatalf("terminal key %q PTTL = (%v, %v), want bounded marker TTL", key, ttl, ttlErr)
		}
	}
	finalized, testRedisEngineAdmissionFinalizationUnknownAndMetersErr = engine.Finalize(context.Background(), finalization)
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || !finalized.Idempotent {
		t.Fatalf("duplicate Finalize() = (%+v, %v), want idempotent", finalized, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	conflict := finalization
	conflict.FinalizationDigest = "different-usage"
	if _, err := engine.Finalize(context.Background(), conflict); !errors.Is(err, ErrConflict) {
		t.Fatalf("conflicting Finalize() error = %v, want %v", err, ErrConflict)
	}

	meters, testRedisEngineAdmissionFinalizationUnknownAndMetersErr := engine.ReadMeters(context.Background(), MeterReadRequest{Partition: partition, Rules: rules})
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil {
		t.Fatalf("ReadMeters() error = %v", testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	tokenMeter := meterByRule(t, meters, "tokens")
	if tokenMeter.Used != "12" || tokenMeter.CapacityState != quota.CapacityOverLimit ||
		tokenMeter.Remaining == nil || *tokenMeter.Remaining != "0" {
		t.Fatalf("token meter = %+v, want exact crossing state", tokenMeter)
	}
	costMeter := meterByRule(t, meters, "cost")
	if costMeter.Used != "2.5" || costMeter.Remaining == nil || *costMeter.Remaining != "2.5" {
		t.Fatalf("cost meter = %+v, want exact decimal state", costMeter)
	}
	requestMeter := meterByRule(t, meters, "requests")
	if requestMeter.Used != "2" {
		t.Fatalf("request meter used = %s, want 2 after guarded and atomic denials", requestMeter.Used)
	}

	third := first
	third.AdmissionID = "admission-c"
	third.Digest = "request-c"
	result, testRedisEngineAdmissionFinalizationUnknownAndMetersErr := engine.Admit(context.Background(), third)
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || result.Disposition != AdmissionRateLimited || result.Limiting == nil ||
		result.Limiting.RuleID != "tokens" || result.ResetAt == nil {
		t.Fatalf("post-settlement Admit() = (%+v, %v), want token denial", result, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
}

func assertUnknownFinalization(
	t *testing.T,
	client *redis.Client,
	engine *RedisEngine,
	partition string,
	first, blocked AdmissionRequest,
	rules []RuleBinding,
	tokenIdentity, costIdentity quota.CounterIdentity,
) {
	t.Helper()
	journal := DispatchJournalRequest{
		Partition: partition, AdmissionID: blocked.AdmissionID, AdmissionDigest: blocked.Digest,
		DispatchID: "dispatch-b", Ordinal: 0, Digest: "dispatch-digest-b",
	}
	if _, err := engine.JournalDispatch(context.Background(), journal); err != nil {
		t.Fatalf("JournalDispatch(B) error = %v", err)
	}
	unknown := FinalizationRequest{
		Partition: partition, AdmissionID: blocked.AdmissionID, AdmissionDigest: blocked.Digest,
		FinalizationDigest: "unknown-b", DispatchCount: 1,
		Event: `{"admissionId":"admission-b"}`, EventEvidenceState: "unknown",
		FenceID: "fence-b", Rules: rules,
		Evidence: map[quota.CounterIdentity]ActualEvidence{
			tokenIdentity: {State: ActualEvidenceUnknown, Reason: "provider_usage_unavailable"},
			costIdentity:  {State: ActualEvidenceUnknown, Reason: "provider_usage_unavailable"},
		},
	}
	finalized, testRedisEngineAdmissionFinalizationUnknownAndMetersErr := engine.Finalize(context.Background(), unknown)
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || finalized.Idempotent || finalized.EvidenceState != "unknown" {
		t.Fatalf("Finalize(unknown) = (%+v, %v), want new unknown", finalized, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	finalized, testRedisEngineAdmissionFinalizationUnknownAndMetersErr = engine.Finalize(context.Background(), unknown)
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || !finalized.Idempotent {
		t.Fatalf("duplicate Finalize(unknown) = (%+v, %v), want idempotent", finalized, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	unknownConflict := unknown
	unknownConflict.Evidence = map[quota.CounterIdentity]ActualEvidence{
		tokenIdentity: {State: ActualEvidenceUnknown, Reason: "different_reason"},
		costIdentity:  {State: ActualEvidenceUnknown, Reason: "provider_usage_unavailable"},
	}
	if _, err := engine.Finalize(context.Background(), unknownConflict); !errors.Is(err, ErrConflict) {
		t.Fatalf("conflicting Finalize(unknown) error = %v, want %v", err, ErrConflict)
	}

	fourth := first
	fourth.AdmissionID = "admission-d"
	fourth.Digest = "request-d"
	result, testRedisEngineAdmissionFinalizationUnknownAndMetersErr := engine.Admit(context.Background(), fourth)
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || result.Disposition != AdmissionUnavailable ||
		result.BlockingReason != "binding has unresolved usage" {
		t.Fatalf("fenced Admit() = (%+v, %v), want fence", result, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	meters, testRedisEngineAdmissionFinalizationUnknownAndMetersErr := engine.ReadMeters(context.Background(), MeterReadRequest{Partition: partition, Rules: rules})
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil {
		t.Fatalf("ReadMeters() after unknown error = %v", testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	tokenMeter := meterByRule(t, meters, "tokens")
	if tokenMeter.Completeness != quota.CompletenessPartial ||
		tokenMeter.CapacityState != quota.CapacityFenced || tokenMeter.Remaining != nil ||
		len(tokenMeter.ActiveFenceIDs) != 1 || tokenMeter.ActiveFenceIDs[0] != "fence-b" {
		t.Fatalf("fenced token meter = %+v, want partial fenced capacity", tokenMeter)
	}
	assertUsageReconciliation(t, engine, partition, blocked, rules)
}

func assertUsageReconciliation(
	t *testing.T,
	engine *RedisEngine,
	partition string,
	blocked AdmissionRequest,
	rules []RuleBinding,
) {
	t.Helper()
	reconciliation := ReconciliationRequest{
		Partition: partition, FenceID: "fence-b", AdmissionID: blocked.AdmissionID,
		ReconciliationID: "reconciliation-b", PlanDigest: strings.Repeat("a", 64),
		Event: `{"admissionId":"admission-b","kind":"correction"}`,
		Corrections: []CounterCorrection{
			{
				BindingID: "binding-user", RuleID: "tokens", Metric: quota.MetricTotalTokens,
				Algorithm: quota.AlgorithmSlidingLog, Enforcement: quota.EnforcementEnforce,
				Amount: "3", CounterIncompleteCount: "1", ChargeAt: time.Now().UTC(),
				Window: time.Minute, Charge: true, Known: true,
			},
			{
				BindingID: "binding-user", RuleID: "cost", Metric: quota.MetricCost,
				Algorithm: quota.AlgorithmSlidingLog, Enforcement: quota.EnforcementEnforce,
				Amount: "1000000000000000", CounterIncompleteCount: "1", ChargeAt: time.Now().UTC(),
				Window: time.Minute, Charge: true, Known: true,
			},
		},
	}
	corrected, testRedisEngineAdmissionFinalizationUnknownAndMetersErr := engine.ApplyReconciliation(context.Background(), reconciliation)
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || corrected.Idempotent || corrected.StreamID == "" {
		t.Fatalf("ApplyReconciliation() = (%+v, %v), want new correction", corrected, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	corrected, testRedisEngineAdmissionFinalizationUnknownAndMetersErr = engine.ApplyReconciliation(context.Background(), reconciliation)
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || !corrected.Idempotent {
		t.Fatalf("duplicate ApplyReconciliation() = (%+v, %v), want idempotent", corrected, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	conflictingReconciliation := reconciliation
	conflictingReconciliation.PlanDigest = strings.Repeat("b", 64)
	if _, err := engine.ApplyReconciliation(context.Background(), conflictingReconciliation); !errors.Is(err, ErrConflict) {
		t.Fatalf("conflicting ApplyReconciliation() error = %v, want %v", err, ErrConflict)
	}
	meters, testRedisEngineAdmissionFinalizationUnknownAndMetersErr := engine.ReadMeters(context.Background(), MeterReadRequest{Partition: partition, Rules: rules})
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || meterByRule(t, meters, "tokens").CapacityState != quota.CapacityFenced {
		t.Fatalf("corrected-before-ledger meters = (%+v, %v), want fence retained", meters, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	released, testRedisEngineAdmissionFinalizationUnknownAndMetersErr := engine.RemoveReconciledFence(context.Background(), FenceRemovalRequest{
		Partition: partition, FenceID: "fence-b", ReconciliationID: "reconciliation-b",
		PlanDigest: strings.Repeat("a", 64), Counters: []FenceCounter{
			{BindingID: "binding-user", RuleID: "tokens", Metric: quota.MetricTotalTokens},
			{BindingID: "binding-user", RuleID: "cost", Metric: quota.MetricCost},
		},
	})
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || released.Idempotent {
		t.Fatalf("RemoveReconciledFence() = (%+v, %v), want new release", released, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	released, testRedisEngineAdmissionFinalizationUnknownAndMetersErr = engine.RemoveReconciledFence(context.Background(), FenceRemovalRequest{
		Partition: partition, FenceID: "fence-b", ReconciliationID: "reconciliation-b",
		PlanDigest: strings.Repeat("a", 64), Counters: []FenceCounter{
			{BindingID: "binding-user", RuleID: "tokens", Metric: quota.MetricTotalTokens},
			{BindingID: "binding-user", RuleID: "cost", Metric: quota.MetricCost},
		},
	})
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || !released.Idempotent {
		t.Fatalf("duplicate RemoveReconciledFence() = (%+v, %v), want idempotent", released, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	meters, testRedisEngineAdmissionFinalizationUnknownAndMetersErr = engine.ReadMeters(context.Background(), MeterReadRequest{Partition: partition, Rules: rules})
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil {
		t.Fatalf("ReadMeters() after reconciliation error = %v", testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	tokenMeter := meterByRule(t, meters, "tokens")
	if tokenMeter.Used != "15" || tokenMeter.IncompleteDispatches != "0" ||
		tokenMeter.KnownDispatches != "2" || len(tokenMeter.ActiveFenceIDs) != 0 {
		t.Fatalf("reconciled token meter = %+v", tokenMeter)
	}
	if costMeter := meterByRule(t, meters, "cost"); costMeter.Used != "3.5" ||
		costMeter.IncompleteDispatches != "0" || costMeter.KnownDispatches != "2" {
		t.Fatalf("reconciled cost meter = %+v", costMeter)
	}
}
