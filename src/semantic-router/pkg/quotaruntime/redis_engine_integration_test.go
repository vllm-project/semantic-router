package quotaruntime

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
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
		PlanDigest: strings.Repeat("a", 64), BindingIDs: []string{"binding-user"},
	})
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || released.Idempotent {
		t.Fatalf("RemoveReconciledFence() = (%+v, %v), want new release", released, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	released, testRedisEngineAdmissionFinalizationUnknownAndMetersErr = engine.RemoveReconciledFence(context.Background(), FenceRemovalRequest{
		Partition: partition, FenceID: "fence-b", ReconciliationID: "reconciliation-b",
		PlanDigest: strings.Repeat("a", 64), BindingIDs: []string{"binding-user"},
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
