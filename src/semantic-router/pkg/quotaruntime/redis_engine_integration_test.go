package quotaruntime

import (
	"context"
	"errors"
	"fmt"
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

func TestRedisEngineEightHourActualCostWindowBlocksOnlyAfterSettlement(t *testing.T) {
	client, partition := integrationRedis(t)
	engine, testRedisEngineEightHourActualCostWindowBlocksOnlyAfterSettlementErr := NewRedisEngine(client, RedisEngineOptions{})
	if testRedisEngineEightHourActualCostWindowBlocksOnlyAfterSettlementErr != nil {
		t.Fatalf("NewRedisEngine() error = %v", testRedisEngineEightHourActualCostWindowBlocksOnlyAfterSettlementErr)
	}
	rules := []RuleBinding{costRule(t, "binding-user", "eight-hour-cost", "1", 8*time.Hour, 0)}
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	first := AdmissionRequest{
		Partition: partition, AdmissionID: "eight-hour-cost-a", Digest: "request-a",
		LeaseDuration: time.Minute, Preconditions: preconditions, Rules: rules,
	}
	admission, testRedisEngineEightHourActualCostWindowBlocksOnlyAfterSettlementErr := engine.Admit(context.Background(), first)
	if testRedisEngineEightHourActualCostWindowBlocksOnlyAfterSettlementErr != nil || !admission.Allowed() {
		t.Fatalf("Admit() before actual cost = (%+v, %v), want allow", admission, testRedisEngineEightHourActualCostWindowBlocksOnlyAfterSettlementErr)
	}
	meters, testRedisEngineEightHourActualCostWindowBlocksOnlyAfterSettlementErr := engine.ReadMeters(context.Background(), MeterReadRequest{Partition: partition, Rules: rules})
	if testRedisEngineEightHourActualCostWindowBlocksOnlyAfterSettlementErr != nil {
		t.Fatalf("ReadMeters() before settlement error = %v", testRedisEngineEightHourActualCostWindowBlocksOnlyAfterSettlementErr)
	}
	if meter := meterByRule(t, meters, "eight-hour-cost"); meter.Used != "0" {
		t.Fatalf("cost was pre-debited at admission: %+v", meter)
	}

	journal := DispatchJournalRequest{
		Partition: partition, AdmissionID: first.AdmissionID, AdmissionDigest: first.Digest,
		DispatchID: "eight-hour-cost-dispatch", Ordinal: 0, Digest: "dispatch-digest",
	}
	if _, err := engine.JournalDispatch(context.Background(), journal); err != nil {
		t.Fatalf("JournalDispatch() error = %v", err)
	}
	identity, _ := rules[0].Counter()
	finalization := FinalizationRequest{
		Partition: partition, AdmissionID: first.AdmissionID, AdmissionDigest: first.Digest,
		FinalizationDigest: "eight-hour-cost-usage", DispatchCount: 1,
		Event: `{"admissionId":"eight-hour-cost-a"}`, EventEvidenceState: "known", Rules: rules,
		Evidence: map[quota.CounterIdentity]ActualEvidence{
			identity: {State: ActualEvidenceKnown, Amount: currencyDecimal(t, "1.25").ScaledInteger()},
		},
	}
	if result, err := engine.Finalize(context.Background(), finalization); err != nil || result.Idempotent {
		t.Fatalf("Finalize() = (%+v, %v), want new actual-cost settlement", result, err)
	}
	meters, testRedisEngineEightHourActualCostWindowBlocksOnlyAfterSettlementErr = engine.ReadMeters(context.Background(), MeterReadRequest{Partition: partition, Rules: rules})
	if testRedisEngineEightHourActualCostWindowBlocksOnlyAfterSettlementErr != nil {
		t.Fatalf("ReadMeters() after settlement error = %v", testRedisEngineEightHourActualCostWindowBlocksOnlyAfterSettlementErr)
	}
	meter := meterByRule(t, meters, "eight-hour-cost")
	if meter.Used != "1.25" || meter.CapacityState != quota.CapacityOverLimit ||
		meter.Remaining == nil || *meter.Remaining != "0" {
		t.Fatalf("settled eight-hour cost meter = %+v, want exact over-limit state", meter)
	}

	second := first
	second.AdmissionID = "eight-hour-cost-b"
	second.Digest = "request-b"
	result, testRedisEngineEightHourActualCostWindowBlocksOnlyAfterSettlementErr := engine.Admit(context.Background(), second)
	if testRedisEngineEightHourActualCostWindowBlocksOnlyAfterSettlementErr != nil || result.Disposition != AdmissionRateLimited || result.Limiting == nil ||
		result.Limiting.RuleID != "eight-hour-cost" || result.ResetAt == nil {
		t.Fatalf("Admit() after actual-cost crossing = (%+v, %v), want cost denial", result, testRedisEngineEightHourActualCostWindowBlocksOnlyAfterSettlementErr)
	}
}

func TestRedisEngineAppliesGlobalUsageBackpressureWithoutBreakingIdempotency(t *testing.T) {
	client, partition := integrationRedis(t)
	engine, testRedisEngineAppliesGlobalUsageBackpressureWithoutBreakingIdempotencyErr := NewRedisEngine(client, RedisEngineOptions{MaxUsageBacklog: 1})
	if testRedisEngineAppliesGlobalUsageBackpressureWithoutBreakingIdempotencyErr != nil {
		t.Fatalf("NewRedisEngine() error = %v", testRedisEngineAppliesGlobalUsageBackpressureWithoutBreakingIdempotencyErr)
	}
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	request := AdmissionRequest{
		Partition: partition, AdmissionID: "before-backlog", Digest: "before-backlog",
		LeaseDuration: time.Minute, Preconditions: preconditions,
	}
	first, testRedisEngineAppliesGlobalUsageBackpressureWithoutBreakingIdempotencyErr := engine.Admit(context.Background(), request)
	if testRedisEngineAppliesGlobalUsageBackpressureWithoutBreakingIdempotencyErr != nil || !first.Allowed() || first.Idempotent {
		t.Fatalf("first Admit() = (%+v, %v), want new allow", first, testRedisEngineAppliesGlobalUsageBackpressureWithoutBreakingIdempotencyErr)
	}

	keys, _ := newPartitionKeys(partition)
	if err := client.XAdd(context.Background(), &redis.XAddArgs{
		Stream: keys.usageStream, Values: map[string]any{"event": "settled"},
	}).Err(); err != nil {
		t.Fatalf("seed usage backlog: %v", err)
	}

	duplicate, testRedisEngineAppliesGlobalUsageBackpressureWithoutBreakingIdempotencyErr := engine.Admit(context.Background(), request)
	if testRedisEngineAppliesGlobalUsageBackpressureWithoutBreakingIdempotencyErr != nil || !duplicate.Allowed() || !duplicate.Idempotent {
		t.Fatalf("idempotent Admit() = (%+v, %v), want existing admission", duplicate, testRedisEngineAppliesGlobalUsageBackpressureWithoutBreakingIdempotencyErr)
	}
	blockedRequest := request
	blockedRequest.AdmissionID = "after-backlog"
	blockedRequest.Digest = "after-backlog"
	blocked, testRedisEngineAppliesGlobalUsageBackpressureWithoutBreakingIdempotencyErr := engine.Admit(context.Background(), blockedRequest)
	if testRedisEngineAppliesGlobalUsageBackpressureWithoutBreakingIdempotencyErr != nil || blocked.Disposition != AdmissionUnavailable ||
		blocked.BlockingReason != "usage accounting backlog is full" {
		t.Fatalf("backpressured Admit() = (%+v, %v), want unavailable", blocked, testRedisEngineAppliesGlobalUsageBackpressureWithoutBreakingIdempotencyErr)
	}
	if exists, existsErr := client.Exists(context.Background(), keys.pending(blockedRequest.AdmissionID)).Result(); existsErr != nil || exists != 0 {
		t.Fatalf("backpressured admission pending state = (%d, %v), want absent", exists, existsErr)
	}

	diagnostics, err := NewRedisDiagnostics(client, "")
	if err != nil {
		t.Fatal(err)
	}
	beforeRead, err := diagnostics.Snapshot(context.Background(), partition)
	if err != nil || beforeRead.UsageStreamBacklog != 1 {
		t.Fatalf("undelivered usage backlog = (%+v, %v), want 1", beforeRead, err)
	}
	messages, err := client.XReadGroup(context.Background(), &redis.XReadGroupArgs{
		Group: usageledger.ConsumerGroupName, Consumer: "writer-a",
		Streams: []string{keys.usageStream, ">"}, Count: 1,
	}).Result()
	if err != nil || len(messages) != 1 || len(messages[0].Messages) != 1 {
		t.Fatalf("read usage backlog = (%+v, %v)", messages, err)
	}
	pendingSnapshot, err := diagnostics.Snapshot(context.Background(), partition)
	if err != nil || pendingSnapshot.UsageStreamBacklog != 1 {
		t.Fatalf("pending usage backlog = (%+v, %v), want 1", pendingSnapshot, err)
	}
	if err := client.XAdd(context.Background(), &redis.XAddArgs{
		Stream: keys.usageStream, Values: map[string]any{"event": "second-settlement"},
	}).Err(); err != nil {
		t.Fatalf("seed combined usage backlog: %v", err)
	}
	combined, err := diagnostics.Snapshot(context.Background(), partition)
	if err != nil || combined.UsageStreamBacklog != 2 {
		t.Fatalf("combined lag plus pending backlog = (%+v, %v), want 2", combined, err)
	}
	messageID := messages[0].Messages[0].ID
	if err := client.XAck(context.Background(), keys.usageStream, usageledger.ConsumerGroupName, messageID).Err(); err != nil {
		t.Fatalf("acknowledge retained usage item: %v", err)
	}
	afterAck, err := diagnostics.Snapshot(context.Background(), partition)
	if err != nil || afterAck.UsageStreamBacklog != 1 {
		t.Fatalf("backlog after acknowledging one retained item = (%+v, %v), want 1", afterAck, err)
	}
	messages, err = client.XReadGroup(context.Background(), &redis.XReadGroupArgs{
		Group: usageledger.ConsumerGroupName, Consumer: "writer-a",
		Streams: []string{keys.usageStream, ">"}, Count: 1,
	}).Result()
	if err != nil || len(messages) != 1 || len(messages[0].Messages) != 1 {
		t.Fatalf("read remaining usage backlog = (%+v, %v)", messages, err)
	}
	if err := client.XAck(
		context.Background(), keys.usageStream, usageledger.ConsumerGroupName, messages[0].Messages[0].ID,
	).Err(); err != nil {
		t.Fatalf("acknowledge remaining retained usage item: %v", err)
	}
	afterAck, err = diagnostics.Snapshot(context.Background(), partition)
	if err != nil || afterAck.UsageStreamBacklog != 0 {
		t.Fatalf("fully acknowledged usage backlog = (%+v, %v), want 0", afterAck, err)
	}
	if length, lengthErr := client.XLen(context.Background(), keys.usageStream).Result(); lengthErr != nil || length != 2 {
		t.Fatalf("retained stream history = (%d, %v), want two acknowledged items", length, lengthErr)
	}
	resumedRequest := request
	resumedRequest.AdmissionID = "after-acknowledgement"
	resumedRequest.Digest = "after-acknowledgement"
	resumed, err := engine.Admit(context.Background(), resumedRequest)
	if err != nil || !resumed.Allowed() {
		t.Fatalf("Admit() after acknowledgement = (%+v, %v), want allowed", resumed, err)
	}
}

func TestRedisEngineFailsClosedWhenUsageConsumerGroupIsUnavailable(t *testing.T) {
	client, partition := integrationRedis(t)
	engine, err := NewRedisEngine(client, RedisEngineOptions{MaxUsageBacklog: 10})
	if err != nil {
		t.Fatal(err)
	}
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	keys, _ := newPartitionKeys(partition)
	if removed, err := client.XGroupDestroy(
		context.Background(), keys.usageStream, usageledger.ConsumerGroupName,
	).Result(); err != nil || removed != 1 {
		t.Fatalf("remove usage consumer group = (%v, %v), want removed", removed, err)
	}
	if err := client.XAdd(context.Background(), &redis.XAddArgs{
		Stream: keys.usageStream, Values: map[string]any{"event": "unconsumed"},
	}).Err(); err != nil {
		t.Fatal(err)
	}
	request := AdmissionRequest{
		Partition: partition, AdmissionID: "missing-consumer-group", Digest: "missing-consumer-group",
		LeaseDuration: time.Minute, Preconditions: preconditions,
	}
	result, err := engine.Admit(context.Background(), request)
	if err != nil || result.Disposition != AdmissionUnavailable ||
		result.BlockingReason != "usage accounting consumer group is unavailable" {
		t.Fatalf("Admit() without usage consumer group = (%+v, %v), want unavailable", result, err)
	}
	diagnostics, err := NewRedisDiagnostics(client, "")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := diagnostics.Snapshot(context.Background(), partition); !errors.Is(err, ErrRuntimeUnavailable) {
		t.Fatalf("Snapshot() without usage consumer group error = %v, want %v", err, ErrRuntimeUnavailable)
	}
}

func TestRedisEngineMultiRuleDenialDoesNotPartiallyConsume(t *testing.T) {
	client, partition := integrationRedis(t)
	engine, testRedisEngineMultiRuleDenialDoesNotPartiallyConsumeErr := NewRedisEngine(client, RedisEngineOptions{})
	if testRedisEngineMultiRuleDenialDoesNotPartiallyConsumeErr != nil {
		t.Fatalf("NewRedisEngine() error = %v", testRedisEngineMultiRuleDenialDoesNotPartiallyConsumeErr)
	}
	rules := []RuleBinding{
		requestRule(t, "binding-a", "wide", "2", time.Minute, 0),
		requestRule(t, "binding-b", "narrow", "1", time.Minute, 1),
	}
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	first := AdmissionRequest{
		Partition: partition, AdmissionID: "atomic-a", Digest: "atomic-a",
		LeaseDuration: time.Minute, Preconditions: preconditions, Rules: rules,
	}
	if result, err := engine.Admit(context.Background(), first); err != nil || !result.Allowed() {
		t.Fatalf("first Admit() = (%+v, %v), want allow", result, err)
	}
	second := first
	second.AdmissionID = "atomic-b"
	second.Digest = "atomic-b"
	if result, err := engine.Admit(context.Background(), second); err != nil || result.Disposition != AdmissionRateLimited {
		t.Fatalf("second Admit() = (%+v, %v), want denial", result, err)
	}
	meters, testRedisEngineMultiRuleDenialDoesNotPartiallyConsumeErr := engine.ReadMeters(context.Background(), MeterReadRequest{Partition: partition, Rules: rules})
	if testRedisEngineMultiRuleDenialDoesNotPartiallyConsumeErr != nil {
		t.Fatalf("ReadMeters() error = %v", testRedisEngineMultiRuleDenialDoesNotPartiallyConsumeErr)
	}
	if wide := meterByRule(t, meters, "wide"); wide.Used != "1" {
		t.Fatalf("wide rule used = %s, want 1 after atomic denial", wide.Used)
	}
}

func TestRedisEngineRequestLimitIsGlobalAcrossReplicasAndMatchesLiveMeter(t *testing.T) {
	client, partition := integrationRedis(t)
	firstReplica, err := NewRedisEngine(client, RedisEngineOptions{})
	if err != nil {
		t.Fatalf("NewRedisEngine(first replica) error = %v", err)
	}
	secondReplica, err := NewRedisEngine(client, RedisEngineOptions{})
	if err != nil {
		t.Fatalf("NewRedisEngine(second replica) error = %v", err)
	}
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	rules := []RuleBinding{requestRule(t, "binding-user", "rpm", "12", time.Minute, 0)}

	for index := 0; index < 12; index++ {
		engine := firstReplica
		if index%2 != 0 {
			engine = secondReplica
		}
		requestID := fmt.Sprintf("global-rpm-%d", index)
		result, admitErr := engine.Admit(context.Background(), AdmissionRequest{
			Partition: partition, AdmissionID: requestID, Digest: requestID,
			LeaseDuration: time.Minute, Preconditions: preconditions, Rules: rules,
		})
		if admitErr != nil || !result.Allowed() {
			t.Fatalf("Admit(%d) = (%+v, %v), want one of twelve global allows", index, result, admitErr)
		}
	}

	meters, err := secondReplica.ReadMeters(context.Background(), MeterReadRequest{
		Partition: partition, Rules: rules,
	})
	if err != nil {
		t.Fatalf("ReadMeters() error = %v", err)
	}
	meter := meterByRule(t, meters, "rpm")
	if meter.Used != "12" || meter.Remaining == nil || *meter.Remaining != "0" ||
		meter.CapacityState != quota.CapacityExhausted {
		t.Fatalf("global RPM meter = %+v, want used 12, remaining 0, exhausted", meter)
	}

	result, err := firstReplica.Admit(context.Background(), AdmissionRequest{
		Partition: partition, AdmissionID: "global-rpm-13", Digest: "global-rpm-13",
		LeaseDuration: time.Minute, Preconditions: preconditions, Rules: rules,
	})
	if err != nil || result.Disposition != AdmissionRateLimited || result.Limiting == nil ||
		result.Limiting.RuleID != "rpm" || result.ResetAt == nil {
		t.Fatalf("thirteenth cross-replica Admit() = (%+v, %v), want RPM denial", result, err)
	}
	meters, err = firstReplica.ReadMeters(context.Background(), MeterReadRequest{
		Partition: partition, Rules: rules,
	})
	if err != nil {
		t.Fatalf("ReadMeters() after denial error = %v", err)
	}
	if meter = meterByRule(t, meters, "rpm"); meter.Used != "12" {
		t.Fatalf("RPM denial consumed quota: %+v", meter)
	}
}

func TestRedisEngineHeartbeatRenewsGlobalConcurrencyLeaseAndStopsAtSettlement(t *testing.T) {
	client, partition := integrationRedis(t)
	admittingReplica, err := NewRedisEngine(client, RedisEngineOptions{})
	if err != nil {
		t.Fatalf("NewRedisEngine(admitting replica) error = %v", err)
	}
	heartbeatReplica, err := NewRedisEngine(client, RedisEngineOptions{})
	if err != nil {
		t.Fatalf("NewRedisEngine(heartbeat replica) error = %v", err)
	}
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	rules := []RuleBinding{concurrencyRule(t, "binding-user", "concurrency", "1", 0)}
	request := AdmissionRequest{
		Partition: partition, AdmissionID: "heartbeat-admission", Digest: "heartbeat-request",
		LeaseDuration: 500 * time.Millisecond, Preconditions: preconditions, Rules: rules,
	}
	admission, err := admittingReplica.Admit(context.Background(), request)
	if err != nil || !admission.Allowed() || admission.PlanDigest == "" {
		t.Fatalf("Admit() = (%+v, %v), want plan-bound allow", admission, err)
	}
	time.Sleep(10 * time.Millisecond)
	heartbeatRequest := AdmissionHeartbeatRequest{
		Partition: partition, AdmissionID: request.AdmissionID, AdmissionDigest: request.Digest,
		PlanDigest: admission.PlanDigest, LeaseDuration: request.LeaseDuration, Rules: rules,
	}
	renewed, err := heartbeatReplica.Heartbeat(context.Background(), heartbeatRequest)
	if err != nil || renewed.Stopped || !renewed.Deadline.After(admission.Deadline) {
		t.Fatalf("cross-replica Heartbeat() = (%+v, %v), want extended lease", renewed, err)
	}
	partitionKeys, _ := newPartitionKeys(partition)
	compiled, err := compileRules(partition, rules)
	if err != nil {
		t.Fatal(err)
	}
	for label, key := range map[string]string{
		"pending":     partitionKeys.pendingIndex,
		"concurrency": compiled[0].keys.events,
	} {
		score, scoreErr := client.ZScore(context.Background(), key, request.AdmissionID).Result()
		if scoreErr != nil || int64(score) != renewed.Deadline.UnixMilli() {
			t.Fatalf("%s heartbeat deadline = (%v, %v), want %d", label, score, scoreErr, renewed.Deadline.UnixMilli())
		}
	}

	wrongPlan := heartbeatRequest
	wrongPlan.PlanDigest = "0" + admission.PlanDigest[1:]
	if wrongPlan.PlanDigest == admission.PlanDigest {
		wrongPlan.PlanDigest = "1" + admission.PlanDigest[1:]
	}
	if _, heartbeatErr := admittingReplica.Heartbeat(context.Background(), wrongPlan); !errors.Is(heartbeatErr, ErrConflict) {
		t.Fatalf("retargeted Heartbeat() error = %v, want %v", heartbeatErr, ErrConflict)
	}
	if _, journalErr := admittingReplica.JournalDispatch(context.Background(), DispatchJournalRequest{
		Partition: partition, AdmissionID: request.AdmissionID, AdmissionDigest: request.Digest,
		DispatchID: "heartbeat-dispatch", Ordinal: 0, Digest: "heartbeat-dispatch-plan",
	}); journalErr != nil {
		t.Fatalf("JournalDispatch() error = %v", journalErr)
	}
	if _, finalizeErr := admittingReplica.Finalize(context.Background(), FinalizationRequest{
		Partition: partition, AdmissionID: request.AdmissionID, AdmissionDigest: request.Digest,
		FinalizationDigest: "heartbeat-final", DispatchCount: 1,
		Event: `{"admissionId":"heartbeat-admission"}`, EventEvidenceState: "known", Rules: rules,
	}); finalizeErr != nil {
		t.Fatalf("Finalize() error = %v", finalizeErr)
	}
	stopped, err := heartbeatReplica.Heartbeat(context.Background(), heartbeatRequest)
	if err != nil || !stopped.Stopped || !stopped.Deadline.IsZero() {
		t.Fatalf("terminal Heartbeat() = (%+v, %v), want idempotent stop", stopped, err)
	}
}

func TestRedisEngineSlidingWindowUsesServerTimeAndPendingDoesNotExpireSilently(t *testing.T) {
	client, partition := integrationRedis(t)
	engine, err := NewRedisEngine(client, RedisEngineOptions{})
	if err != nil {
		t.Fatalf("NewRedisEngine() error = %v", err)
	}
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	rules := []RuleBinding{requestRule(t, "binding-window", "rpm", "1", time.Second, 0)}
	first := AdmissionRequest{
		Partition: partition, AdmissionID: "window-a", Digest: "window-a",
		LeaseDuration: 5 * time.Second, Preconditions: preconditions, Rules: rules,
	}
	if result, err := engine.Admit(context.Background(), first); err != nil || !result.Allowed() {
		t.Fatalf("first Admit() = (%+v, %v), want allow", result, err)
	}
	partitionKeys, _ := newPartitionKeys(partition)
	if ttl, err := client.PTTL(context.Background(), partitionKeys.pending(first.AdmissionID)).Result(); err != nil || ttl != -1 {
		t.Fatalf("pending admission PTTL = (%v, %v), want -1 (no silent expiry)", ttl, err)
	}
	second := first
	second.AdmissionID = "window-b"
	second.Digest = "window-b"
	if result, err := engine.Admit(context.Background(), second); err != nil ||
		result.Disposition != AdmissionRateLimited || result.ResetAt == nil {
		t.Fatalf("second Admit() = (%+v, %v), want sliding-window denial", result, err)
	}
	time.Sleep(1100 * time.Millisecond)
	if result, err := engine.Admit(context.Background(), second); err != nil || !result.Allowed() {
		t.Fatalf("post-window Admit() = (%+v, %v), want allow", result, err)
	}
}

func TestRedisEngineActualRetryWaitsUntilCapacityReturns(t *testing.T) {
	client, partition := integrationRedis(t)
	engine, err := NewRedisEngine(client, RedisEngineOptions{})
	if err != nil {
		t.Fatalf("NewRedisEngine() error = %v", err)
	}
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	rules := []RuleBinding{tokenRule(t, "binding", "tokens", "10", 10*time.Second, 0)}
	identity, _ := rules[0].Counter()
	for index, amount := range []string{"3", "3", "8"} {
		admissionID := fmt.Sprintf("weighted-%d", index)
		admission := AdmissionRequest{
			Partition: partition, AdmissionID: admissionID, Digest: admissionID,
			LeaseDuration: time.Minute, Preconditions: preconditions, Rules: rules,
		}
		if result, admitErr := engine.Admit(context.Background(), admission); admitErr != nil || !result.Allowed() {
			t.Fatalf("Admit(%d) = (%+v, %v), want allow", index, result, admitErr)
		}
		if _, journalErr := engine.JournalDispatch(context.Background(), DispatchJournalRequest{
			Partition: partition, AdmissionID: admissionID, AdmissionDigest: admissionID,
			DispatchID: admissionID, Digest: admissionID,
		}); journalErr != nil {
			t.Fatalf("JournalDispatch(%d) error = %v", index, journalErr)
		}
		if _, settleErr := engine.Finalize(context.Background(), FinalizationRequest{
			Partition: partition, AdmissionID: admissionID, AdmissionDigest: admissionID,
			FinalizationDigest: admissionID, DispatchCount: 1,
			Event: `{"admissionId":"` + admissionID + `"}`, EventEvidenceState: "known",
			Rules: rules,
			Evidence: map[quota.CounterIdentity]ActualEvidence{
				identity: {State: ActualEvidenceKnown, Amount: quotaInteger(t, amount)},
			},
		}); settleErr != nil {
			t.Fatalf("Finalize(%d) error = %v", index, settleErr)
		}
		time.Sleep(10 * time.Millisecond)
	}
	compiled, err := compileRules(partition, rules)
	if err != nil {
		t.Fatalf("compileRules() error = %v", err)
	}
	events, err := client.ZRangeWithScores(context.Background(), compiled[0].keys.events, 0, -1).Result()
	if err != nil || len(events) != 3 {
		t.Fatalf("weighted events = (%v, %v), want three", events, err)
	}
	expectedReset := time.UnixMilli(int64(events[1].Score) + 10_000).UTC()
	result, err := engine.Admit(context.Background(), AdmissionRequest{
		Partition: partition, AdmissionID: "weighted-denied", Digest: "weighted-denied",
		LeaseDuration: time.Minute, Preconditions: preconditions, Rules: rules,
	})
	if err != nil || result.Disposition != AdmissionRateLimited || result.ResetAt == nil ||
		!result.ResetAt.Equal(expectedReset) {
		t.Fatalf("weighted denial = (%+v, %v), want reset %s", result, err, expectedReset)
	}
}

func TestRedisEngineExpiredPendingFailsClosed(t *testing.T) {
	client, partition := integrationRedis(t)
	engine, testRedisEngineExpiredPendingFailsClosedErr := NewRedisEngine(client, RedisEngineOptions{})
	if testRedisEngineExpiredPendingFailsClosedErr != nil {
		t.Fatalf("NewRedisEngine() error = %v", testRedisEngineExpiredPendingFailsClosedErr)
	}
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	first := AdmissionRequest{
		Partition: partition, AdmissionID: "pending-a", Digest: "pending-a",
		LeaseDuration: 50 * time.Millisecond, Preconditions: preconditions,
	}
	if result, err := engine.Admit(context.Background(), first); err != nil || !result.Allowed() {
		t.Fatalf("first Admit() = (%+v, %v), want allow", result, err)
	}
	time.Sleep(75 * time.Millisecond)
	second := first
	second.AdmissionID = "pending-b"
	second.Digest = "pending-b"
	result, testRedisEngineExpiredPendingFailsClosedErr := engine.Admit(context.Background(), second)
	if testRedisEngineExpiredPendingFailsClosedErr != nil || result.Disposition != AdmissionUnavailable ||
		result.BlockingReason != "expired pending admission requires reconciliation" {
		t.Fatalf("expired-pending Admit() = (%+v, %v), want unavailable", result, testRedisEngineExpiredPendingFailsClosedErr)
	}
}

func TestRedisEngineCheckAccessIsReadOnly(t *testing.T) {
	client, partition := integrationRedis(t)
	engine, testRedisEngineCheckAccessIsReadOnlyErr := NewRedisEngine(client, RedisEngineOptions{})
	if testRedisEngineCheckAccessIsReadOnlyErr != nil {
		t.Fatalf("NewRedisEngine() error = %v", testRedisEngineCheckAccessIsReadOnlyErr)
	}
	preconditions, denyKey, credentialKey := seedAccessProjection(t, client, partition)
	request := AccessCheckRequest{Partition: partition, Preconditions: preconditions}

	result, testRedisEngineCheckAccessIsReadOnlyErr := engine.CheckAccess(context.Background(), request)
	if testRedisEngineCheckAccessIsReadOnlyErr != nil || !result.Allowed() || result.Reason != "" || result.ServerTime.IsZero() {
		t.Fatalf("allowed CheckAccess() = (%+v, %v)", result, testRedisEngineCheckAccessIsReadOnlyErr)
	}
	partitionKeys, _ := newPartitionKeys(partition)
	if exists, existsErr := client.Exists(context.Background(), partitionKeys.pendingIndex).Result(); existsErr != nil || exists != 0 {
		t.Fatalf("CheckAccess() pending index exists = (%d, %v), want zero", exists, existsErr)
	}

	if err := client.HSet(context.Background(), credentialKey, "status", "revoked").Err(); err != nil {
		t.Fatalf("revoke credential: %v", err)
	}
	result, testRedisEngineCheckAccessIsReadOnlyErr = engine.CheckAccess(context.Background(), request)
	if testRedisEngineCheckAccessIsReadOnlyErr != nil || result.Disposition != AdmissionUnauthenticated || result.Reason != "credential_inactive" {
		t.Fatalf("revoked CheckAccess() = (%+v, %v), want unauthenticated", result, testRedisEngineCheckAccessIsReadOnlyErr)
	}
	if err := client.HSet(context.Background(), credentialKey, "status", "active").Err(); err != nil {
		t.Fatalf("restore credential: %v", err)
	}
	if err := client.Set(context.Background(), denyKey, "1", 0).Err(); err != nil {
		t.Fatalf("seed deny: %v", err)
	}
	result, testRedisEngineCheckAccessIsReadOnlyErr = engine.CheckAccess(context.Background(), request)
	if testRedisEngineCheckAccessIsReadOnlyErr != nil || result.Disposition != AdmissionForbidden || result.Reason != "deny_barrier_active" {
		t.Fatalf("denied CheckAccess() = (%+v, %v), want forbidden", result, testRedisEngineCheckAccessIsReadOnlyErr)
	}
	if err := client.Del(context.Background(), denyKey).Err(); err != nil {
		t.Fatalf("clear deny: %v", err)
	}
	if err := client.HSet(context.Background(), credentialKey, "expires_at_ms", "invalid").Err(); err != nil {
		t.Fatalf("corrupt credential time: %v", err)
	}
	result, testRedisEngineCheckAccessIsReadOnlyErr = engine.CheckAccess(context.Background(), request)
	if testRedisEngineCheckAccessIsReadOnlyErr != nil || result.Disposition != AdmissionUnavailable ||
		result.Reason != "invalid access time projection" {
		t.Fatalf("corrupt CheckAccess() = (%+v, %v), want unavailable", result, testRedisEngineCheckAccessIsReadOnlyErr)
	}
}
