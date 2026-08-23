package quotaruntime

import (
	"context"
	"errors"
	"fmt"
	"math/big"
	"os"
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
		Partition: partition, Rules: rules,
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
		Partition: partition, AdmissionID: first.AdmissionID, AdmissionDigest: first.Digest, Rules: rules,
	})
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || release.Idempotent {
		t.Fatalf("ReleaseConcurrency() = (%+v, %v), want new release", release, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	release, testRedisEngineAdmissionFinalizationUnknownAndMetersErr = engine.ReleaseConcurrency(context.Background(), ConcurrencyReleaseRequest{
		Partition: partition, AdmissionID: first.AdmissionID, AdmissionDigest: first.Digest, Rules: rules,
	})
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || !release.Idempotent {
		t.Fatalf("duplicate ReleaseConcurrency() = (%+v, %v), want idempotent", release, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	result, testRedisEngineAdmissionFinalizationUnknownAndMetersErr = engine.Admit(context.Background(), blocked)
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || !result.Allowed() {
		t.Fatalf("Admit() after release = (%+v, %v), want allow", result, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}

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

	tokenIdentity, _ := rules[0].Counter()
	costIdentity, _ := rules[1].Counter()
	finalization := FinalizationRequest{
		Partition: partition, AdmissionID: first.AdmissionID, AdmissionDigest: first.Digest,
		FinalizationDigest: "usage-a", DispatchCount: 1,
		Event: `{"admissionId":"admission-a"}`, Rules: rules,
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
	result, testRedisEngineAdmissionFinalizationUnknownAndMetersErr = engine.Admit(context.Background(), third)
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || result.Disposition != AdmissionRateLimited || result.Limiting == nil ||
		result.Limiting.RuleID != "tokens" || result.ResetAt == nil {
		t.Fatalf("post-settlement Admit() = (%+v, %v), want token denial", result, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}

	journal = DispatchJournalRequest{
		Partition: partition, AdmissionID: blocked.AdmissionID, AdmissionDigest: blocked.Digest,
		DispatchID: "dispatch-b", Ordinal: 0, Digest: "dispatch-digest-b",
	}
	if _, err := engine.JournalDispatch(context.Background(), journal); err != nil {
		t.Fatalf("JournalDispatch(B) error = %v", err)
	}
	unknown := FinalizationRequest{
		Partition: partition, AdmissionID: blocked.AdmissionID, AdmissionDigest: blocked.Digest,
		FinalizationDigest: "unknown-b", DispatchCount: 1,
		Event:   `{"admissionId":"admission-b"}`,
		FenceID: "fence-b", Rules: rules,
		Evidence: map[quota.CounterIdentity]ActualEvidence{
			tokenIdentity: {State: ActualEvidenceUnknown, Reason: "provider_usage_unavailable"},
			costIdentity:  {State: ActualEvidenceUnknown, Reason: "provider_usage_unavailable"},
		},
	}
	finalized, testRedisEngineAdmissionFinalizationUnknownAndMetersErr = engine.Finalize(context.Background(), unknown)
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
	result, testRedisEngineAdmissionFinalizationUnknownAndMetersErr = engine.Admit(context.Background(), fourth)
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil || result.Disposition != AdmissionUnavailable ||
		result.BlockingReason != "binding has unresolved usage" {
		t.Fatalf("fenced Admit() = (%+v, %v), want fence", result, testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	meters, testRedisEngineAdmissionFinalizationUnknownAndMetersErr = engine.ReadMeters(context.Background(), MeterReadRequest{Partition: partition, Rules: rules})
	if testRedisEngineAdmissionFinalizationUnknownAndMetersErr != nil {
		t.Fatalf("ReadMeters() after unknown error = %v", testRedisEngineAdmissionFinalizationUnknownAndMetersErr)
	}
	tokenMeter = meterByRule(t, meters, "tokens")
	if tokenMeter.Completeness != quota.CompletenessPartial ||
		tokenMeter.CapacityState != quota.CapacityFenced || tokenMeter.Remaining != nil ||
		len(tokenMeter.ActiveFenceIDs) != 1 || tokenMeter.ActiveFenceIDs[0] != "fence-b" {
		t.Fatalf("fenced token meter = %+v, want partial fenced capacity", tokenMeter)
	}

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
	meters, testRedisEngineAdmissionFinalizationUnknownAndMetersErr = engine.ReadMeters(context.Background(), MeterReadRequest{Partition: partition, Rules: rules})
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
	tokenMeter = meterByRule(t, meters, "tokens")
	if tokenMeter.Used != "15" || tokenMeter.IncompleteDispatches != "0" ||
		tokenMeter.KnownDispatches != "2" || len(tokenMeter.ActiveFenceIDs) != 0 {
		t.Fatalf("reconciled token meter = %+v", tokenMeter)
	}
	if costMeter = meterByRule(t, meters, "cost"); costMeter.Used != "3.5" ||
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
		Event: `{"admissionId":"eight-hour-cost-a"}`, Rules: rules,
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
			Event: `{"admissionId":"` + admissionID + `"}`,
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

func TestRedisEngineEnforcesConfiguredKeyPrefix(t *testing.T) {
	client, partition := integrationRedis(t)
	const prefix = "vllm-sr:access:test"
	engine, testRedisEngineEnforcesConfiguredKeyPrefixErr := NewRedisEngine(client, RedisEngineOptions{KeyPrefix: prefix})
	if testRedisEngineEnforcesConfiguredKeyPrefixErr != nil {
		t.Fatalf("NewRedisEngine() error = %v", testRedisEngineEnforcesConfiguredKeyPrefixErr)
	}
	keyspace, testRedisEngineEnforcesConfiguredKeyPrefixErr := NewAccessProjectionKeyspaceWithPrefix(prefix, partition)
	if testRedisEngineEnforcesConfiguredKeyPrefixErr != nil {
		t.Fatalf("NewAccessProjectionKeyspaceWithPrefix() error = %v", testRedisEngineEnforcesConfiguredKeyPrefixErr)
	}
	activeKey := keyspace.Active("key-1")
	if err := client.HSet(context.Background(), activeKey, "revision", "1").Err(); err != nil {
		t.Fatalf("seed prefixed projection: %v", err)
	}
	request := AccessCheckRequest{Partition: partition, Preconditions: []AdmissionPrecondition{{
		Key: activeKey, Kind: AdmissionCheckHashEqual, Field: "revision", Expected: "1",
		Failure: AdmissionUnavailable, Reason: "active_policy_changed",
	}}}
	if result, checkErr := engine.CheckAccess(context.Background(), request); checkErr != nil || !result.Allowed() {
		t.Fatalf("prefixed CheckAccess() = (%+v, %v), want allowed", result, checkErr)
	}
	unprefixed, _ := NewAccessProjectionKeyspace(partition)
	request.Preconditions[0].Key = unprefixed.Active("key-1")
	if _, checkErr := engine.CheckAccess(context.Background(), request); !errors.Is(checkErr, ErrInvalidRequest) {
		t.Fatalf("unprefixed CheckAccess() error = %v, want %v", checkErr, ErrInvalidRequest)
	}
	directory, testRedisEngineEnforcesConfiguredKeyPrefixErr := CredentialDirectoryKeyWithPrefix(prefix, "api-key", "kid-1")
	if testRedisEngineEnforcesConfiguredKeyPrefixErr != nil {
		t.Fatalf("CredentialDirectoryKeyWithPrefix() error = %v", testRedisEngineEnforcesConfiguredKeyPrefixErr)
	}
	request.Preconditions[0].Key = directory
	if _, checkErr := engine.CheckAccess(context.Background(), request); !errors.Is(checkErr, ErrInvalidRequest) {
		t.Fatalf("directory CheckAccess() error = %v, want %v", checkErr, ErrInvalidRequest)
	}
}

func TestRedisEngineRequestAlgorithms(t *testing.T) {
	tests := []struct {
		name      string
		rule      func(*testing.T) RuleBinding
		wait      time.Duration
		wantReset bool
	}{
		{
			name: "calendar window",
			rule: func(t *testing.T) RuleBinding {
				return calendarRequestRule(t, "binding", "calendar", "1", calendarScheduleAroundNow(), 0)
			},
			wantReset: true,
		},
		{
			name: "token bucket",
			rule: func(t *testing.T) RuleBinding {
				return tokenBucketRule(t, "binding", "bucket", "1", "1", 200*time.Millisecond, 0)
			},
			wait: 250 * time.Millisecond, wantReset: true,
		},
		{
			name: "GCRA",
			rule: func(t *testing.T) RuleBinding {
				return gcraRule(t, "binding", "gcra", 200*time.Millisecond, "0", 0)
			},
			wait: 250 * time.Millisecond, wantReset: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client, partition := integrationRedis(t)
			engine, err := NewRedisEngine(client, RedisEngineOptions{})
			if err != nil {
				t.Fatalf("NewRedisEngine() error = %v", err)
			}
			preconditions, _, _ := seedAccessProjection(t, client, partition)
			rules := []RuleBinding{test.rule(t)}
			first := AdmissionRequest{
				Partition: partition, AdmissionID: "algorithm-a", Digest: "algorithm-a",
				LeaseDuration: 5 * time.Second, Preconditions: preconditions, Rules: rules,
			}
			if result, admitErr := engine.Admit(context.Background(), first); admitErr != nil || !result.Allowed() {
				t.Fatalf("first Admit() = (%+v, %v), want allow", result, admitErr)
			}
			second := first
			second.AdmissionID = "algorithm-b"
			second.Digest = "algorithm-b"
			result, admitErr := engine.Admit(context.Background(), second)
			if admitErr != nil || result.Disposition != AdmissionRateLimited ||
				(test.wantReset && result.ResetAt == nil) {
				t.Fatalf("second Admit() = (%+v, %v), want rate limit", result, admitErr)
			}
			meters, meterErr := engine.ReadMeters(context.Background(), MeterReadRequest{
				Partition: partition, Rules: rules,
			})
			if meterErr != nil {
				t.Fatalf("ReadMeters() error = %v", meterErr)
			}
			if meter := meterByRule(t, meters, rules[0].Rule.ID); meter.Used != "1" {
				t.Fatalf("meter = %+v, want used 1", meter)
			}
			if test.wait > 0 {
				time.Sleep(test.wait)
				if result, admitErr = engine.Admit(context.Background(), second); admitErr != nil || !result.Allowed() {
					t.Fatalf("refilled Admit() = (%+v, %v), want allow", result, admitErr)
				}
			}
		})
	}
}

func TestRedisEngineCalendarScheduleGapFailsClosed(t *testing.T) {
	client, partition := integrationRedis(t)
	engine, err := NewRedisEngine(client, RedisEngineOptions{})
	if err != nil {
		t.Fatalf("NewRedisEngine() error = %v", err)
	}
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	now := time.Now().UTC().Truncate(time.Millisecond)
	rule := calendarRequestRule(t, "binding", "future", "1", []CalendarInterval{{
		Start: now.Add(time.Hour), End: now.Add(2 * time.Hour),
	}}, 0)
	result, err := engine.Admit(context.Background(), AdmissionRequest{
		Partition: partition, AdmissionID: "calendar-gap", Digest: "calendar-gap",
		LeaseDuration: time.Minute, Preconditions: preconditions, Rules: []RuleBinding{rule},
	})
	if err != nil || result.Disposition != AdmissionUnavailable ||
		result.BlockingReason != "calendar schedule unavailable" {
		t.Fatalf("Admit() = (%+v, %v), want typed unavailable", result, err)
	}
}

func TestRedisEngineCalendarActualFinalizationCrossing(t *testing.T) {
	client, partition := integrationRedis(t)
	engine, testRedisEngineCalendarActualFinalizationCrossingErr := NewRedisEngine(client, RedisEngineOptions{})
	if testRedisEngineCalendarActualFinalizationCrossingErr != nil {
		t.Fatalf("NewRedisEngine() error = %v", testRedisEngineCalendarActualFinalizationCrossingErr)
	}
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	rules := []RuleBinding{
		calendarTokenRule(t, "binding", "daily-tokens", "10", calendarScheduleAroundNow(), 0),
	}
	admission := AdmissionRequest{
		Partition: partition, AdmissionID: "calendar-actual-a", Digest: "calendar-actual-a",
		LeaseDuration: time.Minute, Preconditions: preconditions, Rules: rules,
	}
	if result, admitErr := engine.Admit(context.Background(), admission); admitErr != nil || !result.Allowed() {
		t.Fatalf("Admit() = (%+v, %v), want allow", result, admitErr)
	}
	if _, err := engine.JournalDispatch(context.Background(), DispatchJournalRequest{
		Partition: partition, AdmissionID: admission.AdmissionID, AdmissionDigest: admission.Digest,
		DispatchID: "dispatch", Digest: "dispatch", Ordinal: 0,
	}); err != nil {
		t.Fatalf("JournalDispatch() error = %v", err)
	}
	identity, _ := rules[0].Counter()
	if _, err := engine.Finalize(context.Background(), FinalizationRequest{
		Partition: partition, AdmissionID: admission.AdmissionID, AdmissionDigest: admission.Digest,
		FinalizationDigest: "usage", DispatchCount: 1,
		Event: `{"admissionId":"calendar-actual-a"}`,
		Rules: rules,
		Evidence: map[quota.CounterIdentity]ActualEvidence{
			identity: {State: ActualEvidenceKnown, Amount: quotaInteger(t, "12")},
		},
	}); err != nil {
		t.Fatalf("Finalize() error = %v", err)
	}
	next := admission
	next.AdmissionID = "calendar-actual-b"
	next.Digest = "calendar-actual-b"
	result, testRedisEngineCalendarActualFinalizationCrossingErr := engine.Admit(context.Background(), next)
	if testRedisEngineCalendarActualFinalizationCrossingErr != nil || result.Disposition != AdmissionRateLimited || result.ResetAt == nil {
		t.Fatalf("post-crossing Admit() = (%+v, %v), want rate limit", result, testRedisEngineCalendarActualFinalizationCrossingErr)
	}
	meters, testRedisEngineCalendarActualFinalizationCrossingErr := engine.ReadMeters(context.Background(), MeterReadRequest{Partition: partition, Rules: rules})
	if testRedisEngineCalendarActualFinalizationCrossingErr != nil {
		t.Fatalf("ReadMeters() error = %v", testRedisEngineCalendarActualFinalizationCrossingErr)
	}
	if meter := meterByRule(t, meters, "daily-tokens"); meter.Used != "12" ||
		meter.CapacityState != quota.CapacityOverLimit {
		t.Fatalf("calendar meter = %+v, want crossing", meter)
	}
}

func TestRedisEngineMixedFinalizationDebitsAndFencesPrecisely(t *testing.T) {
	client, partition := integrationRedis(t)
	engine, testRedisEngineMixedFinalizationDebitsAndFencesPreciselyErr := NewRedisEngine(client, RedisEngineOptions{})
	if testRedisEngineMixedFinalizationDebitsAndFencesPreciselyErr != nil {
		t.Fatalf("NewRedisEngine() error = %v", testRedisEngineMixedFinalizationDebitsAndFencesPreciselyErr)
	}
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	backendTokens := tokenRule(t, "binding-backend", "backend-tokens", "100", time.Minute, 0)
	servedTokens := tokenRule(t, "binding-served", "served-tokens", "100", time.Minute, 1)
	servedTokens.Rule.Metric = quota.MetricServedTotalTokens
	shadowCost := costRule(t, "binding-cost", "cost", "5", time.Minute, 2)
	shadowCost.Rule.Enforcement = quota.EnforcementShadow
	concurrency := concurrencyRule(t, "binding-concurrency", "concurrency", "1", 3)
	rules := []RuleBinding{backendTokens, servedTokens, shadowCost, concurrency}
	admission := AdmissionRequest{
		Partition: partition, AdmissionID: "mixed-a", Digest: "mixed-request",
		LeaseDuration: time.Minute, Preconditions: preconditions, Rules: rules,
	}
	if result, admitErr := engine.Admit(context.Background(), admission); admitErr != nil || !result.Allowed() {
		t.Fatalf("Admit() = (%+v, %v), want allow", result, admitErr)
	}
	if _, err := engine.JournalDispatch(context.Background(), DispatchJournalRequest{
		Partition: partition, AdmissionID: admission.AdmissionID, AdmissionDigest: admission.Digest,
		DispatchID: "dispatch", Digest: "dispatch", Ordinal: 0,
	}); err != nil {
		t.Fatalf("JournalDispatch() error = %v", err)
	}
	backendIdentity, _ := backendTokens.Counter()
	servedIdentity, _ := servedTokens.Counter()
	costIdentity, _ := shadowCost.Counter()
	request := FinalizationRequest{
		Partition: partition, AdmissionID: admission.AdmissionID, AdmissionDigest: admission.Digest,
		FinalizationDigest: "finalization-1", DispatchCount: 1,
		Event:   `{"admissionId":"mixed-a"}`,
		FenceID: "mixed-fence", Rules: rules,
		Evidence: map[quota.CounterIdentity]ActualEvidence{
			backendIdentity: {State: ActualEvidenceUnknown, Reason: "backend_usage_missing"},
			servedIdentity:  {State: ActualEvidenceKnown, Amount: quotaInteger(t, "7")},
			costIdentity:    {State: ActualEvidenceUnknown, Reason: "cache_price_missing"},
		},
	}
	result, testRedisEngineMixedFinalizationDebitsAndFencesPreciselyErr := engine.Finalize(context.Background(), request)
	if testRedisEngineMixedFinalizationDebitsAndFencesPreciselyErr != nil || result.Idempotent || result.EvidenceState != "mixed" || result.StreamID == "" {
		t.Fatalf("Finalize() = (%+v, %v), want new mixed finalization", result, testRedisEngineMixedFinalizationDebitsAndFencesPreciselyErr)
	}
	duplicate, testRedisEngineMixedFinalizationDebitsAndFencesPreciselyErr := engine.Finalize(context.Background(), request)
	if testRedisEngineMixedFinalizationDebitsAndFencesPreciselyErr != nil || !duplicate.Idempotent || duplicate.StreamID != result.StreamID {
		t.Fatalf("duplicate Finalize() = (%+v, %v), want idempotent", duplicate, testRedisEngineMixedFinalizationDebitsAndFencesPreciselyErr)
	}
	conflict := request
	conflict.FinalizationDigest = "finalization-2"
	if _, err := engine.Finalize(context.Background(), conflict); !errors.Is(err, ErrConflict) {
		t.Fatalf("conflicting Finalize() error = %v, want %v", err, ErrConflict)
	}

	meters, testRedisEngineMixedFinalizationDebitsAndFencesPreciselyErr := engine.ReadMeters(context.Background(), MeterReadRequest{Partition: partition, Rules: rules})
	if testRedisEngineMixedFinalizationDebitsAndFencesPreciselyErr != nil {
		t.Fatalf("ReadMeters() error = %v", testRedisEngineMixedFinalizationDebitsAndFencesPreciselyErr)
	}
	backendMeter := meterByRule(t, meters, "backend-tokens")
	if backendMeter.IncompleteDispatches != "1" ||
		backendMeter.CapacityState != quota.CapacityFenced || len(backendMeter.ActiveFenceIDs) != 1 ||
		backendMeter.ActiveFenceIDs[0] != "mixed-fence" {
		t.Fatalf("backend meter = %+v, want fenced unknown", backendMeter)
	}
	servedMeter := meterByRule(t, meters, "served-tokens")
	if servedMeter.Used != "7" || servedMeter.IncompleteDispatches != "0" ||
		servedMeter.CapacityState != quota.CapacityAvailable {
		t.Fatalf("served meter = %+v, want exact known debit", servedMeter)
	}
	costMeter := meterByRule(t, meters, "cost")
	if costMeter.IncompleteDispatches != "1" ||
		costMeter.CapacityState != quota.CapacityUnknown {
		t.Fatalf("cost meter = %+v, want shadow unknown without fence", costMeter)
	}

	compiled, testRedisEngineMixedFinalizationDebitsAndFencesPreciselyErr := compileRules(partition, rules)
	if testRedisEngineMixedFinalizationDebitsAndFencesPreciselyErr != nil {
		t.Fatalf("compileRules() error = %v", testRedisEngineMixedFinalizationDebitsAndFencesPreciselyErr)
	}
	for _, rule := range compiled {
		want := int64(0)
		if rule.identity == backendIdentity {
			want = 1
		}
		if members, memberErr := client.SCard(context.Background(), rule.keys.fences).Result(); memberErr != nil || members != want {
			t.Fatalf("fence set for %s = (%d, %v), want %d", rule.identity, members, memberErr, want)
		}
	}
	partitionKeys, _ := newPartitionKeys(partition)
	if length, streamErr := client.XLen(context.Background(), partitionKeys.usageStream).Result(); streamErr != nil || length != 1 {
		t.Fatalf("usage stream length = (%d, %v), want 1", length, streamErr)
	}
	concurrencyIdentity, _ := concurrency.Counter()
	for _, rule := range compiled {
		if rule.identity == concurrencyIdentity {
			if count, countErr := client.ZCard(context.Background(), rule.keys.events).Result(); countErr != nil || count != 0 {
				t.Fatalf("concurrency count = (%d, %v), want released", count, countErr)
			}
		}
	}
}

func TestRedisLuaExactMultiplication(t *testing.T) {
	client, _ := integrationRedis(t)
	script := redis.NewScript(exactLua + `
local product, multiply_error = quota_multiply(ARGV[1], ARGV[2])
return {product or "", multiply_error or ""}
`)
	left := "999999999999999999999999999999"
	right := "999999999999"
	want := new(big.Int)
	want.Mul(mustBigInteger(t, left), mustBigInteger(t, right))
	value, err := script.Run(context.Background(), client, nil, left, right).Result()
	if err != nil {
		t.Fatalf("exact multiply script error = %v", err)
	}
	fields, err := scriptStrings(value, 2)
	if err != nil || fields[0] != want.String() || fields[1] != "" {
		t.Fatalf("exact multiply = (%v, %v), want %s", fields, err, want)
	}
	value, err = script.Run(
		context.Background(),
		client,
		nil,
		"999999999999999999999999999999999999999999",
		"2",
	).Result()
	if err != nil {
		t.Fatalf("overflow multiply script error = %v", err)
	}
	fields, err = scriptStrings(value, 2)
	if err != nil || fields[0] != "" || fields[1] != "quantity overflow" {
		t.Fatalf("overflow multiply = (%v, %v), want exact overflow", fields, err)
	}
}

func mustBigInteger(t *testing.T, value string) *big.Int {
	t.Helper()
	result, ok := new(big.Int).SetString(value, 10)
	if !ok {
		t.Fatalf("invalid test integer %q", value)
	}
	return result
}

func seedAccessProjection(
	t *testing.T,
	client *redis.Client,
	partition string,
) ([]AdmissionPrecondition, string, string) {
	t.Helper()
	keyspace, err := NewAccessProjectionKeyspace(partition)
	if err != nil {
		t.Fatalf("NewAccessProjectionKeyspace() error = %v", err)
	}
	credentialKey := keyspace.Credential("api-key", "kid-1")
	now := time.Now()
	if err := client.HSet(context.Background(), credentialKey,
		"public_id", "kid-1",
		"key_id", "key-1",
		"hmac_revision", "9",
		"status", "active",
		"not_before_ms", now.Add(-time.Minute).UnixMilli(),
		"expires_at_ms", now.Add(time.Hour).UnixMilli(),
	).Err(); err != nil {
		t.Fatalf("seed credential projection: %v", err)
	}
	activeKey := keyspace.Active("key-1")
	if err := client.HSet(context.Background(), activeKey,
		"revision", "1",
		"policy_digest", "policy-digest-1",
	).Err(); err != nil {
		t.Fatalf("seed active projection: %v", err)
	}
	policyKey := keyspace.Policy("key-1", "1")
	if err := client.HSet(context.Background(), policyKey,
		"revision", "1",
		"key_status", "active",
		"user_status", "active",
		"team_status", "active",
		"membership_status", "active",
		"grant:entrypoint:ep-1:invoke", "allow",
	).Err(); err != nil {
		t.Fatalf("seed policy projection: %v", err)
	}
	denyKey := keyspace.Deny("key", "key-1")
	return []AdmissionPrecondition{
		{
			Key: credentialKey, Kind: AdmissionCheckHashEqual, Field: "public_id", Expected: "kid-1",
			Failure: AdmissionUnauthenticated, Reason: "credential_binding_changed",
		},
		{
			Key: credentialKey, Kind: AdmissionCheckHashEqual, Field: "key_id", Expected: "key-1",
			Failure: AdmissionUnauthenticated, Reason: "credential_binding_changed",
		},
		{
			Key: credentialKey, Kind: AdmissionCheckHashEqual, Field: "hmac_revision", Expected: "9",
			Failure: AdmissionUnauthenticated, Reason: "credential_revision_changed",
		},
		{
			Key: credentialKey, Kind: AdmissionCheckHashEqual, Field: "status", Expected: "active",
			Failure: AdmissionUnauthenticated, Reason: "credential_inactive",
		},
		{
			Key: credentialKey, Kind: AdmissionCheckNotBefore, Field: "not_before_ms",
			Failure: AdmissionUnauthenticated, Reason: "credential_not_active",
		},
		{
			Key: credentialKey, Kind: AdmissionCheckExpiresAfter, Field: "expires_at_ms",
			Failure: AdmissionUnauthenticated, Reason: "credential_expired",
		},
		{
			Key: activeKey, Kind: AdmissionCheckHashEqual, Field: "revision", Expected: "1",
			Failure: AdmissionUnavailable, Reason: "active_policy_changed",
		},
		{
			Key: activeKey, Kind: AdmissionCheckHashEqual, Field: "policy_digest", Expected: "policy-digest-1",
			Failure: AdmissionUnavailable, Reason: "active_policy_changed",
		},
		{
			Key: policyKey, Kind: AdmissionCheckHashEqual, Field: "key_status", Expected: "active",
			Failure: AdmissionUnauthenticated, Reason: "key_inactive",
		},
		{
			Key: policyKey, Kind: AdmissionCheckHashEqual, Field: "user_status", Expected: "active",
			Failure: AdmissionUnauthenticated, Reason: "user_inactive",
		},
		{
			Key: policyKey, Kind: AdmissionCheckHashEqual, Field: "team_status", Expected: "active",
			Failure: AdmissionForbidden, Reason: "team_inactive",
		},
		{
			Key: policyKey, Kind: AdmissionCheckHashEqual, Field: "membership_status", Expected: "active",
			Failure: AdmissionForbidden, Reason: "membership_inactive",
		},
		{
			Key: policyKey, Kind: AdmissionCheckHashEqual,
			Field: "grant:entrypoint:ep-1:invoke", Expected: "allow",
			Failure: AdmissionForbidden, Reason: "entrypoint_not_granted",
		},
		{
			Key: denyKey, Kind: AdmissionCheckKeyAbsent,
			Failure: AdmissionForbidden, Reason: "deny_barrier_active",
		},
	}, denyKey, credentialKey
}

func integrationRedis(t *testing.T) (*redis.Client, string) {
	t.Helper()
	address := os.Getenv("QUOTARUNTIME_REDIS_ADDR")
	if address == "" {
		t.Skip("QUOTARUNTIME_REDIS_ADDR is not set")
	}
	client := redis.NewClient(&redis.Options{Addr: address})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := client.Ping(ctx).Err(); err != nil {
		client.Close()
		t.Fatalf("Redis PING: %v", err)
	}
	partition := fmt.Sprintf("quota-it-%d", time.Now().UnixNano())
	t.Cleanup(func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cleanupCancel()
		var cursor uint64
		for {
			keys, next, err := client.Scan(cleanupCtx, cursor, "*{"+partition+"}*", 100).Result()
			if err != nil {
				t.Errorf("scan integration keys: %v", err)
				break
			}
			if len(keys) > 0 {
				if err := client.Del(cleanupCtx, keys...).Err(); err != nil {
					t.Errorf("delete integration keys: %v", err)
				}
			}
			cursor = next
			if cursor == 0 {
				break
			}
		}
		if err := client.Close(); err != nil {
			t.Errorf("close Redis client: %v", err)
		}
	})
	return client, partition
}

func meterByRule(t *testing.T, result MeterReadResult, ruleID string) Meter {
	t.Helper()
	for _, meter := range result.Meters {
		if meter.RuleID == ruleID {
			return meter
		}
	}
	t.Fatalf("meter for rule %q not found", ruleID)
	return Meter{}
}
