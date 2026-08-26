package quotaruntime

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
)

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
