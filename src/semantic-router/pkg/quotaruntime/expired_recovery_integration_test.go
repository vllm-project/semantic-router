package quotaruntime

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

func TestRedisExpiredAdmissionWithoutDispatchRecoversKnownZeroExactlyOnce(t *testing.T) {
	client, partition := integrationRedis(t)
	first, second := recoveryEngines(t, client)
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	rules := []RuleBinding{
		tokenRule(t, uuid.NewString(), uuid.NewString(), "100", time.Minute, 0),
		concurrencyRule(t, uuid.NewString(), uuid.NewString(), "1", 1),
	}
	request := recoveryAdmission(t, partition, preconditions, rules, 60*time.Millisecond)
	admission, err := first.Admit(context.Background(), request)
	if err != nil || !admission.Allowed() {
		t.Fatalf("Admit() = (%+v, %v), want allowed", admission, err)
	}
	time.Sleep(90 * time.Millisecond)

	results, recoveryErrors := raceExpiredRecovery(first, second, partition)
	for _, recoverErr := range recoveryErrors {
		if recoverErr != nil {
			t.Fatalf("RecoverOldestExpiredAdmission() error = %v", recoverErr)
		}
	}
	recovered := 0
	for _, result := range results {
		if result.Recovered {
			recovered++
		}
	}
	if recovered == 0 {
		t.Fatalf("recovery results = %+v, want at least one successful observer", results)
	}
	keys, _ := newPartitionKeys(partition)
	if length, err := client.XLen(context.Background(), keys.usageStream).Result(); err != nil || length != 1 {
		t.Fatalf("usage stream length = (%d, %v), want exactly one", length, err)
	}
	event := readOnlyUsageEvent(t, client, keys.usageStream)
	if event.AdmissionID != request.AdmissionID || event.EvidenceState != usageledger.EvidenceKnown ||
		len(event.Dispatches) != 1 || event.Dispatches[0].DispatchType != "not_dispatched" ||
		event.Dispatches[0].UsageState != usageledger.UsageKnownZero || event.Fence != nil {
		t.Fatalf("recovered event = %+v", event)
	}
	meters, err := first.ReadMeters(context.Background(), MeterReadRequest{Partition: partition, Rules: rules})
	if err != nil {
		t.Fatalf("ReadMeters() error = %v", err)
	}
	if token := meterByRule(t, meters, rules[0].Rule.ID); token.Used != "0" ||
		token.IncompleteDispatches != "0" || token.CapacityState == quota.CapacityFenced {
		t.Fatalf("known-zero recovered meter = %+v", token)
	}
}

func TestRedisExpiredAdmissionWithStartedAttemptRecoversUnknownAndFences(t *testing.T) {
	client, partition := integrationRedis(t)
	first, second := recoveryEngines(t, client)
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	rules := []RuleBinding{tokenRule(t, uuid.NewString(), uuid.NewString(), "100", time.Minute, 0)}
	request := recoveryAdmission(t, partition, preconditions, rules, 100*time.Millisecond)
	admission, err := first.Admit(context.Background(), request)
	if err != nil || !admission.Allowed() {
		t.Fatalf("Admit() = (%+v, %v), want allowed", admission, err)
	}
	dispatchID := "dispatch-crash"
	planDigest := strings.Repeat("a", 64)
	if _, err := first.JournalDispatch(context.Background(), DispatchJournalRequest{
		Partition: partition, AdmissionID: request.AdmissionID, AdmissionDigest: request.Digest,
		DispatchID: dispatchID, Ordinal: 0, Digest: planDigest,
	}); err != nil {
		t.Fatalf("JournalDispatch() error = %v", err)
	}
	reference := DispatchReference{
		Partition: partition, AdmissionID: request.AdmissionID, AdmissionDigest: request.Digest,
		DispatchID: dispatchID, DispatchPlanDigest: planDigest,
		ModelID:       request.Recovery.FallbackDispatch.ModelID,
		ModelRevision: request.Recovery.FallbackDispatch.ModelRevision,
		RequestDigest: testBackendRequestDigest("expired"),
	}
	if _, err := first.BeginDispatch(context.Background(), BeginDispatchRequest{
		DispatchReference: reference, DispatchType: "primary", Ordinal: 0,
		Deadline: admission.Deadline, MaxAttempts: 2,
	}); err != nil {
		t.Fatalf("BeginDispatch() error = %v", err)
	}
	if _, err := first.BeginAttempt(context.Background(), BeginAttemptRequest{
		DispatchReference: reference, AttemptID: "attempt-crash", AttemptNumber: 1,
		BackendID: uuid.NewString(), ProviderID: "vllm",
	}); err != nil {
		t.Fatalf("BeginAttempt() error = %v", err)
	}
	time.Sleep(130 * time.Millisecond)

	result, err := second.RecoverOldestExpiredAdmission(context.Background(), partition)
	if err != nil || !result.Recovered || result.EvidenceState != "unknown" {
		t.Fatalf("RecoverOldestExpiredAdmission() = (%+v, %v), want unknown recovery", result, err)
	}
	meters, err := second.ReadMeters(context.Background(), MeterReadRequest{Partition: partition, Rules: rules})
	if err != nil {
		t.Fatalf("ReadMeters() error = %v", err)
	}
	if meter := meterByRule(t, meters, rules[0].Rule.ID); meter.CapacityState != quota.CapacityFenced ||
		meter.IncompleteDispatches != "1" || len(meter.ActiveFenceIDs) != 1 ||
		meter.ActiveFenceIDs[0] != request.Recovery.FenceID {
		t.Fatalf("unknown recovered meter = %+v", meter)
	}
	keys, _ := newPartitionKeys(partition)
	event := readOnlyUsageEvent(t, client, keys.usageStream)
	if event.EvidenceState != usageledger.EvidenceUnknown || event.Fence == nil ||
		len(event.Dispatches) != 1 || event.Dispatches[0].UsageState != usageledger.UsageUnknown ||
		len(event.Dispatches[0].Attempts) != 1 ||
		event.Dispatches[0].Attempts[0].State != usageledger.UsageUnknown {
		t.Fatalf("unknown recovered event = %+v", event)
	}

	blocked := recoveryAdmission(t, partition, preconditions, rules, time.Minute)
	blockedResult, err := first.Admit(context.Background(), blocked)
	if err != nil || blockedResult.Disposition != AdmissionUnavailable ||
		blockedResult.BlockingReason != "binding has unresolved usage" {
		t.Fatalf("post-recovery Admit() = (%+v, %v), want unresolved-usage fence", blockedResult, err)
	}
}

func TestRedisExpiredStartedAttemptWithoutActualRulesPreservesUnknownEventEvidence(t *testing.T) {
	client, partition := integrationRedis(t)
	first, second := recoveryEngines(t, client)
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	rules := []RuleBinding{
		requestRule(t, uuid.NewString(), uuid.NewString(), "100", time.Minute, 0),
		concurrencyRule(t, uuid.NewString(), uuid.NewString(), "1", 1),
	}
	request := recoveryAdmission(t, partition, preconditions, rules, 100*time.Millisecond)
	admission, err := first.Admit(context.Background(), request)
	if err != nil || !admission.Allowed() {
		t.Fatalf("Admit() = (%+v, %v), want allowed", admission, err)
	}
	dispatchID := "dispatch-no-actual-rule"
	planDigest := strings.Repeat("d", 64)
	if _, err := first.JournalDispatch(context.Background(), DispatchJournalRequest{
		Partition: partition, AdmissionID: request.AdmissionID, AdmissionDigest: request.Digest,
		DispatchID: dispatchID, Ordinal: 0, Digest: planDigest,
	}); err != nil {
		t.Fatalf("JournalDispatch() error = %v", err)
	}
	reference := DispatchReference{
		Partition: partition, AdmissionID: request.AdmissionID, AdmissionDigest: request.Digest,
		DispatchID: dispatchID, DispatchPlanDigest: planDigest,
		ModelID:       request.Recovery.FallbackDispatch.ModelID,
		ModelRevision: request.Recovery.FallbackDispatch.ModelRevision,
		RequestDigest: testBackendRequestDigest("expired-no-actual-rule"),
	}
	if _, err := first.BeginDispatch(context.Background(), BeginDispatchRequest{
		DispatchReference: reference, DispatchType: "primary", Ordinal: 0,
		Deadline: admission.Deadline, MaxAttempts: 1,
	}); err != nil {
		t.Fatalf("BeginDispatch() error = %v", err)
	}
	if _, err := first.BeginAttempt(context.Background(), BeginAttemptRequest{
		DispatchReference: reference, AttemptID: "attempt-no-actual-rule", AttemptNumber: 1,
		BackendID: uuid.NewString(), ProviderID: "vllm",
	}); err != nil {
		t.Fatalf("BeginAttempt() error = %v", err)
	}
	time.Sleep(130 * time.Millisecond)

	result, err := second.RecoverOldestExpiredAdmission(context.Background(), partition)
	if err != nil || !result.Recovered || result.EvidenceState != "unknown" {
		t.Fatalf("RecoverOldestExpiredAdmission() = (%+v, %v), want unknown event recovery", result, err)
	}
	keys, _ := newPartitionKeys(partition)
	event, envelopeState := readOnlyUsageEnvelope(t, client, keys.usageStream)
	if envelopeState != string(usageledger.EvidenceUnknown) ||
		event.EvidenceState != usageledger.EvidenceUnknown || event.Fence != nil ||
		len(event.Dispatches) != 1 || event.Dispatches[0].UsageState != usageledger.UsageUnknown {
		t.Fatalf("recovered event/envelope = (%+v, %q), want unfenced unknown", event, envelopeState)
	}
	if markerState, err := client.HGet(context.Background(), keys.terminal(request.AdmissionID), "evidence_state").Result(); err != nil || markerState != string(usageledger.EvidenceUnknown) {
		t.Fatalf("terminal evidence state = (%q, %v), want unknown", markerState, err)
	}
	meters, err := second.ReadMeters(context.Background(), MeterReadRequest{Partition: partition, Rules: rules})
	if err != nil {
		t.Fatalf("ReadMeters() error = %v", err)
	}
	for _, meter := range meters.Meters {
		if meter.IncompleteDispatches != "0" || meter.CapacityState == quota.CapacityFenced {
			t.Fatalf("non-actual meter retained false unresolved usage = %+v", meter)
		}
	}
}

func recoveryEngines(t *testing.T, client *redis.Client) (*RedisEngine, *RedisEngine) {
	t.Helper()
	first, err := NewRedisEngine(client, RedisEngineOptions{})
	if err != nil {
		t.Fatalf("NewRedisEngine(first) error = %v", err)
	}
	second, err := NewRedisEngine(client, RedisEngineOptions{})
	if err != nil {
		t.Fatalf("NewRedisEngine(second) error = %v", err)
	}
	return first, second
}

func recoveryAdmission(
	t *testing.T,
	partition string,
	preconditions []AdmissionPrecondition,
	rules []RuleBinding,
	lease time.Duration,
) AdmissionRequest {
	t.Helper()
	namespaceID := uuid.NewString()
	return AdmissionRequest{
		Partition: partition, AdmissionID: uuid.NewString(), Digest: strings.Repeat("b", 64),
		LeaseDuration: lease, Preconditions: preconditions, Rules: rules,
		Recovery: &AdmissionRecoveryContext{
			EventID: uuid.NewString(), FenceID: uuid.NewString(), NamespaceID: namespaceID,
			Protocol: "openai_chat_completions", Path: "/v1/chat/completions",
			OccurredAt: time.Now().UTC().Truncate(time.Millisecond), Stream: true,
			Principal: RecoveryPrincipal{APIKeyID: uuid.NewString(), UserID: uuid.NewString(), TeamID: uuid.NewString()},
			Routing:   RecoveryRouting{AccessRevision: 1},
			FallbackDispatch: RecoveryDispatch{
				ModelID: uuid.NewString(), ModelName: "local/recovery-model", ModelRevision: 1, Currency: "USD",
			},
		},
	}
}

func raceExpiredRecovery(
	first, second *RedisEngine,
	partition string,
) ([2]ExpiredRecoveryResult, [2]error) {
	var results [2]ExpiredRecoveryResult
	var errs [2]error
	var wait sync.WaitGroup
	start := make(chan struct{})
	for index, engine := range []*RedisEngine{first, second} {
		wait.Add(1)
		go func(index int, engine *RedisEngine) {
			defer wait.Done()
			<-start
			results[index], errs[index] = engine.RecoverOldestExpiredAdmission(context.Background(), partition)
		}(index, engine)
	}
	close(start)
	wait.Wait()
	return results, errs
}

func readOnlyUsageEvent(t *testing.T, client *redis.Client, stream string) usageledger.TerminalEvent {
	t.Helper()
	event, _ := readOnlyUsageEnvelope(t, client, stream)
	return event
}

func readOnlyUsageEnvelope(
	t *testing.T,
	client *redis.Client,
	stream string,
) (usageledger.TerminalEvent, string) {
	t.Helper()
	entries, err := client.XRange(context.Background(), stream, "-", "+").Result()
	if err != nil || len(entries) != 1 {
		t.Fatalf("XRange() = (%+v, %v), want one event", entries, err)
	}
	payload, ok := entries[0].Values["event"].(string)
	if !ok {
		t.Fatalf("usage event payload type = %T", entries[0].Values["event"])
	}
	event, err := usageledger.DecodeTerminalEvent(payload)
	if err != nil {
		t.Fatalf("DecodeTerminalEvent() error = %v", err)
	}
	evidenceState, ok := entries[0].Values["evidence_state"].(string)
	if !ok {
		t.Fatalf("usage envelope evidence state type = %T", entries[0].Values["evidence_state"])
	}
	return event, evidenceState
}

func TestExpiredRecoveryRejectsMissingSnapshotWithoutReleasingAdmission(t *testing.T) {
	client, partition := integrationRedis(t)
	engine, _ := recoveryEngines(t, client)
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	request := AdmissionRequest{
		Partition: partition, AdmissionID: uuid.NewString(), Digest: strings.Repeat("c", 64),
		LeaseDuration: 40 * time.Millisecond, Preconditions: preconditions,
	}
	if result, err := engine.Admit(context.Background(), request); err != nil || !result.Allowed() {
		t.Fatalf("Admit() = (%+v, %v)", result, err)
	}
	time.Sleep(70 * time.Millisecond)
	if _, err := engine.RecoverOldestExpiredAdmission(context.Background(), partition); !errors.Is(err, ErrRuntimeCorrupt) {
		t.Fatalf("RecoverOldestExpiredAdmission() error = %v, want fail-closed corrupt snapshot", err)
	}
	keys, _ := newPartitionKeys(partition)
	if exists, err := client.Exists(context.Background(), keys.pending(request.AdmissionID)).Result(); err != nil || exists != 1 {
		t.Fatalf("expired admission exists = (%d, %v), want retained", exists, err)
	}
}
