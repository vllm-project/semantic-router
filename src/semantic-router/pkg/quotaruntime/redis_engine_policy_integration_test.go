package quotaruntime

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

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
	engine, err := NewRedisEngine(client, RedisEngineOptions{MaxUsageBacklog: 1})
	if err != nil {
		t.Fatalf("NewRedisEngine() error = %v", err)
	}
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	request := AdmissionRequest{
		Partition: partition, AdmissionID: "before-backlog", Digest: "before-backlog",
		LeaseDuration: time.Minute, Preconditions: preconditions,
	}
	first, err := engine.Admit(context.Background(), request)
	if err != nil || !first.Allowed() || first.Idempotent {
		t.Fatalf("first Admit() = (%+v, %v), want new allow", first, err)
	}

	keys, _ := newPartitionKeys(partition)
	if err := client.XAdd(context.Background(), &redis.XAddArgs{
		Stream: keys.usageStream, Values: map[string]any{"event": "settled"},
	}).Err(); err != nil {
		t.Fatalf("seed usage backlog: %v", err)
	}

	duplicate, err := engine.Admit(context.Background(), request)
	if err != nil || !duplicate.Allowed() || !duplicate.Idempotent {
		t.Fatalf("idempotent Admit() = (%+v, %v), want existing admission", duplicate, err)
	}
	blockedRequest := request
	blockedRequest.AdmissionID = "after-backlog"
	blockedRequest.Digest = "after-backlog"
	blocked, err := engine.Admit(context.Background(), blockedRequest)
	if err != nil || blocked.Disposition != AdmissionUnavailable ||
		blocked.BlockingReason != "usage accounting backlog is full" {
		t.Fatalf("backpressured Admit() = (%+v, %v), want unavailable", blocked, err)
	}
	if exists, existsErr := client.Exists(context.Background(), keys.pending(blockedRequest.AdmissionID)).Result(); existsErr != nil || exists != 0 {
		t.Fatalf("backpressured admission pending state = (%d, %v), want absent", exists, existsErr)
	}

	assertUsageBacklogAccounting(t, client, partition, keys)

	resumedRequest := request
	resumedRequest.AdmissionID = "after-acknowledgement"
	resumedRequest.Digest = "after-acknowledgement"
	resumed, err := engine.Admit(context.Background(), resumedRequest)
	if err != nil || !resumed.Allowed() {
		t.Fatalf("Admit() after acknowledgement = (%+v, %v), want allowed", resumed, err)
	}
}

func assertUsageBacklogAccounting(t *testing.T, client *redis.Client, partition string, keys partitionKeys) {
	t.Helper()
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
