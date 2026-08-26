package quotaruntime

import (
	"context"
	"errors"
	"fmt"
	"math/big"
	"os"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

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
		Event: `{"admissionId":"calendar-actual-a"}`, EventEvidenceState: "known",
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
		Event: `{"admissionId":"mixed-a"}`, EventEvidenceState: "mixed",
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

	assertMixedFinalizationRuntimeState(t, client, partition, rules, backendIdentity, concurrency)
}

func assertMixedFinalizationRuntimeState(
	t *testing.T,
	client *redis.Client,
	partition string,
	rules []RuleBinding,
	backendIdentity quota.CounterIdentity,
	concurrency RuleBinding,
) {
	t.Helper()
	compiled, err := compileRules(partition, rules)
	if err != nil {
		t.Fatalf("compileRules() error = %v", err)
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
	return accessProjectionPreconditions(credentialKey, activeKey, policyKey, denyKey), denyKey, credentialKey
}

func accessProjectionPreconditions(
	credentialKey string,
	activeKey string,
	policyKey string,
	denyKey string,
) []AdmissionPrecondition {
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
	}
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
	keys, _ := newPartitionKeys(partition)
	if err := client.XGroupCreateMkStream(ctx, keys.usageStream, usageledger.ConsumerGroupName, "0").Err(); err != nil {
		client.Close()
		t.Fatalf("create usage consumer group: %v", err)
	}
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
