package usageledger

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"strings"
	"testing"
	"time"
)

const (
	testNamespaceID  = "11111111-1111-4111-8111-111111111111"
	testKeyID        = "22222222-2222-4222-8222-222222222222"
	testUserID       = "33333333-3333-4333-8333-333333333333"
	testTeamID       = "44444444-4444-4444-8444-444444444444"
	testEntrypointID = "entrypoint_test"
	testRuleID       = "rule_test"
	testRecipeID     = "recipe_test"
	testModelID      = "model_test"
	testBackendID    = "99999999-9999-4999-8999-999999999999"
)

func TestTerminalEventPreservesPartialUsageAndExactCost(t *testing.T) {
	event := testTerminalEvent("mixed-1", time.Date(2026, 8, 22, 12, 0, 10, 0, time.UTC))
	unknown := event.Dispatches[0]
	unknown.DispatchID = "dispatch-2"
	unknown.Ordinal = 1
	unknown.UsageState = UsageUnknown
	unknown.UnknownReason = "provider_usage_missing"
	unknown.InputTokens = "0"
	unknown.OutputTokens = "0"
	unknown.Cost = DispatchCost{Currency: "USD", State: CostUnknown, Numerator: "0", Reason: "usage_missing"}
	unknown.Attempts = []Attempt{{
		AttemptID: "attempt-2", Ordinal: 0, State: UsageUnknown,
		StartedAt: unknown.StartedAt, CompletedAt: unknown.CompletedAt,
	}}
	event.Dispatches = append(event.Dispatches, unknown)
	event.EvidenceState = EvidenceMixed

	aggregate, err := event.Validate()
	if err != nil {
		t.Fatal(err)
	}
	if aggregate.UsageState != UsageUnknown || aggregate.InputTokens.String() != "100" ||
		aggregate.OutputTokens.String() != "20" || aggregate.IncompleteDispatches.String() != "1" {
		t.Fatalf("aggregate = %+v, want known lower bound plus explicit incomplete dispatch", aggregate)
	}
	if len(aggregate.Costs) != 1 || aggregate.Costs[0].KnownNumerator.String() != "250000000000000" ||
		aggregate.Costs[0].KnownDispatches.String() != "1" || aggregate.Costs[0].IncompleteDispatches.String() != "1" {
		t.Fatalf("cost aggregate = %+v, want exact partial USD amount", aggregate.Costs)
	}
	public := publicCosts(aggregate.Costs)
	if public[0].KnownAmount != "0.25" || public[0].Completeness != CompletenessPartial {
		t.Fatalf("public cost = %+v, want exact 0.25 partial", public[0])
	}
}

func TestTerminalEventRejectsSecretLikeMetadataAndUnknownJSONFields(t *testing.T) {
	event := testTerminalEvent("redaction-1", time.Date(2026, 8, 22, 12, 0, 10, 0, time.UTC))
	event.Metadata = map[string]string{"client_name": "Bearer secret"}
	if _, err := event.Validate(); !errors.Is(err, ErrInvalidEvent) {
		t.Fatalf("secret metadata error = %v, want ErrInvalidEvent", err)
	}
	event.Metadata = nil
	payload, err := EncodeTerminalEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	payload = strings.TrimSuffix(payload, "}") + `,"authorization":"secret"}`
	if _, err := DecodeTerminalEvent(payload); !errors.Is(err, ErrInvalidEvent) {
		t.Fatalf("unknown JSON field error = %v, want ErrInvalidEvent", err)
	}
	event = testTerminalEvent("redaction-2", time.Date(2026, 8, 22, 12, 0, 10, 0, time.UTC))
	event.Dispatches[0].ProviderModelID = "Bearer hidden-credential"
	if _, err := event.Validate(); !errors.Is(err, ErrInvalidEvent) {
		t.Fatalf("secret provider Model ID error = %v, want ErrInvalidEvent", err)
	}
}

func TestTerminalEventRejectsRetryAfterInferenceMayHaveStarted(t *testing.T) {
	event := testTerminalEvent("retry-1", time.Date(2026, 8, 22, 12, 0, 10, 0, time.UTC))
	dispatch := &event.Dispatches[0]
	dispatch.Attempts = []Attempt{
		{AttemptID: "attempt-0", Ordinal: 0, State: UsageUnknown, StartedAt: dispatch.StartedAt, CompletedAt: dispatch.StartedAt.Add(time.Millisecond)},
		{AttemptID: "attempt-1", Ordinal: 1, State: UsageKnownActual, StatusCode: 200, StartedAt: dispatch.StartedAt.Add(time.Millisecond), CompletedAt: dispatch.CompletedAt},
	}
	if _, err := event.Validate(); !errors.Is(err, ErrInvalidEvent) {
		t.Fatalf("unsafe retry error = %v, want ErrInvalidEvent", err)
	}
}

func TestTerminalEventAcceptsOnlyProvenZeroAttemptBeforeRetry(t *testing.T) {
	event := testTerminalEvent("retry-safe", time.Date(2026, 8, 22, 12, 0, 10, 0, time.UTC))
	dispatch := &event.Dispatches[0]
	dispatch.Attempts = []Attempt{
		{AttemptID: "attempt-0", Ordinal: 0, State: UsageKnownZero, ErrorCode: "connect_refused", StartedAt: dispatch.StartedAt, CompletedAt: dispatch.StartedAt.Add(time.Millisecond)},
		{AttemptID: "attempt-1", Ordinal: 1, BackendID: testBackendID, ProviderID: "openai", State: UsageKnownActual, StatusCode: 200, StartedAt: dispatch.StartedAt.Add(time.Millisecond), CompletedAt: dispatch.CompletedAt},
	}
	aggregate, err := event.Validate()
	if err != nil {
		t.Fatal(err)
	}
	if aggregate.UsageState != UsageKnownActual || aggregate.InputTokens.String() != "100" || aggregate.OutputTokens.String() != "20" {
		t.Fatalf("aggregate = %+v, want one authoritative terminal attempt", aggregate)
	}
}

func TestCursorRejectsTamperingAndBindsQuery(t *testing.T) {
	codec, err := NewLogCursorCodec([]byte("0123456789abcdef0123456789abcdef"))
	if err != nil {
		t.Fatal(err)
	}
	value := logCursor{Version: 1, NamespaceID: testNamespaceID, QueryDigest: digestHex("query"), OccurredAt: 42, EventID: testEventID("cursor")}
	encoded, err := codec.encode(value)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := codec.decode(encoded)
	if err != nil || decoded != value {
		t.Fatalf("decode = (%+v, %v), want %+v", decoded, err, value)
	}
	tamperedBytes := []byte(encoded)
	if tamperedBytes[0] == 'A' {
		tamperedBytes[0] = 'B'
	} else {
		tamperedBytes[0] = 'A'
	}
	tampered := string(tamperedBytes)
	if _, err := codec.decode(tampered); err == nil {
		t.Fatal("tampered cursor unexpectedly decoded")
	}
}

func TestAutomaticGrainBoundsPointCardinality(t *testing.T) {
	start := time.Date(2026, 8, 22, 0, 0, 0, 0, time.UTC)
	for _, test := range []struct {
		duration time.Duration
		want     Grain
	}{{6 * time.Hour, GrainMinute}, {7 * 24 * time.Hour, GrainHour}, {90 * 24 * time.Hour, GrainDay}} {
		if got := selectGrain(start, start.Add(test.duration), GrainAuto); got != test.want {
			t.Errorf("selectGrain(%s) = %s, want %s", test.duration, got, test.want)
		}
	}
}

type fakeStream struct {
	claimed []StreamItem
	new     []StreamItem
	acked   []string
	ackErr  error
}

func (s *fakeStream) EnsureGroup(context.Context) error { return nil }
func (s *fakeStream) ReadNew(context.Context, int64, time.Duration) ([]StreamItem, error) {
	return s.new, nil
}

func (s *fakeStream) ClaimStale(context.Context, int64, time.Duration) ([]StreamItem, error) {
	return s.claimed, nil
}

func (s *fakeStream) Ack(_ context.Context, ids []string) error {
	s.acked = append(s.acked, ids...)
	return s.ackErr
}

type fakeStore struct {
	result BatchResult
	err    error
	events []TerminalEvent
}

func (s *fakeStore) PersistBatch(_ context.Context, events []TerminalEvent) (BatchResult, error) {
	s.events = append(s.events, events...)
	return s.result, s.err
}

func TestWorkerAcknowledgesOnlyAfterDurableCommit(t *testing.T) {
	event := testTerminalEvent("worker-1", time.Date(2026, 8, 22, 12, 0, 10, 0, time.UTC))
	payload, err := EncodeTerminalEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	item := StreamItem{ID: "1-0", Values: streamValues(event, payload)}
	stream := &fakeStream{new: []StreamItem{item}}
	store := &fakeStore{err: errors.New("database unavailable")}
	worker, err := NewWorker(stream, store, WorkerOptions{NamespaceID: testNamespaceID, BatchSize: 10, Block: time.Millisecond, ReclaimIdle: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := worker.ProcessOnce(context.Background()); err == nil {
		t.Fatal("database failure unexpectedly succeeded")
	}
	if len(stream.acked) != 0 {
		t.Fatalf("acked = %v before commit", stream.acked)
	}
	store.err = nil
	store.result = BatchResult{Inserted: 1}
	if _, err := worker.ProcessOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(stream.acked) != 1 || stream.acked[0] != item.ID {
		t.Fatalf("acked = %v, want committed item", stream.acked)
	}
}

type failingCommitHook struct{ err error }

func (hook failingCommitHook) AfterCommit(context.Context, []TerminalEvent) error { return hook.err }

func TestWorkerLeavesCommittedItemPendingUntilRollupSucceeds(t *testing.T) {
	event := testTerminalEvent("worker-rollup", time.Date(2026, 8, 22, 12, 0, 10, 0, time.UTC))
	payload, err := EncodeTerminalEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	stream := &fakeStream{new: []StreamItem{{ID: "2-0", Values: streamValues(event, payload)}}}
	store := &fakeStore{result: BatchResult{Inserted: 1}}
	worker, err := NewWorker(stream, store, WorkerOptions{
		NamespaceID: testNamespaceID, BatchSize: 10, Block: time.Millisecond,
		ReclaimIdle: time.Second, AfterCommit: failingCommitHook{err: errors.New("rollup unavailable")},
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := worker.ProcessOnce(context.Background())
	if err == nil || result.Inserted != 1 {
		t.Fatalf("ProcessOnce() = (%+v, %v), want committed ledger and failed projection", result, err)
	}
	if len(store.events) != 1 || len(stream.acked) != 0 {
		t.Fatalf("committed events = %d, acknowledgements = %v", len(store.events), stream.acked)
	}
}

func testTerminalEvent(admission string, occurred time.Time) TerminalEvent {
	completed := occurred.Add(250 * time.Millisecond)
	evidence := digestHex("evidence-" + admission)
	return TerminalEvent{
		Schema: TerminalEventSchema, EventID: testEventID(admission), NamespaceID: testNamespaceID,
		AdmissionID: admission, FinalizationDigest: digestHex("final-" + admission), EvidenceState: EvidenceKnown,
		Protocol: "openai.chat", Path: "/v1/chat/completions", StatusCode: 200,
		OccurredAt: occurred, CompletedAt: completed, LatencyMilliseconds: 250,
		Principal: PrincipalSnapshot{APIKeyID: testKeyID, UserID: testUserID, TeamID: testTeamID, APIKeyName: "developer"},
		Routing:   RoutingSnapshot{EntrypointID: testEntrypointID, EntrypointRuleID: testRuleID, RecipeID: testRecipeID, RecipeRevision: 2, RoutingRevision: 3, AccessRevision: 4},
		Served:    ServedUsage{InputTokens: "100", InputKnown: true, OutputTokens: "20", OutputKnown: true},
		Dispatches: []Dispatch{{
			DispatchID: "dispatch-1", Ordinal: 0, DispatchType: "primary", ModelID: testModelID,
			ModelRevision: 7, BackendID: testBackendID, ProviderID: "openai", ProviderModelID: "model-a",
			PricingRevision: 7, InputTokens: "100", CacheReadTokens: "0", CacheWriteTokens: "0",
			OutputTokens: "20", UsageState: UsageKnownActual,
			Cost:           DispatchCost{Currency: "USD", State: CostComplete, Numerator: "250000000000000"},
			EvidenceDigest: evidence, StartedAt: occurred.Add(10 * time.Millisecond), CompletedAt: completed,
			Attempts: []Attempt{{AttemptID: "attempt-1", Ordinal: 0, BackendID: testBackendID, ProviderID: "openai", State: UsageKnownActual, StatusCode: 200, StartedAt: occurred.Add(10 * time.Millisecond), CompletedAt: completed}},
		}},
	}
}

func TestTerminalEventAllowsFenceForUnknownServedUsageWithKnownBackendUsage(t *testing.T) {
	event := testTerminalEvent("served-usage-fence", time.Date(2026, 8, 22, 12, 0, 10, 0, time.UTC))
	event.Served = ServedUsage{InputTokens: "0", OutputTokens: "0"}
	event.Fence = &UnknownFence{
		FenceID: "00000000-0000-4000-8000-000000000301",
		Reason:  "served_total_tokens_missing",
		Bindings: []FenceBinding{{
			BindingID: "00000000-0000-4000-8000-000000000302",
			RuleID:    "00000000-0000-4000-8000-000000000303",
		}},
	}
	if _, err := event.Validate(); err != nil {
		t.Fatalf("known backend usage with an unknown served-usage fence was rejected: %v", err)
	}
}

func TestTerminalEventRejectsFenceWhenAllAccountingIsComplete(t *testing.T) {
	event := testTerminalEvent("complete-usage-fence", time.Date(2026, 8, 22, 12, 0, 10, 0, time.UTC))
	event.Fence = &UnknownFence{
		FenceID: "00000000-0000-4000-8000-000000000311",
		Reason:  "authoritative_usage_missing",
		Bindings: []FenceBinding{{
			BindingID: "00000000-0000-4000-8000-000000000312",
			RuleID:    "00000000-0000-4000-8000-000000000313",
		}},
	}
	if _, err := event.Validate(); err == nil {
		t.Fatal("complete accounting accepted an unknown-usage fence")
	}
}

func streamValues(event TerminalEvent, payload string) map[string]string {
	return map[string]string{
		"admission_id": event.AdmissionID, "admission_digest": digestHex("admission-" + event.AdmissionID),
		"finalization_digest": event.FinalizationDigest, "evidence_state": string(event.EvidenceState), "event": payload,
	}
}

func digestHex(value string) string {
	digest := sha256.Sum256([]byte(value))
	return hex.EncodeToString(digest[:])
}

func testEventID(value string) string {
	digest := sha256.Sum256([]byte(value))
	bytes := digest[:16]
	bytes[6] = bytes[6]&0x0f | 0x40
	bytes[8] = bytes[8]&0x3f | 0x80
	hexValue := hex.EncodeToString(bytes)
	return hexValue[:8] + "-" + hexValue[8:12] + "-" + hexValue[12:16] + "-" + hexValue[16:20] + "-" + hexValue[20:]
}
