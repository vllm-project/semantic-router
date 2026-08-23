package backendinvoker

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestResponseTerminalStoreSingleConsumeAndDuplicateFence(t *testing.T) {
	store := NewLocalResponseTerminalStore()
	plan, attempt := terminalFixture("req", "dispatch")
	reference := terminalReference(plan)
	if _, ok, err := store.Take(context.Background(), reference); err != nil || ok {
		t.Fatal("Take before Finalize unexpectedly returned a record")
	}
	if err := store.Finalize(context.Background(), plan, attempt, successfulTerminal()); err != nil {
		t.Fatalf("Finalize: %v", err)
	}
	if err := store.Finalize(context.Background(), plan, attempt, successfulTerminal()); err == nil {
		t.Fatal("duplicate Finalize was accepted")
	}
	record, ok, err := store.Take(context.Background(), reference)
	if err != nil || !ok || record.Reference != reference {
		t.Fatalf("Take returned %#v, %v, %v", record, ok, err)
	}
	if _, ok, err := store.Take(context.Background(), reference); err != nil || ok {
		t.Fatal("terminal was replayed after destructive Take")
	}
}

func TestResponseTerminalStoreExpiryAndCapacity(t *testing.T) {
	now := time.Date(2026, time.August, 23, 0, 0, 0, 0, time.UTC)
	store := &LocalResponseTerminalStore{
		records: make(map[string]*terminalEntry),
		expiry:  make(terminalExpiryHeap, 0), capacity: 2, ttl: time.Minute,
		now: func() time.Time { return now },
	}
	for index := 0; index < 2; index++ {
		plan, attempt := terminalFixture(fmt.Sprintf("req-%d", index), fmt.Sprintf("dispatch-%d", index))
		if err := store.Finalize(context.Background(), plan, attempt, successfulTerminal()); err != nil {
			t.Fatalf("Finalize %d: %v", index, err)
		}
	}
	plan, attempt := terminalFixture("req-full", "dispatch-full")
	if err := store.Finalize(context.Background(), plan, attempt, successfulTerminal()); err == nil {
		t.Fatal("capacity overflow was accepted")
	}
	now = now.Add(time.Minute)
	if err := store.Finalize(context.Background(), plan, attempt, successfulTerminal()); err != nil {
		t.Fatalf("Finalize after expiry: %v", err)
	}
	if len(store.records) != 1 || len(store.expiry) != 1 {
		t.Fatalf("expired records were not removed: records=%d heap=%d", len(store.records), len(store.expiry))
	}
}

func TestResponseTerminalStoreFiftyThousandBoundedEntries(t *testing.T) {
	store := NewLocalResponseTerminalStore()
	const count = 50_000
	for index := 0; index < count; index++ {
		plan, attempt := terminalFixture(fmt.Sprintf("req-%d", index), fmt.Sprintf("dispatch-%d", index))
		if err := store.Finalize(context.Background(), plan, attempt, successfulTerminal()); err != nil {
			t.Fatalf("Finalize %d: %v", index, err)
		}
	}
	if len(store.records) != count || len(store.expiry) != count {
		t.Fatalf("store size mismatch: records=%d heap=%d", len(store.records), len(store.expiry))
	}
	for index := 0; index < count; index++ {
		plan, _ := terminalFixture(fmt.Sprintf("req-%d", index), fmt.Sprintf("dispatch-%d", index))
		if _, ok, err := store.Take(context.Background(), terminalReference(plan)); err != nil || !ok {
			t.Fatalf("Take %d failed", index)
		}
	}
	if len(store.records) != 0 || len(store.expiry) != 0 {
		t.Fatalf("consumed entries leaked: records=%d heap=%d", len(store.records), len(store.expiry))
	}
}

func TestResponseTerminalStoreConcurrentUniqueOwnership(t *testing.T) {
	store := NewLocalResponseTerminalStore()
	const count = 1_000
	var writers sync.WaitGroup
	errors := make(chan error, count)
	for index := 0; index < count; index++ {

		writers.Add(1)
		go func() {
			defer writers.Done()
			plan, attempt := terminalFixture(fmt.Sprintf("req-%d", index), fmt.Sprintf("dispatch-%d", index))
			errors <- store.Finalize(context.Background(), plan, attempt, successfulTerminal())
		}()
	}
	writers.Wait()
	close(errors)
	for err := range errors {
		if err != nil {
			t.Fatalf("concurrent Finalize: %v", err)
		}
	}
	var readers sync.WaitGroup
	seen := make(chan int, count)
	for index := 0; index < count; index++ {
		for copy := 0; copy < 2; copy++ {
			readers.Add(1)
			go func() {
				defer readers.Done()
				plan, _ := terminalFixture(fmt.Sprintf("req-%d", index), fmt.Sprintf("dispatch-%d", index))
				if _, ok, err := store.Take(context.Background(), terminalReference(plan)); err == nil && ok {
					seen <- index
				}
			}()
		}
	}
	readers.Wait()
	close(seen)
	counts := make(map[int]int, count)
	for index := range seen {
		counts[index]++
	}
	if len(counts) != count {
		t.Fatalf("only %d/%d terminals were consumed", len(counts), count)
	}
	for index, value := range counts {
		if value != 1 {
			t.Fatalf("terminal %d consumed %d times", index, value)
		}
	}
}

func TestResponseTerminalStorePreservesCanceledEvidenceWithoutContextDependency(t *testing.T) {
	store := NewLocalResponseTerminalStore()
	plan, attempt := terminalFixture("req-cancel", "dispatch-cancel")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	terminal := ResponseTerminal{
		Usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable}, StopReason: llmprotocol.StopError,
		Error: llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_canceled", "stream was canceled", context.Canceled),
	}
	if err := store.Finalize(ctx, plan, attempt, terminal); err != nil {
		t.Fatalf("Finalize: %v", err)
	}
	record, ok, err := store.Take(context.Background(), terminalReference(plan))
	if err != nil || !ok || record.Terminal.Error == nil ||
		record.Terminal.Error.Code != "stream_canceled" || record.Terminal.Error.Cause != nil {
		t.Fatalf("cancel evidence missing: %#v", record)
	}
}

func TestResponseTerminalWireExcludesPrivateMaterialAndCanonicalizesZeroTimes(t *testing.T) {
	plan, attempt := terminalFixture("wire-request", "wire-dispatch")
	plan.Body = []byte("private-request-body-that-must-not-cross-replicas")
	attempt.StartedAt = time.Time{}
	attempt.CompletedAt = time.Time{}
	secret := "private-provider-cause-that-must-not-cross-replicas"
	terminal := ResponseTerminal{
		Usage:      llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
		StopReason: llmprotocol.StopError,
		Error: llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable, "upstream_unavailable", "upstream unavailable",
			errors.New(secret),
		),
	}
	reference := terminalReference(plan)
	payload, err := encodeResponseTerminalRecord(ResponseTerminalRecord{
		Reference: reference, Attempt: attempt, Terminal: terminal,
	})
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(payload, []byte(secret)) || bytes.Contains(payload, plan.Body) {
		t.Fatalf("private material entered terminal wire: %s", payload)
	}
	record, err := decodeResponseTerminalRecord(payload)
	if err != nil {
		t.Fatal(err)
	}
	if !record.Attempt.StartedAt.IsZero() || !record.Attempt.CompletedAt.IsZero() ||
		record.Terminal.Error == nil || record.Terminal.Error.Cause != nil {
		t.Fatalf("decoded terminal = %+v", record)
	}
}

func TestResponseTerminalRejectsOpenEndedEnumsAndInvalidTimeOrder(t *testing.T) {
	plan, attempt := terminalFixture("invalid-request", "invalid-dispatch")
	reference := terminalReference(plan)
	terminal := successfulTerminal()
	terminal.StopReason = llmprotocol.StopReason("provider_specific")
	if err := validateResponseTerminalRecord(reference, attempt, terminal); !errors.Is(err, ErrResponseTerminalInvalid) {
		t.Fatalf("unknown stop reason error = %v", err)
	}
	failed := ResponseTerminal{
		Usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable}, StopReason: llmprotocol.StopError,
		Error: &llmprotocol.ProtocolError{Category: llmprotocol.ErrorCategory("provider_specific")},
	}
	if err := validateResponseTerminalRecord(reference, attempt, failed); !errors.Is(err, ErrResponseTerminalInvalid) {
		t.Fatalf("unknown error category error = %v", err)
	}
	attempt.CompletedAt = attempt.StartedAt.Add(-time.Second)
	if err := validateResponseTerminalRecord(reference, attempt, successfulTerminal()); !errors.Is(err, ErrResponseTerminalInvalid) {
		t.Fatalf("reversed attempt time error = %v", err)
	}
}

func TestResponseTerminalKeysDistributeAcrossShardsAndNamespaceDeployments(t *testing.T) {
	prefix := "tenant:{user-controlled}:prefix"
	first := &RedisResponseTerminalStore{keyNamespace: responseTerminalKeyNamespace(prefix)}
	same := &RedisResponseTerminalStore{keyNamespace: responseTerminalKeyNamespace(prefix)}
	other := &RedisResponseTerminalStore{keyNamespace: responseTerminalKeyNamespace(prefix + "-other")}
	seen := make(map[string]struct{}, managedTerminalShardCount)
	for index := 0; index < 20_000 && len(seen) < managedTerminalShardCount; index++ {
		plan, _ := terminalFixture(
			fmt.Sprintf("shard-request-%d", index),
			fmt.Sprintf("shard-dispatch-%d", index),
		)
		reference := terminalReference(plan)
		firstKeys, err := first.keys(reference)
		if err != nil {
			t.Fatal(err)
		}
		sameKeys, err := same.keys(reference)
		if err != nil {
			t.Fatal(err)
		}
		otherKeys, err := other.keys(reference)
		if err != nil {
			t.Fatal(err)
		}
		if firstKeys != sameKeys {
			t.Fatal("the same configured prefix produced unstable terminal keys")
		}
		if firstKeys == otherKeys {
			t.Fatal("different configured prefixes share one terminal key namespace")
		}
		if strings.Contains(firstKeys.record, prefix) || strings.Contains(firstKeys.expiry, prefix) {
			t.Fatal("user-controlled prefix entered the terminal key or hash tag")
		}
		seen[firstKeys.expiry] = struct{}{}
	}
	if len(seen) != managedTerminalShardCount {
		t.Fatalf("terminal references reached %d/%d shards", len(seen), managedTerminalShardCount)
	}
}

func terminalFixture(requestID, dispatchID string) (Plan, AttemptResult) {
	now := time.Date(2026, time.August, 23, 0, 0, 0, 0, time.UTC)
	attempt := Attempt{ID: "attempt-" + dispatchID, Number: 1, BackendID: "backend", StartedAt: now}
	return Plan{
		NamespaceID: "namespace", QuotaPartition: "partition", PublicationID: "publication",
		RuntimeEpoch: 1, RoutingRevision: 1, RoutingDigest: terminalTestDigest("routing"),
		AdmissionID: "admission", AdmissionDigest: terminalTestDigest("admission"),
		RequestID: requestID, DispatchID: dispatchID, DispatchType: "primary",
		Ordinal: 0, Priority: 0, DispatchPlanDigest: terminalTestDigest("plan-" + dispatchID),
		ModelID: "model", ModelRevision: 1,
	}, AttemptResult{Attempt: attempt, State: AttemptResponseStarted, StatusCode: 200, CompletedAt: now.Add(time.Second)}
}

func terminalReference(plan Plan) ResponseTerminalReference {
	reference, err := ResponseTerminalReferenceFromPlan(plan)
	if err != nil {
		panic(err)
	}
	return reference
}

func terminalTestDigest(value string) string {
	digest := sha256.Sum256([]byte(value))
	return hex.EncodeToString(digest[:])
}

func successfulTerminal() ResponseTerminal {
	input, output, total := int64(3), int64(2), int64(5)
	return ResponseTerminal{StopReason: llmprotocol.StopEndTurn, Usage: llmprotocol.Usage{
		State:       llmprotocol.UsageAvailable,
		InputTotal:  llmprotocol.TokenCount{Value: &input, Provenance: llmprotocol.UsageAuthoritative},
		OutputTotal: llmprotocol.TokenCount{Value: &output, Provenance: llmprotocol.UsageAuthoritative},
		Total:       llmprotocol.TokenCount{Value: &total, Provenance: llmprotocol.UsageAuthoritative},
	}}
}
