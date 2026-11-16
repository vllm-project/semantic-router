package sessiontelemetry

import (
	"errors"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
)

// useSharedBudgetStore points the package at a Redis store backed by miniredis.
// Clearing local memory between calls stands in for a second replica.
func useSharedBudgetStore(t *testing.T) *miniredis.Miniredis {
	t.Helper()
	resetSessionBudget(t)
	server := miniredis.RunT(t)
	store, err := NewRedisRouterSessionStateStore(RedisRouterSessionStoreConfig{Address: server.Addr(), Timeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	SetRouterSessionStateStore(store)
	t.Cleanup(func() { SetRouterSessionStateStore(nil) })
	return server
}

func otherReplica() { ResetRouterSessionMemoryForTesting() }

func ledgerFields(t *testing.T, server *miniredis.Miniredis, session string) map[string]string {
	t.Helper()
	key := "vsr:router-session:v1:" + session + redisBudgetKeySuffix
	fields := map[string]string{}
	if !server.Exists(key) {
		return fields
	}
	keys, err := server.HKeys(key)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range keys {
		fields[field] = server.HGet(key, field)
	}
	return fields
}

func TestSharedHoldCountsOnEveryReplica(t *testing.T) {
	useSharedBudgetStore(t)
	mustReserve(t, "s", 600, 1000)

	otherReplica()
	mustExceed(t, "s", 401, 1000)
	mustReserve(t, "s", 400, 1000)
}

func TestSharedSettleMovesTheHoldToSettled(t *testing.T) {
	server := useSharedBudgetStore(t)
	id := mustReserve(t, "s", 600, 1000)
	RecordSessionUsage(SessionUsageParams{SessionID: "s", Model: "m", PromptTokens: 200, CompletionTokens: 100, Reservation: id})
	ReleaseSessionTokens("s", id)
	if got := ledgerFields(t, server, "s"); len(got) != 1 || got["settled"] != "300" {
		t.Fatalf("ledger after settle and a late release: %v, want only settled=300", got)
	}

	// The snapshot's recorded 300 and the ledger's settled 300 are the same tokens.
	otherReplica()
	mustExceed(t, "s", 701, 1000)
	mustReserve(t, "s", 700, 1000)
}

func TestSharedSettleChargesOnce(t *testing.T) {
	server := useSharedBudgetStore(t)
	id := mustReserve(t, "s", 600, 1000)
	for range 2 {
		RecordSessionUsage(SessionUsageParams{SessionID: "s", Model: "m", PromptTokens: 200, CompletionTokens: 100, Reservation: id})
	}
	if got := ledgerFields(t, server, "s")["settled"]; got != "300" {
		t.Fatalf("ledger settled %q after settling one reservation twice, want 300", got)
	}
}

func TestSharedReleaseChargesNothing(t *testing.T) {
	useSharedBudgetStore(t)
	id := mustReserve(t, "s", 600, 1000)
	ReleaseSessionTokens("s", id)

	otherReplica()
	mustReserve(t, "s", 1000, 1000)
}

func TestSharedMaxInFlightCountsEveryReplica(t *testing.T) {
	useSharedBudgetStore(t)
	reserveCapped := func() error {
		_, err := ReserveSessionTokens(SessionReservationParams{SessionID: "s", Tokens: 1, Limit: 1000, MaxInFlight: 2})
		return err
	}
	for range 2 {
		if err := reserveCapped(); err != nil {
			t.Fatal(err)
		}
		otherReplica()
	}
	if err := reserveCapped(); !errors.Is(err, ErrSessionReservationsFull) {
		t.Fatalf("third open reservation across replicas: got %v, want ErrSessionReservationsFull", err)
	}
}

func TestSharedHoldLapsesWhenNeverSettled(t *testing.T) {
	server := useSharedBudgetStore(t)
	mustReserve(t, "s", 600, 1000)

	otherReplica()
	server.SetTime(time.Now().Add(sharedReservationHold + time.Minute))
	mustReserve(t, "s", 1000, 1000)
}

func TestSharedStoreErrorLeavesNoLocalHold(t *testing.T) {
	server := useSharedBudgetStore(t)
	server.Close()
	if _, err := reserve(t, "s", 600, 1000); err == nil || errors.Is(err, ErrSessionBudgetExceeded) {
		t.Fatalf("reserve with the shared store down: got %v, want a store error", err)
	}
	SetRouterSessionStateStore(nil)
	mustReserve(t, "s", 1000, 1000)
}
