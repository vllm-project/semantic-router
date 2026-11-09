package sessiontelemetry

import (
	"errors"
	"sync"
	"testing"
	"time"
)

func resetSessionBudget(t *testing.T) {
	t.Helper()
	ResetRouterSessionMemoryForTesting()
	t.Cleanup(ResetRouterSessionMemoryForTesting)
}

func reserve(t *testing.T, session string, tokens, limit int64) (uint64, error) {
	t.Helper()
	return ReserveSessionTokens(SessionReservationParams{SessionID: session, Tokens: tokens, Limit: limit})
}

func mustReserve(t *testing.T, session string, tokens, limit int64) uint64 {
	t.Helper()
	id, err := reserve(t, session, tokens, limit)
	if err != nil {
		t.Fatalf("reserve %d of %d: %v", tokens, limit, err)
	}
	return id
}

func mustExceed(t *testing.T, session string, tokens, limit int64) {
	t.Helper()
	if _, err := reserve(t, session, tokens, limit); !errors.Is(err, ErrSessionBudgetExceeded) {
		t.Fatalf("reserve %d of %d: got %v, want ErrSessionBudgetExceeded", tokens, limit, err)
	}
}

func TestReservationsAndRecordedUsageShareTheLimit(t *testing.T) {
	resetSessionBudget(t)
	RecordSessionUsage(SessionUsageParams{SessionID: "s", Model: "m", PromptTokens: 300, CompletionTokens: 100})

	mustReserve(t, "s", 500, 1000)
	mustExceed(t, "s", 101, 1000)
	mustReserve(t, "s", 100, 1000)
	mustExceed(t, "s", 1, 1000)
	mustReserve(t, "other", 1000, 1000)
}

func TestSettleChargesUsageAndClosesTheReservationOnce(t *testing.T) {
	resetSessionBudget(t)
	id := mustReserve(t, "s", 600, 1000)
	RecordSessionUsage(SessionUsageParams{
		SessionID: "s", Model: "m", PromptTokens: 200, CompletionTokens: 50, Reservation: id,
	})

	snapshot, _ := GetRouterSessionSnapshot("s", time.Now())
	if snapshot.CumulativePromptTokens != 200 || snapshot.CumulativeCompletionTokens != 50 {
		t.Fatalf("settled usage not recorded: %+v", snapshot)
	}
	mustReserve(t, "s", 750, 1000)
	ReleaseSessionTokens("s", id)
	mustExceed(t, "s", 1, 1000)
}

func TestReleaseFreesTheReservationWithoutCharge(t *testing.T) {
	resetSessionBudget(t)
	id := mustReserve(t, "s", 800, 1000)
	mustExceed(t, "s", 201, 1000)
	ReleaseSessionTokens("s", id)

	snapshot, _ := GetRouterSessionSnapshot("s", time.Now())
	if snapshot.CumulativePromptTokens != 0 || snapshot.CumulativeCompletionTokens != 0 {
		t.Fatalf("release charged the session: %+v", snapshot)
	}
	mustReserve(t, "s", 1000, 1000)
}

func TestConcurrentReservationsStayWithinTheLimit(t *testing.T) {
	resetSessionBudget(t)
	const turns = 64
	var wg sync.WaitGroup
	results := make(chan error, turns)
	for range turns {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := reserve(t, "s", 100, 1000)
			results <- err
		}()
	}
	wg.Wait()
	close(results)

	reserved := 0
	for err := range results {
		switch {
		case err == nil:
			reserved++
		case !errors.Is(err, ErrSessionBudgetExceeded):
			t.Fatalf("unexpected error: %v", err)
		}
	}
	if reserved != 10 {
		t.Fatalf("%d of %d concurrent 100-token turns reserved under a 1000-token limit, want 10", reserved, turns)
	}
}

func TestMaxInFlightCapsOpenReservations(t *testing.T) {
	resetSessionBudget(t)
	params := SessionReservationParams{SessionID: "s", Tokens: 10, Limit: 1000, MaxInFlight: 2}
	first, err := ReserveSessionTokens(params)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ReserveSessionTokens(params); err != nil {
		t.Fatal(err)
	}
	if _, err := ReserveSessionTokens(params); !errors.Is(err, ErrSessionReservationsFull) {
		t.Fatalf("third open reservation: got %v, want ErrSessionReservationsFull", err)
	}
	ReleaseSessionTokens("s", first)
	if _, err := ReserveSessionTokens(params); err != nil {
		t.Fatalf("reservation after a release: %v", err)
	}
}

func TestStaleReservationCannotCloseALaterOne(t *testing.T) {
	resetSessionBudget(t)
	now := time.Now()
	setRouterSessionMemoryNowForTesting(func() time.Time { return now })
	t.Cleanup(func() { setRouterSessionMemoryNowForTesting(nil) })

	stale := mustReserve(t, "s", 500, 1000)
	now = now.Add(routerMemoryTTL + time.Minute)
	// A sweep a second ago leaves the expired entry in place.
	globalRouterSessionMemory.mu.Lock()
	globalRouterSessionMemory.lastEvict = now.Add(-time.Second)
	globalRouterSessionMemory.mu.Unlock()
	mustReserve(t, "s", 1000, 1000)
	ReleaseSessionTokens("s", stale)
	mustExceed(t, "s", 1, 1000)
}

func TestReserveRejectsInvalidParams(t *testing.T) {
	resetSessionBudget(t)
	for _, p := range []SessionReservationParams{
		{Tokens: 1, Limit: 10},
		{SessionID: "s", Tokens: -1, Limit: 10},
		{SessionID: "s", Tokens: 1},
		{SessionID: "s", Tokens: 1, Limit: 10, MaxInFlight: -1},
	} {
		if _, err := ReserveSessionTokens(p); err == nil || errors.Is(err, ErrSessionBudgetExceeded) {
			t.Fatalf("%+v: got %v, want an invalid-params error", p, err)
		}
	}
}

func TestUsageWithoutModelReleasesTheReservation(t *testing.T) {
	resetSessionBudget(t)
	id := mustReserve(t, "s", 1000, 1000)
	RecordSessionUsage(SessionUsageParams{SessionID: "s", PromptTokens: 5, Reservation: id})
	mustReserve(t, "s", 1000, 1000)
}
