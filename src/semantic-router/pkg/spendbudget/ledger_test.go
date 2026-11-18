package spendbudget

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func testLedger(now *time.Time) *Ledger {
	ledger := NewLedger()
	ledger.now = func() time.Time { return *now }
	return ledger
}

func run(limit float64) Limit {
	return Limit{Key: "run|alice|s1", Scope: "run", Amount: limit, Idle: time.Hour}
}

func TestConcurrentReservationsNeverExceedTheBudget(t *testing.T) {
	now := time.Date(2026, 11, 16, 10, 0, 0, 0, time.UTC)
	ledger := testLedger(&now)
	limits := []Limit{run(10), {Key: "user|daily|alice", Scope: "user/daily", Amount: 100, Window: WindowDay}}
	var admitted atomic.Int64
	var holds sync.Map
	var group sync.WaitGroup
	for i := range 100 {
		group.Go(func() {
			if hold, refusal := ledger.Reserve(limits, 1); refusal == nil {
				admitted.Add(1)
				holds.Store(i, hold)
			}
		})
	}
	group.Wait()
	if admitted.Load() != 10 {
		t.Fatalf("admitted %d calls of estimate 1 under a budget of 10", admitted.Load())
	}
	holds.Range(func(_, hold any) bool {
		hold.(*Hold).Settle(0.5)
		return true
	})
	status, _ := ledger.Tightest(limits)
	if status.Scope != "run" || status.Remaining != 5 {
		t.Fatalf("after settling ten calls at 0.5: %+v", status)
	}
}

func TestSettleReplacesTheReservationWithTheActualCost(t *testing.T) {
	now := time.Date(2026, 11, 16, 10, 0, 0, 0, time.UTC)
	ledger := testLedger(&now)
	limits := []Limit{run(1)}
	hold, refusal := ledger.Reserve(limits, 0.8)
	if refusal != nil {
		t.Fatal(refusal)
	}
	if _, refusal := ledger.Reserve(limits, 0.3); refusal == nil {
		t.Fatal("a reservation of 0.8 left room for 0.3 under a budget of 1")
	}
	hold.Settle(0.2)
	hold.Settle(5) // a second settlement is ignored
	status, _ := ledger.Tightest(limits)
	if status.Remaining != 0.8 {
		t.Fatalf("remaining %v, want 0.8", status.Remaining)
	}
	if _, refusal := ledger.Reserve(limits, 0.3); refusal != nil {
		t.Fatalf("0.3 fits after the actual cost of 0.2 replaced the reservation: %v", refusal)
	}
}

func TestReleaseReturnsTheReservation(t *testing.T) {
	now := time.Date(2026, 11, 16, 10, 0, 0, 0, time.UTC)
	ledger := testLedger(&now)
	limits := []Limit{run(1)}
	hold, _ := ledger.Reserve(limits, 1)
	hold.Release()
	if _, refusal := ledger.Reserve(limits, 1); refusal != nil {
		t.Fatalf("a released reservation still counts: %v", refusal)
	}
}

func TestRefusalNamesTheBudgetThatDoesNotFit(t *testing.T) {
	now := time.Date(2026, 11, 16, 10, 30, 0, 0, time.UTC)
	ledger := testLedger(&now)
	user := Limit{Key: "user|hourly|alice", Scope: "user/hourly", Amount: 2, Window: WindowHour}
	limits := []Limit{run(10), user}
	hold, _ := ledger.Reserve(limits, 1.5)
	hold.Settle(1.5)
	_, refusal := ledger.Reserve(limits, 1)
	if refusal == nil {
		t.Fatal("1 fit in the 0.5 the hourly budget has left")
	}
	if refusal.Scope != "user/hourly" || refusal.Remaining != 0.5 || refusal.Estimate != 1 {
		t.Fatalf("refusal %+v", refusal)
	}
	if want := time.Date(2026, 11, 16, 11, 0, 0, 0, time.UTC); !refusal.ResetAt.Equal(want) {
		t.Fatalf("reset %v, want %v", refusal.ResetAt, want)
	}
	if refusal.Error() == "" {
		t.Fatal("empty refusal message")
	}
}

func TestCheckRefusesOnlyAnExhaustedBudget(t *testing.T) {
	now := time.Date(2026, 11, 16, 10, 0, 0, 0, time.UTC)
	ledger := testLedger(&now)
	limits := []Limit{run(1)}
	if refusal := ledger.Check(limits); refusal != nil {
		t.Fatalf("a fresh budget is refused: %v", refusal)
	}
	hold, _ := ledger.Reserve(limits, 0.9)
	if refusal := ledger.Check(limits); refusal != nil {
		t.Fatalf("a budget with 0.1 left is refused: %v", refusal)
	}
	hold.Settle(1.2)
	refusal := ledger.Check(limits)
	if refusal == nil || refusal.Remaining >= 0 {
		t.Fatalf("an overspent budget is admitted: %+v", refusal)
	}
	if want := "the run spend budget of 1.000000 has nothing left"; refusal.Error() != want {
		t.Fatalf("refusal %q, want %q", refusal.Error(), want)
	}
}

func TestChargeCountsSpendWithoutAReservation(t *testing.T) {
	now := time.Date(2026, 11, 16, 10, 0, 0, 0, time.UTC)
	ledger := testLedger(&now)
	limits := []Limit{run(1)}
	ledger.Charge(limits, 0.25)
	status, _ := ledger.Tightest(limits)
	if status.Remaining != 0.75 {
		t.Fatalf("remaining %v, want 0.75", status.Remaining)
	}
}

func TestWindowResetsAtItsCalendarBoundaryAndKeepsReservations(t *testing.T) {
	now := time.Date(2026, 11, 16, 23, 59, 0, 0, time.UTC)
	ledger := testLedger(&now)
	limits := []Limit{{Key: "user|daily|alice", Scope: "user/daily", Amount: 1, Window: WindowDay}}
	spent, _ := ledger.Reserve(limits, 0.6)
	spent.Settle(0.6)
	inflight, _ := ledger.Reserve(limits, 0.3)
	now = now.Add(2 * time.Minute)
	status, _ := ledger.Tightest(limits)
	if status.Remaining != 0.7 {
		t.Fatalf("a new day keeps only the in-flight 0.3: remaining %v", status.Remaining)
	}
	inflight.Settle(0.1)
	if status, _ = ledger.Tightest(limits); status.Remaining != 0.9 {
		t.Fatalf("the in-flight call settles into the new day: remaining %v", status.Remaining)
	}
}

func TestIdleRunIsForgottenAndOthersAreKept(t *testing.T) {
	now := time.Date(2026, 11, 16, 10, 0, 0, 0, time.UTC)
	ledger := testLedger(&now)
	idle, busy := []Limit{run(1)}, []Limit{{Key: "run|bob|s2", Scope: "run", Amount: 1, Idle: time.Hour}}
	hold, _ := ledger.Reserve(idle, 1)
	hold.Settle(1)
	open, _ := ledger.Reserve(busy, 0.5)
	now = now.Add(2 * time.Hour)
	ledger.prune(now)
	if len(ledger.accounts) != 1 {
		t.Fatalf("%d accounts after pruning, want the open run only", len(ledger.accounts))
	}
	if refusal := ledger.Check(idle); refusal != nil {
		t.Fatalf("an idle run keeps its spend: %v", refusal)
	}
	open.Release()
}

func TestWindowBounds(t *testing.T) {
	at := time.Date(2026, 11, 18, 15, 4, 5, 0, time.FixedZone("KST", 9*3600))
	cases := map[Window][2]time.Time{
		WindowHour:  {time.Date(2026, 11, 18, 6, 0, 0, 0, time.UTC), time.Date(2026, 11, 18, 7, 0, 0, 0, time.UTC)},
		WindowDay:   {time.Date(2026, 11, 18, 0, 0, 0, 0, time.UTC), time.Date(2026, 11, 19, 0, 0, 0, 0, time.UTC)},
		WindowWeek:  {time.Date(2026, 11, 16, 0, 0, 0, 0, time.UTC), time.Date(2026, 11, 23, 0, 0, 0, 0, time.UTC)},
		WindowMonth: {time.Date(2026, 11, 1, 0, 0, 0, 0, time.UTC), time.Date(2026, 12, 1, 0, 0, 0, 0, time.UTC)},
	}
	for window, want := range cases {
		start, end := window.Bounds(at)
		if !start.Equal(want[0]) || !end.Equal(want[1]) {
			t.Errorf("%s: [%v, %v), want [%v, %v)", window, start, end, want[0], want[1])
		}
	}
	if !WindowMonth.Valid() || Window("fortnight").Valid() {
		t.Error("Valid accepts the wrong windows")
	}
}
