// Package spendbudget holds spend budgets across requests. A call reserves
// its estimated cost against every budget it belongs to before it is
// dispatched, and settles its actual cost afterwards, so calls in flight at
// once cannot together spend past a budget.
package spendbudget

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// Window is the calendar span, in UTC, a budget resets over. The empty
// window never resets: its account lasts while it is used.
type Window string

const (
	WindowHour  Window = "hour"
	WindowDay   Window = "day"
	WindowWeek  Window = "week"
	WindowMonth Window = "month"
)

// Valid reports whether w is a window a budget may name.
func (w Window) Valid() bool {
	switch w {
	case WindowHour, WindowDay, WindowWeek, WindowMonth:
		return true
	}
	return false
}

// Bounds returns the start of the window holding t and the next start. A
// week starts on Monday.
func (w Window) Bounds(t time.Time) (time.Time, time.Time) {
	t = t.UTC()
	day := time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)
	switch w {
	case WindowHour:
		start := t.Truncate(time.Hour)
		return start, start.Add(time.Hour)
	case WindowDay:
		return day, day.AddDate(0, 0, 1)
	case WindowWeek:
		start := day.AddDate(0, 0, -(int(day.Weekday())+6)%7)
		return start, start.AddDate(0, 0, 7)
	case WindowMonth:
		start := time.Date(t.Year(), t.Month(), 1, 0, 0, 0, 0, time.UTC)
		return start, start.AddDate(0, 1, 0)
	}
	return time.Time{}, time.Time{}
}

// Limit is one budget a call belongs to. Key names its account; Scope is the
// name a refusal, a header and Replay report.
type Limit struct {
	Key    string
	Scope  string
	Amount float64
	Window Window
	// Idle forgets an account without a window once it has been unused
	// this long.
	Idle time.Duration
}

// Status is a budget's state. Remaining falls below zero when calls cost
// more than their estimates.
type Status struct {
	Scope     string
	Limit     float64
	Remaining float64
	// ResetAt is when the budget's window starts again; zero without one.
	ResetAt time.Time
}

// Refusal is the budget a call does not fit.
type Refusal struct {
	Status
	Estimate float64
}

func (r *Refusal) Error() string {
	if r.Remaining <= 0 {
		return fmt.Sprintf("the %s spend budget of %s has nothing left", r.Scope, formatAmount(r.Limit))
	}
	return fmt.Sprintf("the %s spend budget of %s has %s left, and this call is estimated at %s",
		r.Scope, formatAmount(r.Limit), formatAmount(r.Remaining), formatAmount(r.Estimate))
}

func formatAmount(amount float64) string {
	return fmt.Sprintf("%.6f", amount)
}

type account struct {
	spent, reserved float64
	start, end      time.Time
	used            time.Time
	idle            time.Duration
}

// Ledger holds every account under one lock, so a reservation against
// several budgets is taken or refused as one.
type Ledger struct {
	mu        sync.Mutex
	accounts  map[string]*account
	now       func() time.Time
	lastPrune time.Time
}

// pruneEvery is how often accounts nothing can use any more are dropped.
const pruneEvery = time.Minute

func NewLedger() *Ledger {
	return &Ledger{accounts: make(map[string]*account), now: time.Now}
}

var process = NewLedger()

// Process is the ledger every router generation of this process shares: a
// configuration reload changes the limits, not the spend.
func Process() *Ledger { return process }

// Hold is a reservation; the first Settle or Release ends it.
type Hold struct {
	ledger   *Ledger
	limits   []Limit
	estimate float64
	done     bool
}

// Estimate is the amount the hold reserved.
func (h *Hold) Estimate() float64 { return h.estimate }

// Reserve takes estimate from every budget in limits, or refuses with the
// tightest one it does not fit. A budget with nothing left refuses even a
// call estimated at zero.
func (l *Ledger) Reserve(limits []Limit, estimate float64) (*Hold, *Refusal) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if refusal := l.admit(limits, estimate); refusal != nil {
		return nil, refusal
	}
	for _, limit := range limits {
		l.account(limit).reserved += estimate
	}
	return &Hold{ledger: l, limits: limits, estimate: estimate}, nil
}

// Check refuses when a budget in limits has nothing left, reserving nothing.
func (l *Ledger) Check(limits []Limit) *Refusal {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.admit(limits, 0)
}

func (l *Ledger) admit(limits []Limit, estimate float64) *Refusal {
	var refusal *Refusal
	for _, limit := range limits {
		status := l.status(limit)
		if (status.Remaining <= 0 || estimate > status.Remaining) &&
			(refusal == nil || status.Remaining < refusal.Remaining) {
			refusal = &Refusal{Status: status, Estimate: estimate}
		}
	}
	return refusal
}

// Charge adds spend no reservation covered, such as a model call the Router
// made before the request's dispatch.
func (l *Ledger) Charge(limits []Limit, amount float64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, limit := range limits {
		l.account(limit).spent += amount
	}
}

// Tightest is the budget in limits with the least left.
func (l *Ledger) Tightest(limits []Limit) (Status, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	var tightest Status
	found := false
	for _, limit := range limits {
		if status := l.status(limit); !found || status.Remaining < tightest.Remaining {
			tightest, found = status, true
		}
	}
	return tightest, found
}

// Settle replaces the reservation with the actual cost.
func (h *Hold) Settle(actual float64) { h.finish(actual) }

// Release returns the reservation: the call cost nothing.
func (h *Hold) Release() { h.finish(0) }

func (h *Hold) finish(actual float64) {
	if h == nil {
		return
	}
	l := h.ledger
	l.mu.Lock()
	defer l.mu.Unlock()
	if h.done {
		return
	}
	h.done = true
	for _, limit := range h.limits {
		acct := l.account(limit)
		acct.reserved = max(acct.reserved-h.estimate, 0)
		acct.spent += actual
	}
}

func (l *Ledger) status(limit Limit) Status {
	acct := l.account(limit)
	remaining := Round(limit.Amount - acct.spent - acct.reserved)
	return Status{Scope: limit.Scope, Limit: limit.Amount, Remaining: remaining, ResetAt: acct.end}
}

// account returns limit's account, opened or rolled into the current
// window. A reservation in flight across a window boundary stays reserved
// and settles into the new window.
func (l *Ledger) account(limit Limit) *account {
	now := l.now()
	l.prune(now)
	acct, ok := l.accounts[limit.Key]
	if !ok {
		acct = &account{}
		l.accounts[limit.Key] = acct
	}
	acct.used, acct.idle = now, limit.Idle
	if limit.Window.Valid() {
		if start, end := limit.Window.Bounds(now); !start.Equal(acct.start) {
			acct.spent, acct.start, acct.end = 0, start, end
		}
	}
	return acct
}

// prune drops, at most once a minute, the accounts nothing holds a
// reservation on whose window has ended or that have been idle too long.
func (l *Ledger) prune(now time.Time) {
	if now.Sub(l.lastPrune) < pruneEvery {
		return
	}
	l.lastPrune = now
	for key, acct := range l.accounts {
		expired := !acct.end.IsZero() && !now.Before(acct.end) ||
			acct.end.IsZero() && acct.idle > 0 && now.Sub(acct.used) >= acct.idle
		if acct.reserved == 0 && expired {
			delete(l.accounts, key)
		}
	}
}

// Round keeps amounts free of float noise: a billionth of a currency unit
// is far below any price.
func Round(amount float64) float64 {
	return math.Round(amount*1e9) / 1e9
}
