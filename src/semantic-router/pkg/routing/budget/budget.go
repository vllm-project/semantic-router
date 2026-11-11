// Package budget bounds physical model calls within an execution context.
// The caller chooses where execution begins; nested transports and retries share
// the installed ledger.
package budget

import (
	"context"
	"errors"
	"sync/atomic"
)

// ErrExhausted means no further model exchange is allowed for this request.
var ErrExhausted = errors.New("inference call budget exhausted")

type contextKey struct{}

// Ledger counts physical model HTTP exchanges, not logical stages or GPU
// forwards. One batched exchange can perform multiple model forwards.
type Ledger struct {
	limit atomic.Int64
	used  atomic.Int64
}

// WithLimit installs an execution call limit. A nested caller retains the
// original ledger; creating a child context must never replenish its budget.
// A non-positive limit permits no calls. Requests without a ledger are unbounded.
func WithLimit(ctx context.Context, limit int) (context.Context, *Ledger) {
	if ledger, ok := ctx.Value(contextKey{}).(*Ledger); ok {
		return ctx, ledger
	}
	ledger := &Ledger{}
	ledger.limit.Store(int64(max(0, limit)))
	return context.WithValue(ctx, contextKey{}, ledger), ledger
}

// Consume reserves one exchange immediately before transport. A reservation is
// not refunded on errors: failures and retries use the same request budget.
func Consume(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	ledger, ok := ctx.Value(contextKey{}).(*Ledger)
	if !ok {
		return nil
	}
	for {
		used := ledger.used.Load()
		if used >= ledger.limit.Load() {
			return ErrExhausted
		}
		if ledger.used.CompareAndSwap(used, used+1) {
			return nil
		}
	}
}

// TightenLimit lowers a request limit without replenishing consumed calls.
// It returns false when the request has already used the tightened limit.
func (l *Ledger) TightenLimit(limit int) bool {
	if l == nil {
		return false
	}
	target := int64(max(0, limit))
	for {
		current := l.limit.Load()
		if target >= current || l.limit.CompareAndSwap(current, target) {
			return l.used.Load() < l.limit.Load()
		}
	}
}

// Unmetered returns ctx without its call ledger, for an exchange its caller's
// contract keeps out of the call budget. The deadline and every other value
// stay. The untyped nil reads as no ledger.
func Unmetered(ctx context.Context) context.Context {
	return context.WithValue(ctx, contextKey{}, nil)
}

// Current returns the request ledger carried by ctx, when one is installed.
func Current(ctx context.Context) (*Ledger, bool) {
	ledger, ok := ctx.Value(contextKey{}).(*Ledger)
	return ledger, ok
}

// Used reports the number of exchanges reserved in this execution so far.
func (l *Ledger) Used() int { return int(l.used.Load()) }

// Remaining reports the number of additional exchanges the request can afford.
func (l *Ledger) Remaining() int { return int(l.limit.Load() - l.used.Load()) }
