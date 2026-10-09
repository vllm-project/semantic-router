// Package budget bounds model calls across a request's routing and inference
// phases. The context carries one ledger through selectors, signals and retries.
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
	limit int64
	used  atomic.Int64
}

// WithLimit installs a request-wide call limit. A nested caller retains the
// original ledger; creating a child context must never replenish its budget.
// A non-positive limit permits no calls. Requests without a ledger are unbounded.
func WithLimit(ctx context.Context, limit int) (context.Context, *Ledger) {
	if ledger, ok := ctx.Value(contextKey{}).(*Ledger); ok {
		return ctx, ledger
	}
	ledger := &Ledger{limit: int64(max(0, limit))}
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
		if used >= ledger.limit {
			return ErrExhausted
		}
		if ledger.used.CompareAndSwap(used, used+1) {
			return nil
		}
	}
}

// Used reports the number of exchanges reserved by all phases so far.
func (l *Ledger) Used() int { return int(l.used.Load()) }

// Remaining reports the number of additional exchanges the request can afford.
func (l *Ledger) Remaining() int { return int(l.limit - l.used.Load()) }
