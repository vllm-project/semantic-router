package upstream

import (
	"container/list"
	"context"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// Envoy's circuit-breaker defaults, which apply even where the template
// renders no circuit_breakers block.
const (
	defaultMaxConnections     = 1024
	defaultMaxPendingRequests = 1024
	defaultMaxRequests        = 1024
	defaultMaxRetries         = 3
)

// Overflow limits, as metric labels.
const (
	limitMaxRequests        = "max_requests"
	limitMaxPendingRequests = "max_pending_requests"
	limitMaxRetries         = "max_retries"
)

func (b Breakers) withDefaults() Breakers {
	defaults := Breakers{
		MaxConnections:     pickInt(b.MaxConnections, defaultMaxConnections),
		MaxPendingRequests: pickInt(b.MaxPendingRequests, defaultMaxPendingRequests),
		MaxRequests:        pickInt(b.MaxRequests, defaultMaxRequests),
		MaxRetries:         pickInt(b.MaxRetries, defaultMaxRetries),
	}
	if b.RetryBudget != nil {
		budget := *b.RetryBudget
		if budget.Percent <= 0 {
			budget.Percent = defaultRetryBudgetPercent
		}
		budget.MinConcurrency = pickInt(budget.MinConcurrency, defaultRetryBudgetMinConcurrency)
		defaults.RetryBudget = &budget
	}
	return defaults
}

func pickInt(value, fallback int) int {
	if value > 0 {
		return value
	}
	return fallback
}

// breaker enforces a cluster's limits the way Envoy's HTTP/1.1 connection
// pool does: a request over max_requests overflows at once; otherwise it
// takes a connection if one is free, or waits for one in a bounded queue.
type breaker struct {
	cluster string
	limits  Breakers

	mu      sync.Mutex
	active  int
	retries int
	waiting list.List // of chan struct{}, oldest first
}

func newBreaker(cluster string, limits Breakers) *breaker {
	return &breaker{cluster: cluster, limits: limits.withDefaults()}
}

// acquire admits one attempt. The returned release must be called once the
// attempt ends, its streamed body included.
func (b *breaker) acquire(ctx context.Context) (func(), *Error) {
	b.mu.Lock()
	switch {
	case b.active >= b.limits.MaxRequests:
		b.mu.Unlock()
		return nil, b.overflow(limitMaxRequests)
	case b.active < b.limits.MaxConnections:
		b.active++
		b.mu.Unlock()
		return b.release, nil
	case b.waiting.Len() >= b.limits.MaxPendingRequests:
		b.mu.Unlock()
		return nil, b.overflow(limitMaxPendingRequests)
	}
	ready := make(chan struct{})
	slot := b.waiting.PushBack(ready)
	b.mu.Unlock()
	select {
	case <-ready:
		return b.release, nil
	case <-ctx.Done():
		b.mu.Lock()
		defer b.mu.Unlock()
		select {
		case <-ready:
			// Handed a connection as the wait ended: pass it on.
			b.releaseLocked()
		default:
			b.waiting.Remove(slot)
		}
		return nil, contextError(ctx)
	}
}

func (b *breaker) release() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.releaseLocked()
}

// releaseLocked frees a connection, or hands it straight to the oldest
// waiting request.
func (b *breaker) releaseLocked() {
	if front := b.waiting.Front(); front != nil {
		b.waiting.Remove(front)
		close(front.Value.(chan struct{}))
		return
	}
	b.active--
}

// acquireRetry admits one retry under max_retries, or under the retry budget:
// a share of the active and pending requests, never below its floor.
func (b *breaker) acquireRetry() (func(), *Error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	limit := b.limits.MaxRetries
	if budget := b.limits.RetryBudget; budget != nil {
		outstanding := float64(b.active + b.waiting.Len())
		limit = max(budget.MinConcurrency, int(budget.Percent*outstanding/100))
	}
	if b.retries >= limit {
		return nil, b.overflow(limitMaxRetries)
	}
	b.retries++
	return func() {
		b.mu.Lock()
		defer b.mu.Unlock()
		b.retries--
	}, nil
}

func (b *breaker) overflow(limit string) *Error {
	metrics.RecordUpstreamOverflow(b.cluster, limit)
	return &Error{Kind: KindOverflow, Cluster: b.cluster, Err: errOverflow(limit)}
}

type errOverflow string

func (e errOverflow) Error() string { return "circuit breaker " + string(e) + " reached" }
