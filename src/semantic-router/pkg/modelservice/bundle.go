package modelservice

import (
	"context"
	"strconv"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// DefaultBundleWindow caps how long a parked call waits for the other
// participants of its bundle.
const DefaultBundleWindow = 2 * time.Millisecond

type bundleKey struct{}

// Bundle coalesces the runtime calls of one request stage into one
// /v1/bundle call per runtime process.
//
// The goroutines of the stage Join the bundle. A runtime call made with the
// bundle's context parks in it. The bundle flushes when no participant can
// still add a call (every one is blocked on a call or on its own fan-out, or
// has left), or when the window since the first parked call has passed, and
// hands each caller its own result. Calls keep their own deadlines: a caller
// stops waiting when its context ends, and its task carries that deadline.
type Bundle struct {
	base    context.Context
	window  time.Duration
	mu      sync.Mutex
	active  int
	blocked int
	pending map[*Client][]*bundleCall
	first   time.Time
	timer   *time.Timer
	seq     int
	flushes int
}

type bundleCall struct {
	ctx    context.Context
	task   api.BundleTask
	done   chan struct{}
	result api.BundleResult
	err    error
}

// WithBundle returns a context whose runtime calls are bundled, and the bundle.
// A window of zero or less uses DefaultBundleWindow.
func WithBundle(ctx context.Context, window time.Duration) (context.Context, *Bundle) {
	if window <= 0 {
		window = DefaultBundleWindow
	}
	bundle := &Bundle{base: ctx, window: window, pending: make(map[*Client][]*bundleCall)}
	return context.WithValue(ctx, bundleKey{}, bundle), bundle
}

func bundleFrom(ctx context.Context) *Bundle {
	bundle, _ := ctx.Value(bundleKey{}).(*Bundle)
	return bundle
}

// InBundle reports whether runtime calls made with ctx park in a bundle.
func InBundle(ctx context.Context) bool {
	return bundleFrom(ctx) != nil
}

// Join adds a participant; call the returned function when it has finished.
func (b *Bundle) Join() (leave func()) {
	b.mu.Lock()
	b.active++
	b.mu.Unlock()
	var once sync.Once
	return func() {
		once.Do(func() {
			b.mu.Lock()
			b.active--
			b.flushIfIdleLocked()
			b.mu.Unlock()
		})
	}
}

// Fan runs work(i) for every i in [0, n) in its own goroutine, each a
// participant of the context's bundle, and waits for all of them. While it
// waits, the calling participant cannot add a call, so it counts as blocked.
func Fan(ctx context.Context, n int, work func(i int)) {
	bundle := bundleFrom(ctx)
	if bundle != nil {
		bundle.mu.Lock()
		bundle.active += n
		bundle.blocked++
		bundle.mu.Unlock()
	}
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if bundle != nil {
				defer bundle.leaveFanned()
			}
			work(i)
		}(i)
	}
	wg.Wait()
	if bundle != nil {
		bundle.mu.Lock()
		bundle.blocked--
		bundle.mu.Unlock()
	}
}

func (b *Bundle) leaveFanned() {
	b.mu.Lock()
	b.active--
	b.flushIfIdleLocked()
	b.mu.Unlock()
}

// Flushes reports how many /v1/bundle rounds the bundle has sent (tests, metrics).
func (b *Bundle) Flushes() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.flushes
}

func (b *Bundle) submit(ctx context.Context, client *Client, task api.BundleTask) (api.BundleResult, error) {
	call := &bundleCall{ctx: ctx, task: task, done: make(chan struct{})}
	b.mu.Lock()
	b.seq++
	call.task.Id = strconv.Itoa(b.seq)
	if len(b.pending) == 0 {
		b.first = time.Now()
	}
	b.pending[client] = append(b.pending[client], call)
	b.blocked++
	if b.blocked >= b.active {
		b.flushLocked()
	} else if b.timer == nil {
		b.timer = time.AfterFunc(b.window, b.flushOnTimer)
	}
	b.mu.Unlock()
	defer func() {
		b.mu.Lock()
		b.blocked--
		b.mu.Unlock()
	}()
	select {
	case <-call.done:
		return call.result, call.err
	case <-ctx.Done():
		return api.BundleResult{}, ctx.Err()
	}
}

// flushIfIdleLocked flushes when no participant is still running.
func (b *Bundle) flushIfIdleLocked() {
	if len(b.pending) > 0 && b.blocked >= b.active {
		b.flushLocked()
	}
}

func (b *Bundle) flushOnTimer() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.flushLocked()
}

// flushLocked sends every pending call; the caller holds b.mu.
func (b *Bundle) flushLocked() {
	if b.timer != nil {
		b.timer.Stop()
		b.timer = nil
	}
	if len(b.pending) == 0 {
		return
	}
	bundleWait.Observe(time.Since(b.first).Seconds())
	pending := b.pending
	b.pending = make(map[*Client][]*bundleCall)
	b.flushes++
	for client, calls := range pending {
		go b.send(client, calls)
	}
}

// send makes one /v1/bundle call; its context lasts as long as the latest caller waits.
func (b *Bundle) send(client *Client, calls []*bundleCall) {
	ctx := b.base
	var latest time.Time
	unbounded := false
	for _, call := range calls {
		deadline, ok := call.ctx.Deadline()
		if !ok {
			unbounded = true
			continue
		}
		if deadline.After(latest) {
			latest = deadline
		}
	}
	if !unbounded && !latest.IsZero() {
		var cancel context.CancelFunc
		ctx, cancel = context.WithDeadline(b.base, latest)
		defer cancel()
	}
	tasks := make([]api.BundleTask, len(calls))
	for index, call := range calls {
		tasks[index] = call.task
	}
	bundleTasks.Observe(float64(len(tasks)))
	results, err := client.Bundle(ctx, tasks)
	for index, call := range calls {
		switch {
		case err != nil:
			call.err = err
		case results[index].Id != call.task.Id:
			call.err = ErrFailed
		default:
			call.result = results[index]
		}
		close(call.done)
	}
}
