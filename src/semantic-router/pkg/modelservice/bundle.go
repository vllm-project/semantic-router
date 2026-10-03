package modelservice

import (
	"context"
	"strconv"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// DefaultBundleWindow bounds how long a parked call waits for the other
// participants of its bundle.
const DefaultBundleWindow = 2 * time.Millisecond

type bundleKey struct{}

// Bundle coalesces the runtime calls made while serving one request stage.
//
// The goroutines of the stage Join the bundle. A runtime call made with the
// bundle's context parks in it; when every joined goroutine has parked or
// left, or the window since the first parked call has passed, the bundle
// sends one /v1/bundle call per runtime process and hands each caller its own
// result. Calls keep their own deadlines: a caller stops waiting when its
// context ends, and its task carries that deadline to the runtime.
type Bundle struct {
	base    context.Context
	window  time.Duration
	mu      sync.Mutex
	active  int
	parked  int
	pending map[*Client][]*bundleCall
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
			if b.parked > 0 && b.parked >= b.active {
				b.flushLocked()
			}
			b.mu.Unlock()
		})
	}
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
	b.pending[client] = append(b.pending[client], call)
	b.parked++
	if b.timer == nil {
		b.timer = time.AfterFunc(b.window, b.flushOnTimer)
	}
	if b.parked >= b.active {
		b.flushLocked()
	}
	b.mu.Unlock()
	select {
	case <-call.done:
		return call.result, call.err
	case <-ctx.Done():
		return api.BundleResult{}, ctx.Err()
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
	pending := b.pending
	b.pending = make(map[*Client][]*bundleCall)
	b.parked = 0
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
	results, err := client.Bundle(ctx, tasks)
	byID := make(map[string]api.BundleResult, len(results))
	for _, result := range results {
		byID[result.Id] = result
	}
	for _, call := range calls {
		result, ok := byID[call.task.Id]
		switch {
		case err != nil:
			call.err = err
		case !ok:
			call.err = ErrFailed
		default:
			call.result = result
		}
		close(call.done)
	}
}
