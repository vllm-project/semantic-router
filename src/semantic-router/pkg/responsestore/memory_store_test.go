package responsestore

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func newChainTestStore(t *testing.T) *MemoryStore {
	t.Helper()

	store, err := NewMemoryStore(StoreConfig{Enabled: true, TTLSeconds: 60})
	if err != nil {
		t.Fatalf("NewMemoryStore() error = %v", err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Errorf("MemoryStore.Close() error = %v", err)
		}
	})
	return store
}

func mustStoreResponse(t *testing.T, store *MemoryStore, response *responseapi.StoredResponse) {
	t.Helper()

	if err := store.StoreResponse(context.Background(), response); err != nil {
		t.Fatalf("StoreResponse(%s) error = %v", response.ID, err)
	}
}

// getConversationChainWithTimeout fails the test instead of hanging it if the
// traversal never terminates.
func getConversationChainWithTimeout(t *testing.T, store *MemoryStore, responseID string) ([]*responseapi.StoredResponse, error) {
	t.Helper()

	type result struct {
		chain []*responseapi.StoredResponse
		err   error
	}
	done := make(chan result, 1)
	go func() {
		chain, err := store.GetConversationChain(context.Background(), responseID)
		done <- result{chain: chain, err: err}
	}()

	select {
	case res := <-done:
		return res.chain, res.err
	case <-time.After(5 * time.Second):
		t.Fatalf("GetConversationChain(%s) hung - circular reference protection failed", responseID)
		return nil, nil
	}
}

func TestMemoryStoreGetConversationChain(t *testing.T) {
	store := newChainTestStore(t)

	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_1"})
	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_2", PreviousResponseID: "resp_1"})
	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_3", PreviousResponseID: "resp_2"})

	chain, err := getConversationChainWithTimeout(t, store, "resp_3")
	if err != nil {
		t.Fatalf("GetConversationChain(resp_3) error = %v", err)
	}
	if len(chain) != 3 {
		t.Fatalf("chain length = %d, want 3", len(chain))
	}
	for i, wantID := range []string{"resp_1", "resp_2", "resp_3"} {
		if chain[i].ID != wantID {
			t.Errorf("chain[%d].ID = %q, want %q", i, chain[i].ID, wantID)
		}
	}
}

func TestMemoryStoreGetConversationChainCircularReference(t *testing.T) {
	store := newChainTestStore(t)

	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_circular_1", PreviousResponseID: "resp_circular_2"})
	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_circular_2", PreviousResponseID: "resp_circular_1"})

	chain, err := getConversationChainWithTimeout(t, store, "resp_circular_1")
	if err == nil {
		t.Fatalf("GetConversationChain(resp_circular_1) = %d entries, err = nil; want an error", len(chain))
	}

	// The store must stay writable after walking a circular chain.
	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_after_cycle"})
}

func TestMemoryStoreGetConversationChainSelfReference(t *testing.T) {
	store := newChainTestStore(t)

	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_self", PreviousResponseID: "resp_self"})

	chain, err := getConversationChainWithTimeout(t, store, "resp_self")
	if err == nil {
		t.Fatalf("GetConversationChain(resp_self) = %d entries, err = nil; want an error", len(chain))
	}
}

const logSinkTimeout = 5 * time.Second

// blockingLogSink stands in for a log destination that has stopped accepting
// writes, such as a full pipe: every write blocks until the gate is released.
type blockingLogSink struct {
	entered     chan struct{}
	gate        chan struct{}
	enterOnce   sync.Once
	releaseOnce sync.Once

	mu    sync.Mutex
	lines []string
}

func (s *blockingLogSink) Write(p []byte) (int, error) {
	s.enterOnce.Do(func() { close(s.entered) })
	<-s.gate

	s.mu.Lock()
	defer s.mu.Unlock()
	s.lines = append(s.lines, string(p))
	return len(p), nil
}

func (s *blockingLogSink) release() {
	s.releaseOnce.Do(func() { close(s.gate) })
}

func (s *blockingLogSink) written() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.lines...)
}

// useBlockingLogSink points the global logger at a sink that blocks every
// warning. The gate is released on cleanup so a failing test cannot leave a
// goroutine stuck in a write while the store is closed.
func useBlockingLogSink(t *testing.T) *blockingLogSink {
	t.Helper()

	sink := &blockingLogSink{entered: make(chan struct{}), gate: make(chan struct{})}
	core := zapcore.NewCore(zapcore.NewConsoleEncoder(zap.NewProductionEncoderConfig()), zapcore.AddSync(sink), zapcore.WarnLevel)
	t.Cleanup(zap.ReplaceGlobals(zap.New(core)))
	t.Cleanup(sink.release)
	return sink
}

func waitForLogSinkEvent(t *testing.T, done <-chan struct{}, event string) {
	t.Helper()

	select {
	case <-done:
	case <-time.After(logSinkTimeout):
		t.Fatalf("timed out waiting for %s", event)
	}
}

// The cycle warning goes through a synchronous logger whose destination can
// stall. A write stuck on that destination must not also hold the store lock,
// or every writer waits for the log sink.
func TestMemoryStoreGetConversationChainCycleWarningDoesNotBlockWriters(t *testing.T) {
	store := newChainTestStore(t)

	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_log_1", PreviousResponseID: "resp_log_2"})
	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_log_2", PreviousResponseID: "resp_log_1"})
	sink := useBlockingLogSink(t)

	var walkErr error
	walkDone := make(chan struct{})
	go func() {
		defer close(walkDone)
		_, walkErr = store.GetConversationChain(context.Background(), "resp_log_1")
	}()
	waitForLogSinkEvent(t, sink.entered, "the cycle warning to reach the log sink")

	var storeErr error
	storeDone := make(chan struct{})
	started := time.Now()
	go func() {
		defer close(storeDone)
		storeErr = store.StoreResponse(context.Background(), &responseapi.StoredResponse{ID: "resp_log_writer"})
	}()
	stalled := false
	select {
	case <-storeDone:
		t.Logf("StoreResponse returned after %v while the log sink was blocked", time.Since(started))
	case <-time.After(logSinkTimeout):
		stalled = true
	}

	sink.release()
	waitForLogSinkEvent(t, walkDone, "GetConversationChain to return once the log sink is released")
	waitForLogSinkEvent(t, storeDone, "StoreResponse to return once the log sink is released")

	if stalled {
		t.Errorf("StoreResponse blocked for %v until the log sink was released: the log write holds the store lock", logSinkTimeout)
	}
	if storeErr != nil {
		t.Errorf("StoreResponse(resp_log_writer) error = %v", storeErr)
	}
	if want := "response chain loops back to resp_log_1"; walkErr == nil || walkErr.Error() != want {
		t.Errorf("GetConversationChain(resp_log_1) error = %v, want %q", walkErr, want)
	}
	lines := sink.written()
	if len(lines) != 1 || !strings.Contains(lines[0], "MemoryStore: circular reference detected at resp_log_1") {
		t.Errorf("log lines = %q, want one circular reference warning for resp_log_1", lines)
	}
}

// An ancestor that expired before its descendants leaves the chain incomplete:
// continuing with the remainder answers a shorter conversation as if it were
// the whole one.
func TestMemoryStoreGetConversationChainExpiredAncestor(t *testing.T) {
	store := newChainTestStore(t)

	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_root_expired", TTL: time.Now().Add(-time.Minute)})
	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_mid", PreviousResponseID: "resp_root_expired"})
	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_leaf", PreviousResponseID: "resp_mid"})

	chain, err := store.GetConversationChain(context.Background(), "resp_leaf")
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetConversationChain(resp_leaf) = %d entries, err = %v; want ErrNotFound", len(chain), err)
	}
}

// The cap is a loop guard, not a truncation policy: a walk that stops with
// history still ahead of it cannot answer as if it had reached the start.
func TestMemoryStoreGetConversationChainRejectsAChainPastTheCap(t *testing.T) {
	store := newChainTestStore(t)
	storeChain := func(length int) string {
		previous := ""
		leaf := ""
		for index := 0; index < length; index++ {
			id := fmt.Sprintf("resp_cap_%d_%d", length, index)
			mustStoreResponse(t, store, &responseapi.StoredResponse{ID: id, PreviousResponseID: previous})
			previous, leaf = id, id
		}
		return leaf
	}

	chain, err := store.GetConversationChain(context.Background(), storeChain(1000))
	if err != nil || len(chain) != 1000 {
		t.Fatalf("a chain of exactly the cap = %d entries, err = %v; want 1000 and no error", len(chain), err)
	}

	chain, err = store.GetConversationChain(context.Background(), storeChain(1001))
	if err == nil {
		t.Fatalf("a chain past the cap = %d entries, err = nil; want an error", len(chain))
	}
}
