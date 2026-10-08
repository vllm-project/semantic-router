package classification

import (
	"context"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// recordingBundle records joins in the order the dispatchers make them.
type recordingBundle struct {
	mu     sync.Mutex
	events *[]string
}

func (b *recordingBundle) JoinAsking(ctx context.Context, _ ...string) (context.Context, func()) {
	b.mu.Lock()
	*b.events = append(*b.events, "join")
	b.mu.Unlock()
	return ctx, func() {}
}

func TestEveryModelBackedSignalJoinsTheBundleBeforeAnyStarts(t *testing.T) {
	var events []string
	bundle := &recordingBundle{events: &events}
	signals := []string{config.SignalTypeDomain, config.SignalTypeJailbreak, config.SignalTypePII, config.SignalTypeFactCheck, config.SignalTypeModality}
	used, ready := map[string]bool{}, map[string]bool{}
	var dispatchers []signalDispatch
	for _, signal := range signals {
		used[signal+":rule"], ready[signal] = true, true
		dispatchers = append(dispatchers, signalDispatch{signalType: signal, name: signal, evaluate: func(context.Context) {
			bundle.mu.Lock()
			events = append(events, "evaluate")
			bundle.mu.Unlock()
		}})
	}
	var wg sync.WaitGroup
	runSignalDispatchers(context.Background(), dispatchers, used, ready, bundle, func(string) []string { return nil }, &wg)
	wg.Wait()
	if len(events) != 2*len(signals) {
		t.Fatalf("events = %v", events)
	}
	for i, event := range events {
		if want := map[bool]string{true: "join", false: "evaluate"}[i < len(signals)]; event != want {
			t.Fatalf("a signal started before every model-backed signal joined the bundle: %v", events)
		}
	}
}
