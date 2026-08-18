package extproc

import (
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestOpenAIRouterCloseClosesSharedReplayStoreOnce(t *testing.T) {
	storage := &countingCloseStore{Storage: store.NewMemoryStore(10, 0)}
	primary := routerreplay.NewRecorder(storage)
	router := &OpenAIRouter{
		ReplayRecorder:    primary,
		ReplayStoreShared: true,
		ReplayRecorders: map[string]*routerreplay.Recorder{
			"route-a": primary,
			"route-b": routerreplay.NewRecorder(storage),
		},
	}

	if err := router.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if got := storage.closeCalls.Load(); got != 1 {
		t.Fatalf("shared replay store close calls = %d, want 1", got)
	}
}

func TestOpenAIRouterCloseClosesEachIsolatedReplayStore(t *testing.T) {
	firstStore := &countingCloseStore{Storage: store.NewMemoryStore(10, 0)}
	secondStore := &countingCloseStore{Storage: store.NewMemoryStore(10, 0)}
	first := routerreplay.NewRecorder(firstStore)
	second := routerreplay.NewRecorder(secondStore)
	router := &OpenAIRouter{
		ReplayRecorder: first,
		ReplayRecorders: map[string]*routerreplay.Recorder{
			"route-a": first,
			"route-b": second,
		},
	}

	if err := router.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if firstStore.closeCalls.Load() != 1 || secondStore.closeCalls.Load() != 1 {
		t.Fatalf("isolated close calls = %d / %d, want 1 / 1", firstStore.closeCalls.Load(), secondStore.closeCalls.Load())
	}
}

type countingCloseStore struct {
	store.Storage
	closeCalls atomic.Int32
}

func (s *countingCloseStore) Close() error {
	s.closeCalls.Add(1)
	return s.Storage.Close()
}
