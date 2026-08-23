package extproc

import (
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

type trackingRouterSessionStateStore struct {
	loadCalls  atomic.Int32
	saveCalls  atomic.Int32
	closeCalls atomic.Int32
	closed     chan struct{}
	closeOnce  sync.Once
}

func (s *trackingRouterSessionStateStore) Load(string) (sessiontelemetry.RouterSessionSnapshot, bool, error) {
	s.loadCalls.Add(1)
	return sessiontelemetry.RouterSessionSnapshot{}, false, nil
}

func (s *trackingRouterSessionStateStore) Save(
	sessiontelemetry.RouterSessionSnapshot,
	time.Duration,
) error {
	s.saveCalls.Add(1)
	return nil
}

func (s *trackingRouterSessionStateStore) Close() error {
	s.closeCalls.Add(1)
	s.closeOnce.Do(func() {
		if s.closed != nil {
			close(s.closed)
		}
	})
	return nil
}

func TestBuildRouterComponentsReplayFailurePreservesPublishedSessionStore(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	activeStore := &trackingRouterSessionStateStore{}
	sessiontelemetry.SetRouterSessionStateStore(activeStore)
	t.Cleanup(func() {
		sessiontelemetry.SetRouterSessionStateStore(nil)
		sessiontelemetry.ResetRouterSessionMemoryForTesting()
	})

	candidateStore := &trackingRouterSessionStateStore{}
	originalStoreBuilder := newRouterSessionStateStore
	newRouterSessionStateStore = func(
		sessiontelemetry.RedisRouterSessionStoreConfig,
	) (sessiontelemetry.RouterSessionStateStore, error) {
		return candidateStore, nil
	}
	t.Cleanup(func() {
		newRouterSessionStateStore = originalStoreBuilder
	})

	cfg := newCoreSignalMappingGateConfig(t)
	cfg.RouterLearning.StateStore = config.RouterLearningStateStoreConfig{
		Backend: "redis",
		Redis: config.RouterLearningRedisStateStoreConfig{
			Address: "candidate.invalid:6379",
		},
	}
	cfg.RouterReplay = config.RouterReplayConfig{
		Enabled:      true,
		StoreBackend: "redis",
	}

	_, err := buildRouterComponentsWithDependencies(cfg, RuntimeDependencies{
		DispatchCapabilities: dispatchCapabilityRuntimeStub{},
		ResponseTerminals:    backendinvoker.NewLocalResponseTerminalStore(),
		ProtocolCodecs:       protocolcodec.NewBuiltinRegistry(),
	})
	if err == nil || !strings.Contains(err.Error(), "redis config required") {
		t.Fatalf("buildRouterComponents() error = %v, want replay initialization failure", err)
	}
	if got := activeStore.closeCalls.Load(); got != 0 {
		t.Fatalf("active store close calls = %d, want 0", got)
	}
	if got := candidateStore.closeCalls.Load(); got != 1 {
		t.Fatalf("candidate store close calls = %d, want 1", got)
	}

	sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{
		SessionID:     "active-after-replay-failure",
		SelectedModel: "model-a",
		Timestamp:     time.Now(),
	})
	if got := activeStore.saveCalls.Load(); got != 1 {
		t.Fatalf("active store save calls = %d, want 1 after candidate failure", got)
	}
	if got := candidateStore.saveCalls.Load(); got != 0 {
		t.Fatalf("candidate store save calls = %d, want 0", got)
	}
}
