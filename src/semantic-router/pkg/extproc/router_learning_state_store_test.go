package extproc

import (
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
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

func (s *trackingRouterSessionStateStore) SaveIfVersion(
	sessiontelemetry.RouterSessionSnapshot,
	uint64,
	time.Duration,
) (bool, error) {
	s.saveCalls.Add(1)
	return true, nil
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
	cfg.IntelligentRouting.Decisions[0].ModelRefs = []config.ModelRef{{Model: "model-a"}}

	_, err := buildRouterComponents(cfg)
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

func TestReloadWarmupFailurePreservesPublishedSessionStore(t *testing.T) {
	restoreReloadSeams := stubReloadSeams(t)
	t.Cleanup(restoreReloadSeams)
	sessiontelemetry.ResetRouterSessionMemoryForTesting()

	activeStore := &trackingRouterSessionStateStore{}
	candidateStore := &trackingRouterSessionStateStore{}
	activeStoreSlot := sessiontelemetry.NewRouterSessionStateStoreSlot(activeStore)
	candidateStoreSlot := sessiontelemetry.NewRouterSessionStateStoreSlot(candidateStore)
	sessiontelemetry.PublishRouterSessionStateStore(activeStoreSlot)
	t.Cleanup(func() {
		sessiontelemetry.SetRouterSessionStateStore(nil)
		sessiontelemetry.ResetRouterSessionMemoryForTesting()
	})

	oldRouter := &OpenAIRouter{
		Config:                  &config.RouterConfig{},
		routerSessionStateStore: activeStoreSlot,
	}
	candidateCfg := &config.RouterConfig{ConfigSource: config.ConfigSourceKubernetes}
	candidateRouter := &OpenAIRouter{
		Config:                  candidateCfg,
		routerSessionStateStore: candidateStoreSlot,
	}
	server := &Server{
		service: NewRouterService(oldRouter),
	}
	t.Cleanup(func() {
		_ = server.service.Close()
	})

	prepareReloadRuntime = func(*config.RouterConfig) (modelruntime.EmbeddingRuntimeState, error) {
		return modelruntime.EmbeddingRuntimeState{AnyReady: true}, nil
	}
	buildReloadRouter = func(*config.RouterConfig) (*OpenAIRouter, error) {
		return candidateRouter, nil
	}
	warmupReloadRouter = func(*OpenAIRouter, modelruntime.EmbeddingRuntimeState) error {
		return errors.New("injected warmup failure")
	}
	replaceReloadConfig = func(*config.RouterConfig) {
		t.Fatal("replaceReloadConfig() called after warmup failure")
	}

	err := server.reloadRouterFromConfig("kubernetes", "/unused/config.yaml", candidateCfg)
	if err == nil || !strings.Contains(err.Error(), "injected warmup failure") {
		t.Fatalf("reloadRouterFromConfig() error = %v, want injected warmup failure", err)
	}
	if got := server.service.GetRouter(); got != oldRouter {
		t.Fatalf("router changed on warmup failure: got %p, want %p", got, oldRouter)
	}
	if got := activeStore.closeCalls.Load(); got != 0 {
		t.Fatalf("active store close calls = %d, want 0", got)
	}
	if got := candidateStore.closeCalls.Load(); got != 1 {
		t.Fatalf("candidate store close calls = %d, want 1", got)
	}

	sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{
		SessionID:     "active-after-warmup-failure",
		SelectedModel: "model-a",
		Timestamp:     time.Now(),
	})
	if got := activeStore.saveCalls.Load(); got != 1 {
		t.Fatalf("active store save calls = %d, want 1 after warmup failure", got)
	}
	if got := candidateStore.saveCalls.Load(); got != 0 {
		t.Fatalf("candidate store save calls = %d, want 0", got)
	}
}
