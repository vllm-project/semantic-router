package extproc

import (
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
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

	cfg := newDefaultRouteConfig()
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

func TestRejectedReloadCandidatePreservesPublishedSessionStore(t *testing.T) {
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

	activeResources := newResourceScope()
	registerRouterSessionStore(activeResources, activeStoreSlot)
	oldRouter := (&routerComponents{
		cfg:                &config.RouterConfig{},
		resources:          activeResources,
		routerSessionStore: activeStoreSlot,
	}).buildRouter()
	candidateCfg := &config.RouterConfig{ConfigSource: config.ConfigSourceKubernetes}
	candidateResources := newResourceScope()
	registerRouterSessionStore(candidateResources, candidateStoreSlot)
	candidateRouter := (&routerComponents{
		cfg:                candidateCfg,
		resources:          candidateResources,
		routerSessionStore: candidateStoreSlot,
	}).buildRouter()
	server := &Server{
		service: NewRouterService(oldRouter),
	}
	t.Cleanup(func() {
		_ = server.service.Close()
	})

	buildReloadRouter = func(*config.RouterConfig, ...*binding.Pool) (*OpenAIRouter, error) {
		server.lifecycle.beginShutdown()
		return candidateRouter, nil
	}
	replaceReloadConfig = func(*config.RouterConfig) {
		t.Fatal("replaceReloadConfig() called for a rejected candidate")
	}

	err := server.reloadRouterFromConfig("kubernetes", "/unused/config.yaml", candidateCfg)
	if err == nil || !strings.Contains(err.Error(), errServerShuttingDown.Error()) {
		t.Fatalf("reloadRouterFromConfig() error = %v, want the candidate rejected at shutdown", err)
	}
	if got := server.service.GetRouter(); got != oldRouter {
		t.Fatalf("router changed for a rejected candidate: got %p, want %p", got, oldRouter)
	}
	if got := activeStore.closeCalls.Load(); got != 0 {
		t.Fatalf("active store close calls = %d, want 0", got)
	}
	if got := candidateStore.closeCalls.Load(); got != 1 {
		t.Fatalf("candidate store close calls = %d, want 1", got)
	}

	sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{
		SessionID:     "active-after-rejected-candidate",
		SelectedModel: "model-a",
		Timestamp:     time.Now(),
	})
	if got := activeStore.saveCalls.Load(); got != 1 {
		t.Fatalf("active store save calls = %d, want 1 after a rejected candidate", got)
	}
	if got := candidateStore.saveCalls.Load(); got != 0 {
		t.Fatalf("candidate store save calls = %d, want 0", got)
	}
}
