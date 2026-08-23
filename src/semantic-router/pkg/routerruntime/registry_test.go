package routerruntime

import (
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestRegistryPublishRouterRuntime(t *testing.T) {
	initial := &config.RouterConfig{}
	next := &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "next"},
	}

	registry := NewRegistry(initial)
	if registry.CurrentConfig() != initial {
		t.Fatalf("CurrentConfig() = %p, want %p", registry.CurrentConfig(), initial)
	}

	registry.PublishRouterRuntime(next, nil, nil)

	if registry.CurrentConfig() != next {
		t.Fatalf("CurrentConfig() = %p, want %p", registry.CurrentConfig(), next)
	}
}

func TestRegistryRejectsPartialClassifierRefresh(t *testing.T) {
	initial := &config.RouterConfig{}
	service := services.NewClassificationService(
		&classification.Classifier{Config: initial},
		initial,
	)
	registry := NewRegistry(initial)
	registry.PublishRouterRuntime(initial, service, nil)

	registry.RefreshRuntimeConfig(nil)

	if registry.CurrentConfig() != initial {
		t.Fatal("registry published config after classifier rebuild failure")
	}
	if service.GetConfig() != initial {
		t.Fatal("classification service published partial config")
	}
}

func TestRegistryPublishesModelSelector(t *testing.T) {
	registry := NewRegistry(&config.RouterConfig{})
	selectorRegistry := selection.NewRegistry()

	registry.SetModelSelector(selectorRegistry)

	if got := registry.ModelSelector(); got != selectorRegistry {
		t.Fatalf("ModelSelector() = %p, want %p", got, selectorRegistry)
	}

	registry.SetModelSelector(nil)
	if got := registry.ModelSelector(); got != nil {
		t.Fatalf("ModelSelector() = %p, want nil after clear", got)
	}
}

func TestRegistryPublishesRouterManagementSnapshotAtomically(t *testing.T) {
	initial := &config.RouterConfig{BackendModels: config.BackendModels{DefaultModel: "initial"}}
	next := &config.RouterConfig{BackendModels: config.BackendModels{DefaultModel: "next"}}
	initialRuntime := &leasedLearningRuntime{}
	nextRuntime := &leasedLearningRuntime{}
	registry := NewRegistry(initial)
	registry.PublishRouterRuntimeSnapshot(RouterRuntimeSnapshot{
		Config:          initial,
		LearningRuntime: initialRuntime,
	})

	runtime, release := registry.AcquireLearningRuntime()
	if runtime != initialRuntime {
		t.Fatalf("initial runtime = %T", runtime)
	}
	registry.PublishRouterRuntimeSnapshot(RouterRuntimeSnapshot{
		Config:          next,
		LearningRuntime: nextRuntime,
	})
	if registry.CurrentConfig() != next {
		t.Fatal("new config was not published with the management snapshot")
	}
	current, currentRelease := registry.AcquireLearningRuntime()
	if current != nextRuntime {
		t.Fatalf("current runtime = %T, want next", current)
	}
	currentRelease()
	if initialRuntime.released() {
		t.Fatal("initial management lease released before its handler completed")
	}
	release()
	if !initialRuntime.released() {
		t.Fatal("initial management lease did not release")
	}
}

func TestRegistryCacheLeaseUsesRouterGeneration(t *testing.T) {
	runtime := &leasedLearningRuntime{}
	registry := NewRegistry(nil)
	registry.PublishRouterRuntimeSnapshot(RouterRuntimeSnapshot{LearningRuntime: runtime})

	_, release := registry.AcquireResponseCache()
	retired := make(chan struct{})
	go func() {
		runtime.retireAndWait()
		close(retired)
	}()
	select {
	case <-retired:
		t.Fatal("router generation retired while cache handler held a lease")
	case <-time.After(25 * time.Millisecond):
	}
	release()
	select {
	case <-retired:
	case <-time.After(time.Second):
		t.Fatal("router generation did not retire after cache handler released")
	}
}

type leasedLearningRuntime struct {
	mu       sync.Mutex
	refs     sync.WaitGroup
	retired  bool
	releases int
}

func (runtime *leasedLearningRuntime) AcquireLease() (func(), bool) {
	runtime.mu.Lock()
	defer runtime.mu.Unlock()
	if runtime.retired {
		return nil, false
	}
	runtime.refs.Add(1)
	return func() {
		runtime.mu.Lock()
		runtime.releases++
		runtime.mu.Unlock()
		runtime.refs.Done()
	}, true
}

func (runtime *leasedLearningRuntime) retireAndWait() {
	runtime.mu.Lock()
	runtime.retired = true
	runtime.mu.Unlock()
	runtime.refs.Wait()
}

func (runtime *leasedLearningRuntime) released() bool {
	runtime.mu.Lock()
	defer runtime.mu.Unlock()
	return runtime.releases > 0
}
