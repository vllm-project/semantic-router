package extproc

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

type reloadMemoryStore struct{}

func (reloadMemoryStore) Store(_ context.Context, _ *memory.Memory) error { return nil }
func (reloadMemoryStore) Retrieve(_ context.Context, _ memory.RetrieveOptions) ([]*memory.RetrieveResult, error) {
	return nil, nil
}

func (reloadMemoryStore) Get(_ context.Context, _ string) (*memory.Memory, error)    { return nil, nil }
func (reloadMemoryStore) Update(_ context.Context, _ string, _ *memory.Memory) error { return nil }
func (reloadMemoryStore) List(_ context.Context, _ memory.ListOptions) (*memory.ListResult, error) {
	return nil, nil
}
func (reloadMemoryStore) Forget(_ context.Context, _ string) error                    { return nil }
func (reloadMemoryStore) ForgetByScope(_ context.Context, _ memory.MemoryScope) error { return nil }
func (reloadMemoryStore) IsEnabled() bool                                             { return true }
func (reloadMemoryStore) CheckConnection(_ context.Context) error                     { return nil }
func (reloadMemoryStore) Close() error                                                { return nil }

// reloadRendezvousCache is a CacheBackend stub whose LookupSimilarWithThreshold
// signals entry on a channel and then blocks until released, so a test can
// land a "request" goroutine inside the call before triggering a concurrent
// close.
type reloadRendezvousCache struct {
	entered chan struct{}
	release chan struct{}
	closed  atomic.Bool
}

func (c *reloadRendezvousCache) IsEnabled() bool                                   { return true }
func (c *reloadRendezvousCache) CheckConnection() error                            { return nil }
func (c *reloadRendezvousCache) GetStats() cache.CacheStats                        { return cache.CacheStats{} }
func (c *reloadRendezvousCache) AddEntry(_, _, _ string, _, _ []byte, _ int) error { return nil }

func (c *reloadRendezvousCache) LookupSimilarWithThreshold(_, _ string, _ float32) (cache.LookupResult, error) {
	close(c.entered)
	<-c.release
	if c.closed.Load() {
		return cache.LookupResult{}, errors.New("cache closed while request was in flight")
	}
	return cache.LookupResult{}, nil
}

func (c *reloadRendezvousCache) Close() error {
	c.closed.Store(true)
	return nil
}

// reloadRendezvousSelector is a selection.Selector stub whose Select signals
// entry on a channel and then blocks until released, mirroring
// reloadRendezvousCache but for the classification/model-selection path.
type reloadRendezvousSelector struct {
	method  selection.SelectionMethod
	entered chan struct{}
	release chan struct{}
	closed  atomic.Bool
}

func (s *reloadRendezvousSelector) Select(context.Context, *selection.SelectionContext) (*selection.SelectionResult, error) {
	close(s.entered)
	<-s.release
	if s.closed.Load() {
		return nil, errors.New("selector closed while classification request was in flight")
	}
	return &selection.SelectionResult{}, nil
}

func (s *reloadRendezvousSelector) Method() selection.SelectionMethod { return s.method }

func (s *reloadRendezvousSelector) UpdateFeedback(context.Context, *selection.Feedback) error {
	return nil
}

func (s *reloadRendezvousSelector) Tier() selection.AlgorithmTier { return selection.TierSupported }

func (s *reloadRendezvousSelector) ExternalDependencies() []selection.Dependency { return nil }

func (s *reloadRendezvousSelector) Close() error {
	s.closed.Store(true)
	return nil
}

// TestReloadDrainsInFlightRequestBeforeClosingOldRouterCache exercises the
// real reload sequence — RouterService.Swap followed by RouterService.Retire
// (what reloadRouterFromConfig now calls, replacing the old bare
// Swap+oldRouter.Close()) — while a "request" goroutine holds a lease on the
// old router's Cache. It asserts Retire blocks until that lease releases,
// leaves the cache open the whole time it's in flight, and only closes it
// once the request has safely finished.
//
// This test used to assert the opposite (that the in-flight call observed an
// error from a concurrently-closed cache) back when Close() and reload had no
// lease coordination. It flips here to the safe-drain behavior now that
// RouterService leases every call and Retire waits on the lease before
// closing.
func TestReloadDrainsInFlightRequestBeforeClosingOldRouterCache(t *testing.T) {
	fakeCache := &reloadRendezvousCache{
		entered: make(chan struct{}),
		release: make(chan struct{}),
	}
	oldRouter := &OpenAIRouter{Cache: fakeCache}
	rs := NewRouterService(oldRouter)

	// Acquire a lease the same way RouterService.Process does for a real
	// request, so the in-flight call below is protected exactly as it would
	// be in production.
	oldLease := rs.current.Load()
	if !oldLease.acquire() {
		t.Fatal("acquire() = false, want true before any reload")
	}

	resultErr := make(chan error, 1)
	go func() {
		defer oldLease.release()
		_, err := oldRouter.Cache.LookupSimilarWithThreshold("model", "query", 0)
		resultErr <- err
	}()

	<-fakeCache.entered // wait until the "request" is inside the blocking call

	newRouter := &OpenAIRouter{Cache: &reloadRendezvousCache{}}
	retireDone := make(chan error, 1)
	go func() {
		swappedLease := rs.Swap(newRouter)
		retireDone <- rs.Retire(swappedLease, 2*time.Second)
	}()

	// Retire must block on the still-held lease, so the cache must not be
	// closed yet even though the swap and retire have both started.
	select {
	case <-retireDone:
		t.Fatal("Retire() returned before the in-flight request released its lease")
	case <-time.After(50 * time.Millisecond):
	}
	if fakeCache.closed.Load() {
		t.Fatal("cache was closed while a request was still in flight")
	}

	close(fakeCache.release) // let the "request" finish, releasing its lease

	select {
	case err := <-resultErr:
		if err != nil {
			t.Fatalf("in-flight request observed an error even though it finished before the reload closed its resources: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("in-flight request never returned")
	}

	select {
	case err := <-retireDone:
		if err != nil {
			t.Fatalf("Retire() error = %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Retire() did not return after the in-flight request released its lease")
	}
	if !fakeCache.closed.Load() {
		t.Fatal("Retire() did not close the old router's cache after the in-flight request finished")
	}
}

// TestReloadDrainsInFlightClassificationRequestBeforeClosingOldRouterModelSelector
// is the classification-path sibling of
// TestReloadDrainsInFlightRequestBeforeClosingOldRouterCache: routerLease
// doesn't distinguish which resource a call is using, but ModelSelector's
// Close fans out to per-recipe registries and native ML handles, so it's
// worth exercising directly rather than only inferring safety from Cache.
func TestReloadDrainsInFlightClassificationRequestBeforeClosingOldRouterModelSelector(t *testing.T) {
	fakeSelector := &reloadRendezvousSelector{
		method:  selection.MethodKNN,
		entered: make(chan struct{}),
		release: make(chan struct{}),
	}
	oldRegistry := selection.NewRegistry()
	oldRegistry.Register(selection.MethodKNN, fakeSelector)
	oldRouter := &OpenAIRouter{ModelSelector: oldRegistry}
	rs := NewRouterService(oldRouter)

	oldLease := rs.current.Load()
	if !oldLease.acquire() {
		t.Fatal("acquire() = false, want true before any reload")
	}

	resultErr := make(chan error, 1)
	go func() {
		defer oldLease.release()
		selector, ok := oldRouter.ModelSelector.Get(selection.MethodKNN)
		if !ok {
			resultErr <- errors.New("selector not registered")
			return
		}
		_, err := selector.Select(context.Background(), &selection.SelectionContext{})
		resultErr <- err
	}()

	<-fakeSelector.entered // wait until the "request" is inside the blocking call

	newRouter := &OpenAIRouter{ModelSelector: selection.NewRegistry()}
	retireDone := make(chan error, 1)
	go func() {
		swappedLease := rs.Swap(newRouter)
		retireDone <- rs.Retire(swappedLease, 2*time.Second)
	}()

	select {
	case <-retireDone:
		t.Fatal("Retire() returned before the in-flight classification request released its lease")
	case <-time.After(50 * time.Millisecond):
	}
	if fakeSelector.closed.Load() {
		t.Fatal("model selector was closed while a classification request was still in flight")
	}

	close(fakeSelector.release) // let the "request" finish, releasing its lease

	select {
	case err := <-resultErr:
		if err != nil {
			t.Fatalf("in-flight classification request observed an error even though it finished before the reload closed its resources: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("in-flight classification request never returned")
	}

	select {
	case err := <-retireDone:
		if err != nil {
			t.Fatalf("Retire() error = %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Retire() did not return after the in-flight classification request released its lease")
	}
	if !fakeSelector.closed.Load() {
		t.Fatal("Retire() did not close the old router's model selector after the in-flight classification request finished")
	}
}

// TestReloadPreservesAcknowledgedResponseUntilInFlightRequestReleases extends
// the drain guarantee from a store surviving in-flight requests to the data a
// request already wrote to it being readable for as long as that request is
// still in flight, even while a reload retires the router that owns it.
func TestReloadPreservesAcknowledgedResponseUntilInFlightRequestReleases(t *testing.T) {
	store, err := responsestore.NewMemoryStore(responsestore.StoreConfig{
		Enabled:    true,
		TTLSeconds: 60,
	})
	if err != nil {
		t.Fatalf("NewMemoryStore() error = %v", err)
	}
	filter := NewResponseAPIFilter(store)
	oldRouter := &OpenAIRouter{ResponseAPIFilter: filter}
	rs := NewRouterService(oldRouter)

	oldLease := rs.current.Load()
	if !oldLease.acquire() {
		t.Fatal("acquire() = false, want true before any reload")
	}

	// Acknowledge a response the way a real request would: store it, then
	// keep the request in flight a while longer (translating or streaming
	// the rest of the reply) before it actually returns.
	stored := &responseapi.StoredResponse{ID: "resp_ack_1", Object: "response", Status: "completed"}
	if err := store.StoreResponse(context.Background(), stored); err != nil {
		t.Fatalf("StoreResponse() error = %v", err)
	}

	holdRelease := make(chan struct{})
	requestDone := make(chan error, 1)
	go func() {
		defer oldLease.release()
		<-holdRelease
		got, err := store.GetResponse(context.Background(), stored.ID)
		if err != nil {
			requestDone <- err
			return
		}
		if got.ID != stored.ID {
			requestDone <- fmt.Errorf("GetResponse() ID = %q, want %q", got.ID, stored.ID)
			return
		}
		requestDone <- nil
	}()

	newRouter := &OpenAIRouter{ResponseAPIFilter: NewResponseAPIFilter(nil)}
	retireDone := make(chan error, 1)
	go func() {
		swappedLease := rs.Swap(newRouter)
		retireDone <- rs.Retire(swappedLease, 2*time.Second)
	}()

	// Retire must not have closed the store yet, so the response acknowledged
	// before the reload started must still be readable.
	select {
	case <-retireDone:
		t.Fatal("Retire() returned before the in-flight request released its lease")
	case <-time.After(50 * time.Millisecond):
	}
	if _, err := store.GetResponse(context.Background(), stored.ID); err != nil {
		t.Fatalf("acknowledged response was lost while its request was still in flight: %v", err)
	}

	close(holdRelease) // let the "request" read its own response and finish

	select {
	case err := <-requestDone:
		if err != nil {
			t.Fatalf("in-flight request could not read back its own acknowledged response: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("in-flight request never returned")
	}

	select {
	case err := <-retireDone:
		if err != nil {
			t.Fatalf("Retire() error = %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Retire() did not return after the in-flight request released its lease")
	}
}

// TestReloadPublishesRuntimeStateBeforeDrainingOldRouter asserts a reload
// updates the control plane as soon as the new router starts serving,
// instead of after the old router has drained. Swap makes the new router
// live immediately, so publishing behind Retire — which blocks for up to
// defaultRouterDrainTimeout — would leave the runtime registry describing
// the old router while the data plane already runs the new one.
func TestReloadPublishesRuntimeStateBeforeDrainingOldRouter(t *testing.T) {
	restoreReloadSeams := stubReloadSeams(t)
	defer restoreReloadSeams()

	newRouter := &OpenAIRouter{MemoryStore: reloadMemoryStore{}}
	ensureReloadConfigModels = func(*config.RouterConfig) error { return nil }
	prepareReloadRuntime = func(*config.RouterConfig) (modelruntime.EmbeddingRuntimeState, error) {
		return modelruntime.EmbeddingRuntimeState{}, nil
	}
	buildReloadRouter = func(*config.RouterConfig) (*OpenAIRouter, error) { return newRouter, nil }
	warmupReloadRouter = func(*OpenAIRouter, modelruntime.EmbeddingRuntimeState) error { return nil }

	registry := routerruntime.NewRegistry(&config.RouterConfig{})
	server := &Server{
		service: NewRouterService(&OpenAIRouter{}),
		runtime: registry,
	}

	// Hold a lease on the old router so the reload's Retire has to block on
	// the drain, making the publish-vs-drain ordering observable.
	oldLease := server.service.current.Load()
	if !oldLease.acquire() {
		t.Fatal("acquire() = false, want true before any reload")
	}

	reloadDone := make(chan error, 1)
	go func() {
		reloadDone <- server.reloadRouterFromConfig("file", "config.yaml", &config.RouterConfig{})
	}()

	deadline := time.After(5 * time.Second)
	for registry.MemoryStore() == nil {
		select {
		case err := <-reloadDone:
			t.Fatalf("reload returned before publishing runtime state (error = %v)", err)
		case <-deadline:
			t.Fatal("reload never published runtime state; it is blocked draining the old router first")
		default:
			time.Sleep(time.Millisecond)
		}
	}

	// The publish landed while the reload was still draining — which is the
	// whole point, so confirm it really is still blocked.
	select {
	case err := <-reloadDone:
		t.Fatalf("reload finished without draining the held lease (error = %v)", err)
	default:
	}

	oldLease.release()
	select {
	case err := <-reloadDone:
		if err != nil {
			t.Fatalf("reloadRouterFromConfig() error = %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("reload did not finish after the in-flight lease released")
	}
}

func TestReloadRouterFromConfigSkipsReplaceForKubernetesSource(t *testing.T) {
	restoreReloadSeams := stubReloadSeams(t)
	defer restoreReloadSeams()

	candidateCfg := &config.RouterConfig{
		ConfigSource:  config.ConfigSourceKubernetes,
		BackendModels: config.BackendModels{DefaultModel: "new-model"},
	}
	oldRouter := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "old-model"},
	}}
	server := &Server{
		configPath: "/unused/config.yaml",
		service:    NewRouterService(oldRouter),
	}

	ensureReloadConfigModels = func(cfg *config.RouterConfig) error {
		t.Fatalf("ensureReloadConfigModels() should not run during kubernetes watcher reload")
		return nil
	}
	prepareReloadRuntime = func(cfg *config.RouterConfig) (modelruntime.EmbeddingRuntimeState, error) {
		if cfg != candidateCfg {
			t.Fatalf("prepareReloadRuntime() cfg = %p, want %p", cfg, candidateCfg)
		}
		return modelruntime.EmbeddingRuntimeState{AnyReady: true}, nil
	}

	buildCalls := 0
	buildReloadRouter = func(cfg *config.RouterConfig) (*OpenAIRouter, error) {
		buildCalls++
		if cfg != candidateCfg {
			t.Fatalf("buildReloadRouter() cfg = %p, want %p", cfg, candidateCfg)
		}
		return &OpenAIRouter{Config: cfg}, nil
	}
	warmupCalls := 0
	warmupReloadRouter = func(router *OpenAIRouter, state modelruntime.EmbeddingRuntimeState) error {
		warmupCalls++
		if router == nil || router.Config != candidateCfg {
			t.Fatalf("warmupReloadRouter() router config mismatch")
		}
		if !state.AnyReady {
			t.Fatalf("warmupReloadRouter() state = %+v, want AnyReady=true", state)
		}
		return nil
	}

	replaceCalls := 0
	replaceReloadConfig = func(cfg *config.RouterConfig) {
		replaceCalls++
	}

	if err := server.reloadRouterFromConfig("kubernetes", server.configPath, candidateCfg); err != nil {
		t.Fatalf("reloadRouterFromConfig() error = %v", err)
	}

	if buildCalls != 1 {
		t.Fatalf("buildReloadRouter() calls = %d, want 1", buildCalls)
	}
	if warmupCalls != 1 {
		t.Fatalf("warmupReloadRouter() calls = %d, want 1", warmupCalls)
	}
	if replaceCalls != 0 {
		t.Fatalf("replaceReloadConfig() calls = %d, want 0", replaceCalls)
	}
	if got := server.service.GetRouter(); got == oldRouter || got.Config != candidateCfg {
		t.Fatalf("router swap did not install kubernetes config")
	}
}

func TestReloadRouterFromConfigDoesNotSwapWhenRuntimePreparationFails(t *testing.T) {
	restoreReloadSeams := stubReloadSeams(t)
	defer restoreReloadSeams()

	candidateCfg := &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "candidate"},
	}
	oldRouter := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "old"},
	}}
	server := &Server{
		configPath: "/tmp/router-config.yaml",
		service:    NewRouterService(oldRouter),
	}

	ensureReloadConfigModels = func(cfg *config.RouterConfig) error { return nil }
	prepareReloadRuntime = func(cfg *config.RouterConfig) (modelruntime.EmbeddingRuntimeState, error) {
		return modelruntime.EmbeddingRuntimeState{}, errors.New("modality init failed")
	}
	buildReloadRouter = func(cfg *config.RouterConfig) (*OpenAIRouter, error) {
		t.Fatalf("buildReloadRouter() should not be called when runtime prep fails")
		return nil, nil
	}
	warmupReloadRouter = func(router *OpenAIRouter, state modelruntime.EmbeddingRuntimeState) error {
		t.Fatalf("warmupReloadRouter() should not be called when runtime prep fails")
		return nil
	}
	replaceReloadConfig = func(cfg *config.RouterConfig) {
		t.Fatalf("replaceReloadConfig() should not be called when runtime prep fails")
	}

	err := server.reloadRouterFromConfig("file", server.configPath, candidateCfg)
	if err == nil {
		t.Fatal("reloadRouterFromConfig() error = nil, want failure")
	}
	if got := err.Error(); got != "runtime dependency init failed: modality init failed" {
		t.Fatalf("reloadRouterFromConfig() error = %q", got)
	}
	if got := server.service.GetRouter(); got != oldRouter {
		t.Fatalf("router changed on runtime prep failure")
	}
}

func TestReloadRouterFromConfigPublishesRuntimeRegistryAfterSwap(t *testing.T) {
	restoreReloadSeams := stubReloadSeams(t)
	defer restoreReloadSeams()

	globalCfg := &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "global"},
	}
	restoreGlobalConfig := replaceExtProcGlobalConfigForTest(globalCfg)
	defer restoreGlobalConfig()

	oldCfg := &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "old"},
	}
	newCfg := &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "new"},
	}
	oldService := &services.ClassificationService{}
	newService := &services.ClassificationService{}
	newModelSelector := selection.NewRegistry()
	registry := routerruntime.NewRegistry(oldCfg)
	registry.PublishRouterRuntime(oldCfg, oldService, nil)

	server := &Server{
		configPath: "/tmp/router-config.yaml",
		service: NewRouterService(&OpenAIRouter{
			Config:                oldCfg,
			ClassificationService: oldService,
		}),
		runtime: registry,
	}

	ensureReloadConfigModels = func(cfg *config.RouterConfig) error { return nil }
	prepareReloadRuntime = func(cfg *config.RouterConfig) (modelruntime.EmbeddingRuntimeState, error) {
		return modelruntime.EmbeddingRuntimeState{AnyReady: true, ToolsReady: true}, nil
	}
	buildReloadRouter = func(cfg *config.RouterConfig) (*OpenAIRouter, error) {
		return &OpenAIRouter{
			Config:                newCfg,
			ClassificationService: newService,
			MemoryStore:           reloadMemoryStore{},
			ModelSelector:         newModelSelector,
		}, nil
	}
	warmupReloadRouter = func(router *OpenAIRouter, state modelruntime.EmbeddingRuntimeState) error { return nil }
	replaceReloadConfig = func(cfg *config.RouterConfig) {
		t.Fatalf("replaceReloadConfig() should not run for registry-backed reload")
	}

	if err := server.reloadRouterFromConfig("file", server.configPath, newCfg); err != nil {
		t.Fatalf("reloadRouterFromConfig() error = %v", err)
	}

	if got := registry.CurrentConfig(); got != newCfg {
		t.Fatalf("registry.CurrentConfig() = %p, want %p", got, newCfg)
	}
	if got := config.Get(); got != globalCfg {
		t.Fatalf("config.Get() = %p, want unchanged global cfg %p", got, globalCfg)
	}
	if got := registry.ClassificationService(); got != newService {
		t.Fatalf("registry.ClassificationService() = %p, want %p", got, newService)
	}
	if got := registry.MemoryStore(); got == nil {
		t.Fatal("registry.MemoryStore() = nil, want populated store")
	}
	if got := registry.ModelSelector(); got != newModelSelector {
		t.Fatalf("registry.ModelSelector() = %p, want %p", got, newModelSelector)
	}
	if got := server.service.GetRouter(); got == nil || got.RuntimeRegistry != registry {
		t.Fatalf("reloaded router did not retain runtime registry")
	}
}
