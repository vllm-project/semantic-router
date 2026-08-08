package extproc

import (
	"context"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

// closeTrackingCache is a minimal CacheBackend stub that records how many
// times Close() was invoked.
type closeTrackingCache struct {
	mu         sync.Mutex
	closed     bool
	closeCalls int
}

func (c *closeTrackingCache) IsEnabled() bool                                         { return true }
func (c *closeTrackingCache) CheckConnection() error                                  { return nil }
func (c *closeTrackingCache) LastSimilarity() float32                                 { return 0 }
func (c *closeTrackingCache) GetStats() cache.CacheStats                              { return cache.CacheStats{} }
func (c *closeTrackingCache) AddPendingRequest(_, _, _ string, _ []byte, _ int) error { return nil }
func (c *closeTrackingCache) UpdateWithResponse(_ string, _ []byte, _ int) error      { return nil }
func (c *closeTrackingCache) AddEntry(_, _, _ string, _, _ []byte, _ int) error       { return nil }
func (c *closeTrackingCache) FindSimilar(_, _ string) ([]byte, bool, error)           { return nil, false, nil }

func (c *closeTrackingCache) FindSimilarWithThreshold(_, _ string, _ float32) ([]byte, bool, error) {
	return nil, false, nil
}

func (c *closeTrackingCache) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
	c.closeCalls++
	return nil
}

func (c *closeTrackingCache) closeCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.closeCalls
}

// closeTrackingMemoryStore is a minimal memory.Store stub that records
// whether Close() was ever invoked.
type closeTrackingMemoryStore struct {
	closed bool
}

func (m *closeTrackingMemoryStore) Store(context.Context, *memory.Memory) error { return nil }
func (m *closeTrackingMemoryStore) Retrieve(context.Context, memory.RetrieveOptions) ([]*memory.RetrieveResult, error) {
	return nil, nil
}

func (m *closeTrackingMemoryStore) Get(context.Context, string) (*memory.Memory, error) {
	return nil, nil
}

func (m *closeTrackingMemoryStore) Update(context.Context, string, *memory.Memory) error { return nil }

func (m *closeTrackingMemoryStore) List(context.Context, memory.ListOptions) (*memory.ListResult, error) {
	return nil, nil
}
func (m *closeTrackingMemoryStore) Forget(context.Context, string) error { return nil }
func (m *closeTrackingMemoryStore) ForgetByScope(context.Context, memory.MemoryScope) error {
	return nil
}
func (m *closeTrackingMemoryStore) IsEnabled() bool                       { return true }
func (m *closeTrackingMemoryStore) CheckConnection(context.Context) error { return nil }

func (m *closeTrackingMemoryStore) Close() error {
	m.closed = true
	return nil
}

// closeTrackingReplayStorage is a minimal replay store.Storage stub that
// records whether Close() was ever invoked.
type closeTrackingReplayStorage struct {
	closed bool
}

func (s *closeTrackingReplayStorage) Add(context.Context, store.Record) (string, error) {
	return "", nil
}

func (s *closeTrackingReplayStorage) UpdateStatus(context.Context, string, int, bool, bool) error {
	return nil
}

func (s *closeTrackingReplayStorage) AttachRequest(context.Context, string, string, bool) error {
	return nil
}

func (s *closeTrackingReplayStorage) AttachResponse(context.Context, string, string, bool) error {
	return nil
}

func (s *closeTrackingReplayStorage) AppendOutcome(context.Context, string, store.Outcome) error {
	return nil
}

func (s *closeTrackingReplayStorage) Get(context.Context, string) (store.Record, bool, error) {
	return store.Record{}, false, nil
}

func (s *closeTrackingReplayStorage) List(context.Context) ([]store.Record, error) { return nil, nil }

func (s *closeTrackingReplayStorage) UpdateHallucinationStatus(context.Context, string, bool, float32, []string, []store.HallucinationSpan) error {
	return nil
}

func (s *closeTrackingReplayStorage) UpdateUsageCost(context.Context, string, store.UsageCost) error {
	return nil
}

func (s *closeTrackingReplayStorage) UpdateToolTrace(context.Context, string, store.ToolTrace) error {
	return nil
}

func (s *closeTrackingReplayStorage) Close() error {
	s.closed = true
	return nil
}

// TestOpenAIRouterCloseClosesOwnedResources asserts OpenAIRouter.Close()
// closes every owned resource, not just lookupTableCancel. This router is
// assembled by hand rather than by buildRouterComponents, so it exercises
// Close's field-by-field fallback path (no generation to delegate to).
func TestOpenAIRouterCloseClosesOwnedResources(t *testing.T) {
	fakeCache := &closeTrackingCache{}
	fakeMemStore := &closeTrackingMemoryStore{}
	fakeStorage := &closeTrackingReplayStorage{}
	recorder := routerreplay.NewRecorder(fakeStorage)

	router := &OpenAIRouter{
		Cache:          fakeCache,
		MemoryStore:    fakeMemStore,
		ReplayRecorder: recorder,
	}

	if err := router.Close(); err != nil {
		t.Fatalf("router.Close() error = %v", err)
	}

	if !fakeCache.closed {
		t.Error("OpenAIRouter.Close() did not close Cache")
	}
	if !fakeMemStore.closed {
		t.Error("OpenAIRouter.Close() did not close MemoryStore")
	}
	if !fakeStorage.closed {
		t.Error("OpenAIRouter.Close() did not close ReplayRecorder")
	}
}

// TestOpenAIRouterCloseIsIdempotent asserts a router closes its resources
// exactly once no matter how many times, or from how many goroutines, Close
// is called. A router is reachable from more than one shutdown path — a
// reload retiring the lease it replaced, and process shutdown retiring
// whatever lease is current — and not every underlying resource tolerates a
// second Close (a gRPC ClientConn, an MCP client).
func TestOpenAIRouterCloseIsIdempotent(t *testing.T) {
	fakeCache := &closeTrackingCache{}
	router := &OpenAIRouter{Cache: fakeCache}

	const closers = 8
	var wg sync.WaitGroup
	wg.Add(closers)
	errs := make([]error, closers)
	for i := 0; i < closers; i++ {
		go func() {
			defer wg.Done()
			errs[i] = router.Close()
		}()
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("concurrent Close() #%d error = %v", i, err)
		}
	}
	if got := fakeCache.closeCount(); got != 1 {
		t.Fatalf("Cache.Close() called %d times, want exactly 1", got)
	}
}

// TestOpenAIRouterCloseDelegatesToOwningGeneration asserts a router built by
// buildRouterComponents tears down through its generation instead of the
// field-by-field fallback. That is what keeps a single list of what a router
// owns: if Close also walked the fields, every new resource would have to be
// registered in two places and could silently drift out of sync.
func TestOpenAIRouterCloseDelegatesToOwningGeneration(t *testing.T) {
	generationClosers := 0
	gen := routerruntime.NewGeneration()
	gen.Defer(func() error {
		generationClosers++
		return nil
	})

	fakeCache := &closeTrackingCache{}
	router := &OpenAIRouter{Cache: fakeCache, generation: gen}

	if err := router.Close(); err != nil {
		t.Fatalf("router.Close() error = %v", err)
	}

	if generationClosers != 1 {
		t.Fatalf("generation closers ran %d times, want exactly 1", generationClosers)
	}
	if got := fakeCache.closeCount(); got != 0 {
		t.Fatalf("Close() bypassed the generation and closed Cache directly %d times", got)
	}
}

// closeTrackingEmbeddingProvider is a minimal embedding.Provider that is also
// an io.Closer, standing in for a remote embedding backend's client — the case
// where a tools database actually holds something to release.
type closeTrackingEmbeddingProvider struct {
	mu     sync.Mutex
	closes int
}

func (p *closeTrackingEmbeddingProvider) Embed(context.Context, string) ([]float32, error) {
	return nil, nil
}

func (p *closeTrackingEmbeddingProvider) EmbedBatch(context.Context, []string) ([][]float32, error) {
	return nil, nil
}

func (p *closeTrackingEmbeddingProvider) Dimension() int { return 1 }

func (p *closeTrackingEmbeddingProvider) Backend() string { return "close-tracking" }

func (p *closeTrackingEmbeddingProvider) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.closes++
	return nil
}

func (p *closeTrackingEmbeddingProvider) closeCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.closes
}

func toolsDatabaseWithProvider(provider *closeTrackingEmbeddingProvider) *tools.ToolsDatabase {
	return tools.NewToolsDatabase(tools.ToolsDatabaseOptions{Enabled: true, Provider: provider})
}

// TestOpenAIRouterCloseClosesLazilyLoadedToolDatabases covers the field-by-field
// fallback. The per-path databases are built at request time by
// getOrLoadToolDatabaseForSelection, not by the router build, so they are the
// one owned resource that cannot be registered from construction alone.
func TestOpenAIRouterCloseClosesLazilyLoadedToolDatabases(t *testing.T) {
	first := &closeTrackingEmbeddingProvider{}
	second := &closeTrackingEmbeddingProvider{}
	router := &OpenAIRouter{
		toolSelectionDBByPath: map[string]*tools.ToolsDatabase{
			"/decisions/support.json": toolsDatabaseWithProvider(first),
			"/decisions/billing.json": toolsDatabaseWithProvider(second),
		},
	}

	if err := router.Close(); err != nil {
		t.Fatalf("router.Close() error = %v", err)
	}

	if got := first.closeCount(); got != 1 {
		t.Errorf("first tools database's provider closed %d times, want 1", got)
	}
	if got := second.closeCount(); got != 1 {
		t.Errorf("second tools database's provider closed %d times, want 1;"+
			" every per-path database is a separate provider client", got)
	}
	if router.toolSelectionDBByPath != nil {
		t.Error("Close() left the per-path cache populated, so a request arriving after teardown reuses a closed database")
	}
}

// TestBuildRouterRegistersLazyToolDatabasesOnGeneration is the one that matters
// in production: a router built by buildRouterComponents tears down purely
// through its generation, so a resource reachable only from the router itself
// leaks unless the build registered a closer pointing back at it.
func TestBuildRouterRegistersLazyToolDatabasesOnGeneration(t *testing.T) {
	gen := routerruntime.NewGeneration()
	router := (&routerComponents{generation: gen}).buildRouter()

	provider := &closeTrackingEmbeddingProvider{}
	router.toolSelectionDBByPath = map[string]*tools.ToolsDatabase{
		"/decisions/support.json": toolsDatabaseWithProvider(provider),
	}

	if err := router.Close(); err != nil {
		t.Fatalf("router.Close() error = %v", err)
	}

	if got := provider.closeCount(); got != 1 {
		t.Fatalf("per-path tools database's provider closed %d times, want 1;"+
			" the generation never learned the router owned it", got)
	}
}

// TestCloseToolSelectionDatabasesIsIdempotent guards the double-teardown path:
// the fallback calls closeToolSelectionDatabases directly and the generation
// calls it through a registered closer, and an embedding client is not
// necessarily safe to close twice.
func TestCloseToolSelectionDatabasesIsIdempotent(t *testing.T) {
	provider := &closeTrackingEmbeddingProvider{}
	router := &OpenAIRouter{
		toolSelectionDBByPath: map[string]*tools.ToolsDatabase{
			"/decisions/support.json": toolsDatabaseWithProvider(provider),
		},
	}

	for i := 0; i < 3; i++ {
		if err := router.closeToolSelectionDatabases(); err != nil {
			t.Fatalf("closeToolSelectionDatabases() call %d error = %v", i+1, err)
		}
	}

	if got := provider.closeCount(); got != 1 {
		t.Fatalf("provider closed %d times across repeated teardown, want exactly 1", got)
	}
}

// TestCloseRecipeModelSelectorsClosesEveryRecipe asserts the teardown reaches
// non-default recipes. createModelSelectorRegistries builds one registry per
// recipe, each owning its own Elo storage and native ML handles, so registering
// only the default registry's Close leaks the rest on every reload.
func TestCloseRecipeModelSelectorsClosesEveryRecipe(t *testing.T) {
	defaultRegistry := selection.NewRegistry()
	defaultSelector := &closeTrackingSelector{method: selection.MethodStatic}
	defaultRegistry.Register(selection.MethodStatic, defaultSelector)

	otherRegistry := selection.NewRegistry()
	otherSelector := &closeTrackingSelector{method: selection.MethodStatic}
	otherRegistry.Register(selection.MethodStatic, otherSelector)

	err := closeRecipeModelSelectors(map[config.RecipeName]*selection.Registry{
		config.DefaultRecipeName: defaultRegistry,
		"other":                  otherRegistry,
	})
	if err != nil {
		t.Fatalf("closeRecipeModelSelectors() error = %v", err)
	}

	if got := defaultSelector.closeCount(); got != 1 {
		t.Errorf("default recipe's selector closed %d times, want 1", got)
	}
	if got := otherSelector.closeCount(); got != 1 {
		t.Errorf("non-default recipe's selector closed %d times, want 1;"+
			" every recipe has its own registry", got)
	}
}

// TestCloseRecipeModelSelectorsDeduplicatesAliases guards the alias case: the
// default registry is an existing map entry rather than a separate registry, and
// Elo storage does not necessarily tolerate a second Close.
func TestCloseRecipeModelSelectorsDeduplicatesAliases(t *testing.T) {
	shared := selection.NewRegistry()
	selector := &closeTrackingSelector{method: selection.MethodStatic}
	shared.Register(selection.MethodStatic, selector)

	err := closeRecipeModelSelectors(map[config.RecipeName]*selection.Registry{
		config.DefaultRecipeName: shared,
		"alias":                  shared,
		"nil-entry":              nil,
	})
	if err != nil {
		t.Fatalf("closeRecipeModelSelectors() error = %v", err)
	}

	if got := selector.closeCount(); got != 1 {
		t.Fatalf("aliased registry's selector closed %d times, want exactly 1", got)
	}
}

// closeTrackingSelector is a minimal selection.Selector that also implements
// io.Closer, which is how Registry.Close decides what it owns.
type closeTrackingSelector struct {
	method selection.SelectionMethod
	mu     sync.Mutex
	closes int
}

func (s *closeTrackingSelector) Select(context.Context, *selection.SelectionContext) (*selection.SelectionResult, error) {
	return nil, nil
}

func (s *closeTrackingSelector) Method() selection.SelectionMethod { return s.method }

func (s *closeTrackingSelector) UpdateFeedback(context.Context, *selection.Feedback) error {
	return nil
}

func (s *closeTrackingSelector) Tier() selection.AlgorithmTier { return selection.TierSupported }

func (s *closeTrackingSelector) ExternalDependencies() []selection.Dependency { return nil }

func (s *closeTrackingSelector) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closes++
	return nil
}

func (s *closeTrackingSelector) closeCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closes
}
