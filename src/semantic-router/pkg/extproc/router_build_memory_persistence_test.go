package extproc

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

func memoryPersistenceConfig(persistence config.MemoryPersistenceConfig) *config.RouterConfig {
	return &config.RouterConfig{Memory: config.MemoryConfig{Enabled: true, Persistence: persistence}}
}

func memoryPersistenceExtractor() *memory.MemoryExtractor {
	return memory.NewMemoryChunkStore(&noopMemoryStore{})
}

func TestCreateMemoryPersistenceRunnerReturnsNilWithoutConfig(t *testing.T) {
	assert.Nil(t, createMemoryPersistenceRunner(nil, memoryPersistenceExtractor()))
	assert.Nil(t, createMemoryPersistenceRunner(&config.RouterConfig{}, memoryPersistenceExtractor()))
}

// An unreachable backend leaves createMemoryRuntime with a nil extractor while
// enablement still reads true, and every write would be suppressed as
// "no_extractor". No workers or queue storage may be allocated for that.
func TestCreateMemoryPersistenceRunnerReturnsNilWithoutExtractor(t *testing.T) {
	assert.Nil(t, createMemoryPersistenceRunner(memoryPersistenceConfig(config.MemoryPersistenceConfig{}), nil))
}

func TestCreateMemoryPersistenceRunnerRunsWorkOnUnsetBounds(t *testing.T) {
	runner := createMemoryPersistenceRunner(memoryPersistenceConfig(config.MemoryPersistenceConfig{}), memoryPersistenceExtractor())
	require.NotNil(t, runner)

	done := make(chan string, 2)
	runner.Submit(context.Background(), memory.PersistenceJob{
		Run:    func(context.Context) (memory.PersistenceOutcome, error) { return memory.PersistenceOutcome{}, nil },
		Report: func(status, _ string, _ bool, _ error) { done <- status },
	})
	require.NoError(t, runner.RetireAndWait(time.Second))

	close(done)
	var seen []string
	for status := range done {
		seen = append(seen, status)
	}
	assert.Equal(t, []string{"scheduled", "completed"}, seen)
}

func TestBuildRouterWiresMemoryPersistenceRunner(t *testing.T) {
	runner := memory.NewPersistenceRunner(time.Second, 1, 4)
	components := &routerComponents{
		cfg:               &config.RouterConfig{Memory: config.MemoryConfig{AutoStore: true}},
		memoryExtractor:   memory.NewMemoryChunkStore(&noopMemoryStore{}),
		memoryPersistence: runner,
		resources:         newResourceScope(),
	}

	router := components.buildRouter()
	require.Same(t, runner, router.memoryPersistence)

	assert.NotPanics(t, func() {
		router.scheduleSemanticResponseMemoryStore(&RequestContext{
			RequestID:    "req-build-wiring",
			TraceContext: context.Background(),
		}, memoryTestResponse("wired through the built router"))
	})
	require.NoError(t, runner.RetireAndWait(time.Second))
}
