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
	return &config.RouterConfig{Memory: config.MemoryConfig{Persistence: persistence}}
}

func TestMemoryPersistenceGraceFallsBackToDefault(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  *config.RouterConfig
		want time.Duration
	}{
		{"nil config", nil, memory.DefaultPersistenceShutdownGrace},
		{"unset", memoryPersistenceConfig(config.MemoryPersistenceConfig{}), memory.DefaultPersistenceShutdownGrace},
		{
			"non-positive",
			memoryPersistenceConfig(config.MemoryPersistenceConfig{ShutdownGraceSeconds: -1}),
			memory.DefaultPersistenceShutdownGrace,
		},
		{
			"configured",
			memoryPersistenceConfig(config.MemoryPersistenceConfig{ShutdownGraceSeconds: 12}),
			12 * time.Second,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, memoryPersistenceGrace(tc.cfg))
		})
	}
}

func TestCreateMemoryPersistenceRunnerReturnsNilWithoutConfig(t *testing.T) {
	assert.Nil(t, createMemoryPersistenceRunner(nil))
}

func TestCreateMemoryPersistenceRunnerRunsWorkOnUnsetBounds(t *testing.T) {
	runner := createMemoryPersistenceRunner(memoryPersistenceConfig(config.MemoryPersistenceConfig{}))
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
