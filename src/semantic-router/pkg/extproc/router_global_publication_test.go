package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func restoreProcessGlobals(t *testing.T) {
	t.Helper()

	originalRegistry := selection.GetGlobalRegistry()
	originalMemoryStore := memory.GetGlobalMemoryStore()
	originalClassification := services.GetGlobalClassificationService()
	t.Cleanup(func() {
		selection.SetGlobalRegistry(originalRegistry)
		memory.SetGlobalMemoryStore(originalMemoryStore)
		services.SetGlobalClassificationService(originalClassification)
	})
}

func TestBuildRouterComponentsDoesNotPublishProcessGlobals(t *testing.T) {
	restoreProcessGlobals(t)

	previous := selection.NewRegistry()
	selection.SetGlobalRegistry(previous)

	cfg := &config.RouterConfig{}
	components, err := buildRouterComponents(cfg)
	require.NoError(t, err)
	require.NotNil(t, components.modelSelector,
		"the build produced no selection registry, so this test would pass without exercising publication")

	require.Same(t, previous, selection.GetGlobalRegistry(),
		"buildRouterComponents published its candidate registry process-wide before the build committed")

	require.NoError(t, components.buildRouter().Close())
	require.Same(t, previous, selection.GetGlobalRegistry(),
		"a discarded candidate left the process-wide registry pointing at a closed registry")
}

func TestPublishRouterStateAdoptsCommittedRouterGlobals(t *testing.T) {
	restoreProcessGlobals(t)

	selection.SetGlobalRegistry(selection.NewRegistry())

	committed := selection.NewRegistry()
	router := &OpenAIRouter{ModelSelector: committed}

	publishRouterState(&config.RouterConfig{}, router, nil)

	require.Same(t, committed, selection.GetGlobalRegistry(),
		"publishRouterState did not adopt the committed router's selection registry")
}
