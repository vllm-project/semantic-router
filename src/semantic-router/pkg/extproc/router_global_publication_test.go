package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// restoreProcessGlobals snapshots the legacy package-level globals
// publishRouterState writes to and restores them when the test ends, so a test
// asserting on publication cannot leak state into the rest of the package.
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

// TestBuildRouterComponentsDoesNotPublishProcessGlobals is the "construction
// commits only after full success" invariant from issue #2470. A candidate build
// can still be discarded by a later construction step or a failed warmup, and
// its Close then tears down the very objects the globals would be pointing at —
// leaving, for the selection registry, a closed Elo storage that silently drops
// every rating write while the previous router keeps serving. Nothing rolls the
// globals back, so the fix is to never publish from a candidate.
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

	// Discard the candidate the way a failed warmup does. The global must still
	// be the previous registry, not a closed one.
	require.NoError(t, components.buildRouter().Close())
	require.Same(t, previous, selection.GetGlobalRegistry(),
		"a discarded candidate left the process-wide registry pointing at a closed registry")
}

// TestPublishRouterStateAdoptsCommittedRouterGlobals asserts the publication the
// test above moved the responsibility to. Without this the previous test would
// be satisfied by never publishing at all.
func TestPublishRouterStateAdoptsCommittedRouterGlobals(t *testing.T) {
	restoreProcessGlobals(t)

	selection.SetGlobalRegistry(selection.NewRegistry())

	committed := selection.NewRegistry()
	router := &OpenAIRouter{ModelSelector: committed}

	// A nil runtime registry selects the legacy global path; production wires a
	// registry, which publishRouterState prefers.
	publishRouterState(&config.RouterConfig{}, router, nil)

	require.Same(t, committed, selection.GetGlobalRegistry(),
		"publishRouterState did not adopt the committed router's selection registry")
}
