package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Every store the configuration schema and the Dashboard offer is one the
// Router creates: memory per decision, the rest shared, each asking at most
// for its own connection settings. A name outside the list is refused.
func TestReplayStoreBackendsAreTheStoresTheRouterCreates(t *testing.T) {
	cfg := &config.RouterConfig{}
	plugin := &config.RouterReplayPluginConfig{}
	for _, backend := range config.RouterReplayStoreBackends() {
		var err error
		if backend == "memory" {
			_, err = createReplayStore("decision", backend, plugin, cfg)
		} else {
			require.True(t, usesSharedReplayStorage(backend), backend)
			_, err = createSharedReplayStore(backend, cfg, plugin)
		}
		if err != nil {
			require.Contains(t, err.Error(), backend+" config required", backend)
		}
	}
	require.False(t, usesSharedReplayStorage("sqlite"))
	_, err := createReplayStore("decision", "sqlite", plugin, cfg)
	require.ErrorContains(t, err, "supported: file, memory, redis, postgres, milvus, qdrant")
}
