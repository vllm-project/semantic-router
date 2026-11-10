package extproc

import (
	"runtime"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildRouterComponentsClosesEarlierResourcesOnLaterFailure(t *testing.T) {
	cfg := newDefaultRouteConfig()
	cfg.SemanticCache = config.SemanticCache{InUse: true, TTLSeconds: 60}
	// The replay store is built after the signal runtime and the cache.
	cfg.RouterReplay = config.RouterReplayConfig{StoreBackend: "redis"}
	withGlobalPlugins(t, cfg, replayTestPlugin(map[string]interface{}{"enabled": true}))

	baseline := stableGoroutineCount(t)

	components, err := buildRouterComponents(cfg)
	require.ErrorContains(t, err, "redis config required")
	require.Nil(t, components)

	require.Eventually(t, func() bool {
		runtime.GC()
		return runtime.NumGoroutine() <= baseline
	}, 10*time.Second, 10*time.Millisecond)
}

func stableGoroutineCount(t *testing.T) int {
	t.Helper()
	var last int
	consecutive := 0
	require.Eventually(t, func() bool {
		runtime.GC()
		current := runtime.NumGoroutine()
		if current == last {
			consecutive++
		} else {
			consecutive = 0
			last = current
		}
		return consecutive >= 3
	}, 10*time.Second, 10*time.Millisecond, "goroutine count never settled")
	return last
}

func newDefaultRouteConfig() *config.RouterConfig {
	return &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name:  "default-route",
				Rules: config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{}},
			}},
		},
	}
}
