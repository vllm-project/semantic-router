package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// withGlobalPlugins makes plugins cfg's global.plugins and layers them over its
// recipes and decisions, as the loader does.
func withGlobalPlugins(t testing.TB, cfg *config.RouterConfig, plugins ...config.DecisionPlugin) *config.RouterConfig {
	t.Helper()
	cfg.GlobalPlugins = append(cfg.GlobalPlugins, plugins...)
	if err := cfg.LayerPlugins(); err != nil {
		t.Fatal(err)
	}
	return cfg
}

func testPlugin(pluginType string, configuration map[string]interface{}) config.DecisionPlugin {
	return config.DecisionPlugin{Type: pluginType, Configuration: config.MustStructuredPayload(configuration)}
}

// replayTestPlugin is a router_replay plugin of the given configuration.
func replayTestPlugin(configuration map[string]interface{}) config.DecisionPlugin {
	return testPlugin(config.DecisionPluginRouterReplay, configuration)
}

// memoryTestPlugin is a context plugin that recalls memories.
func memoryTestPlugin() config.DecisionPlugin {
	return testPlugin(config.DecisionPluginContext, map[string]interface{}{
		"enabled": true, "sources": []interface{}{map[string]interface{}{"type": config.ContextSourceMemory}},
	})
}
