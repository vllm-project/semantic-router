package routerruntime

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
)

// PluginInventory observes published runtime dependencies, not network health.
// Its inspector and configuration belong to the same leased generation.
type PluginInventory struct {
	Config    *config.RouterConfig
	Inspector pluginruntime.BindingInspector
}

func (r *Registry) AcquirePluginInventory() (PluginInventory, func(), bool) {
	cfg, plugins, release, ok := r.AcquirePluginRuntime()
	return PluginInventory{Config: cfg, Inspector: plugins.Inspector}, release, ok
}
