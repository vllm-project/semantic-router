package routerruntime

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
)

// AcquirePluginRuntime pins the router generation for the complete diagnostic,
// including slow retrieval or model calls. Reload cannot close its dependencies.
func (r *Registry) AcquirePluginRuntime() (*config.RouterConfig, pluginruntime.Capabilities, func(), bool) {
	if r == nil {
		return nil, pluginruntime.Capabilities{}, func() {}, false
	}
	r.mu.RLock()
	release, ok := acquireRuntimeLease(r.learningRuntime)
	if !ok {
		r.mu.RUnlock()
		return nil, pluginruntime.Capabilities{}, func() {}, false
	}
	classificationRelease := func() {}
	if r.acquireGeneration != nil {
		classificationRelease, ok = r.acquireGeneration()
		if !ok {
			r.mu.RUnlock()
			release()
			return nil, pluginruntime.Capabilities{}, func() {}, false
		}
	}
	cfg, plugins := r.config, r.plugins
	r.mu.RUnlock()
	return cfg, plugins, func() { classificationRelease(); release() }, true
}
