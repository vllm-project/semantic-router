//go:build !windows

package apiserver

import (
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type liveRuntimeConfig struct {
	mu       sync.RWMutex
	fallback *config.RouterConfig
	resolver func() *config.RouterConfig
}

func newLiveRuntimeConfig(fallback *config.RouterConfig, resolver func() *config.RouterConfig) *liveRuntimeConfig {
	return &liveRuntimeConfig{
		fallback: fallback,
		resolver: resolver,
	}
}

func (c *liveRuntimeConfig) Current() *config.RouterConfig {
	if c == nil {
		return nil
	}
	if c.resolver != nil {
		if cfg := c.resolver(); cfg != nil {
			return cfg
		}
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.fallback
}

func (s *ClassificationAPIServer) currentConfig() *config.RouterConfig {
	if s == nil {
		return nil
	}
	if s.runtimeConfig != nil {
		return s.runtimeConfig.Current()
	}
	s.configMu.RLock()
	defer s.configMu.RUnlock()
	return s.config
}

// servingConfig is the configuration the Router serves, or nil for an API
// without the Router's runtime. The persisted document differs from it while
// a change activates, and after one was rejected.
func (s *ClassificationAPIServer) servingConfig() *config.RouterConfig {
	if s == nil || s.runtimeRegistry == nil {
		return nil
	}
	return s.runtimeRegistry.CurrentConfig()
}
