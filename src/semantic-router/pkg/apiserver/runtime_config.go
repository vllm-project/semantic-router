package apiserver

import (
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type liveRuntimeConfig struct {
	mu       sync.RWMutex
	fallback *config.RouterConfig
	resolver func() *config.RouterConfig
	updater  func(*config.RouterConfig)
}

func newLiveRuntimeConfig(
	fallback *config.RouterConfig,
	resolver func() *config.RouterConfig,
	updater func(*config.RouterConfig),
) *liveRuntimeConfig {
	return &liveRuntimeConfig{
		fallback: fallback,
		resolver: resolver,
		updater:  updater,
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

func (c *liveRuntimeConfig) Update(newCfg *config.RouterConfig) {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.fallback = newCfg
	c.mu.Unlock()
	if c.updater != nil {
		c.updater(newCfg)
		return
	}
	config.Replace(newCfg)
}

func (s *ClassificationAPIServer) currentConfig() *config.RouterConfig {
	if s.runtimeConfig != nil {
		return s.runtimeConfig.Current()
	}
	return s.config
}

func (s *ClassificationAPIServer) updateRuntimeConfig(newCfg *config.RouterConfig) {
	if s.runtimeConfig != nil {
		s.runtimeConfig.Update(newCfg)
		return
	}
	s.config = newCfg
	if s.classificationSvc != nil {
		s.classificationSvc.RefreshRuntimeConfig(newCfg)
		return
	}
	config.Replace(newCfg)
}
