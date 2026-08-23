package apiserver

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

type liveRuntimeConfig struct {
	fallback *config.RouterConfig
	resolver func() *config.RouterConfig
}

func newLiveRuntimeConfig(
	fallback *config.RouterConfig,
	resolver func() *config.RouterConfig,
) *liveRuntimeConfig {
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
	return c.fallback
}

func (s *ClassificationAPIServer) currentConfig() *config.RouterConfig {
	if s == nil {
		return nil
	}
	if s.runtimeConfig != nil {
		return s.runtimeConfig.Current()
	}
	return s.config
}
