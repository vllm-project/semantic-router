//go:build !windows && cgo

package apiserver

import (
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/mcpconfig"
)

const internalMCPConfigPath = mcpconfig.InternalHTTPPath

func (s *ClassificationAPIServer) registerMCPConfigRoutes(mux *http.ServeMux) {
	handler := s.buildMCPConfigHandler()
	if handler == nil {
		return
	}
	mux.Handle(internalMCPConfigPath, handler)
}

func (s *ClassificationAPIServer) buildMCPConfigHandler() http.Handler {
	if s == nil || s.configPath == "" {
		return nil
	}

	cfg := s.currentMCPConfig()
	if !cfg.Enabled {
		return nil
	}

	server, err := mcpconfig.NewServer(s.configPath, cfg)
	if err != nil {
		return nil
	}

	handler := server.HTTPHandler()
	if cfg.LoopbackOnly {
		return mcpconfig.LoopbackOnly(handler)
	}
	return handler
}

func (s *ClassificationAPIServer) currentMCPConfig() config.MCPConfigServerConfig {
	if s == nil {
		return config.MCPConfigServerConfig{}
	}
	if s.config != nil {
		return s.config.MCPConfig.WithDefaults()
	}
	if s.runtimeConfig != nil {
		if cfg := s.runtimeConfig.Current(); cfg != nil {
			return cfg.MCPConfig.WithDefaults()
		}
	}
	return config.MCPConfigServerConfig{}
}
