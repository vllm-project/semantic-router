//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"net/http"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func (s *ClassificationAPIServer) handleStartupStatus(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	state := s.loadStartupState()
	if state == nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"phase":"unknown","ready":false,"message":"startup status not available"}`))
		return
	}

	payload, err := json.Marshal(state)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	status := http.StatusOK
	if !state.Ready {
		status = http.StatusServiceUnavailable
	}
	w.WriteHeader(status)
	_, _ = w.Write(payload)
}

func (s *ClassificationAPIServer) loadStartupState() *startupstatus.State {
	if s.startupStateLoader != nil {
		return s.startupStateLoader()
	}

	var startupConfig *config.StartupStatusConfig
	if s.startupStatusConfig != nil {
		startupConfig = s.startupStatusConfig
	} else if cfg := s.currentConfig(); cfg != nil {
		startupConfig = &cfg.StartupStatus
	}
	if startupConfig != nil && startupConfig.StoreBackend == "redis" && startupConfig.Redis != nil {
		client := redis.NewClient(&redis.Options{
			Addr:     startupConfig.Redis.Address,
			Password: startupConfig.Redis.Password,
			DB:       startupConfig.Redis.DB,
		})
		defer func() { _ = client.Close() }()

		state, err := startupstatus.LoadFromRedis(client)
		if err == nil {
			return state
		}
	}

	state, _ := startupstatus.Load(startupstatus.StatusPathFromConfigPath(s.configPath))
	return state
}
