package router

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/accesscontrol"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
)

func setupAccessControlRoutes(mux *http.ServeMux, cfg *config.Config) *accesscontrol.Service {
	if !cfg.AccessControlEnabled {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	store, err := accesscontrol.OpenStore(ctx, cfg.AccessControlDatabaseURL)
	if err != nil {
		registerAccessControlUnavailable(mux, err)
		return nil
	}
	quota, err := accesscontrol.OpenQuotaManager(ctx, cfg.AccessControlRedisURL)
	if err != nil {
		store.Close()
		registerAccessControlUnavailable(mux, err)
		return nil
	}
	service, err := accesscontrol.NewService(store, quota, cfg.AccessControlKeySecret)
	if err != nil {
		quota.Close()
		store.Close()
		registerAccessControlUnavailable(mux, err)
		return nil
	}
	gateway, err := handlers.NewAccessGatewayHandler(service, cfg.EnvoyURL)
	if err != nil {
		service.Close()
		registerAccessControlUnavailable(mux, err)
		return nil
	}
	management := handlers.NewAccessControlHandler(service)
	selfAccess := handlers.NewSelfAccessControlHandler(service)
	mux.Handle("/api/v1/access-control/self", selfAccess)
	mux.Handle("/api/v1/access-control/self/", selfAccess)
	mux.Handle("/api/v1/access-control", management)
	mux.Handle("/api/v1/access-control/", management)
	mux.HandleFunc("/v1/models", gateway.Models)
	mux.HandleFunc("/v1/chat/completions", gateway.ChatCompletions)
	mux.HandleFunc("/api/playground/v1/models", gateway.DashboardModels)
	mux.HandleFunc("/api/playground/v1/chat/completions", gateway.DashboardChatCompletions)
	log.Printf("Inference access control registered: /api/v1/access-control/*, /api/playground/v1/*, /v1/models, /v1/chat/completions")
	return service
}

func registerAccessControlUnavailable(mux *http.ServeMux, cause error) {
	log.Printf("ERROR: inference access control is enabled but unavailable: %v", cause)
	handler := func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{"message": "inference access control is unavailable", "status": http.StatusServiceUnavailable},
		})
	}
	mux.HandleFunc("/api/v1/access-control", handler)
	mux.HandleFunc("/api/v1/access-control/", handler)
	mux.HandleFunc("/api/playground/v1/models", handler)
	mux.HandleFunc("/api/playground/v1/chat/completions", handler)
	mux.HandleFunc("/v1/models", handler)
	mux.HandleFunc("/v1/chat/completions", handler)
}
