package router

import (
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/proxy"
)

func buildOpenClawHandler(cfg *config.Config) *handlers.OpenClawHandler {
	if !cfg.OpenClawEnabled {
		return nil
	}
	openClawHandler := handlers.NewOpenClawHandler(cfg.OpenClawDataDir, cfg.ReadonlyMode)
	openClawHandler.SetRouterConfigPath(cfg.AbsConfigPath)
	return openClawHandler
}

func registerOpenClawRoutes(mux *http.ServeMux, cfg *config.Config, openClawHandler *handlers.OpenClawHandler) {
	if cfg.OpenClawEnabled && openClawHandler != nil {
		registerOpenClawAPIRoutes(mux, openClawHandler)
		registerOpenClawEmbeddedProxy(mux, openClawHandler)
		log.Printf("OpenClaw API endpoints registered: /api/openclaw/*")
		log.Printf("OpenClaw dynamic proxy configured: /embedded/openclaw/{name}/ (WebSocket enabled)")
		return
	}

	registerOpenClawDisabledRoutes(mux)
	log.Printf("OpenClaw feature disabled")
}

func registerOpenClawAPIRoutes(mux *http.ServeMux, openClawHandler *handlers.OpenClawHandler) {
	mux.HandleFunc("/api/openclaw/status", openClawHandler.StatusHandler())
	mux.HandleFunc("/api/openclaw/skills", openClawHandler.SkillsHandler())
	mux.HandleFunc("/api/openclaw/teams", openClawHandler.TeamsHandler())
	mux.HandleFunc("/api/openclaw/teams/", openClawHandler.TeamByIDHandler())
	mux.HandleFunc("/api/openclaw/workers", openClawHandler.WorkersHandler())
	mux.HandleFunc("/api/openclaw/workers/", openClawHandler.WorkerByIDHandler())
	mux.HandleFunc("/api/openclaw/rooms", openClawHandler.RoomsHandler())
	mux.HandleFunc("/api/openclaw/rooms/", openClawHandler.RoomByIDHandler())
	mux.HandleFunc("/api/openclaw/provision", openClawHandler.ProvisionHandler())
	mux.HandleFunc("/api/openclaw/start", openClawHandler.StartHandler())
	mux.HandleFunc("/api/openclaw/stop", openClawHandler.StopHandler())
	mux.HandleFunc("/api/openclaw/token", openClawHandler.TokenHandler())
	mux.HandleFunc("/api/openclaw/next-port", openClawHandler.NextPortHandler())
	mux.HandleFunc("/api/openclaw/containers/", openClawHandler.DeleteHandler())
}

func registerOpenClawEmbeddedProxy(mux *http.ServeMux, openClawHandler *handlers.OpenClawHandler) {
	var proxyCache sync.Map

	mux.HandleFunc("/embedded/openclaw/", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		rest := strings.TrimPrefix(r.URL.Path, "/embedded/openclaw/")
		parts := strings.SplitN(rest, "/", 2)
		name := parts[0]
		if name == "" {
			http.Error(w, "container name required in path", http.StatusBadRequest)
			return
		}

		targetBase, token, ok := resolveOpenClawTarget(name, openClawHandler)
		if !ok {
			http.Error(w, "container not found in registry", http.StatusNotFound)
			return
		}

		handler, err := getOpenClawProxyHandler(&proxyCache, name, targetBase, token)
		if err != nil {
			log.Printf("Failed to create proxy for %s: %v", name, err)
			http.Error(w, "proxy error", http.StatusBadGateway)
			return
		}
		handler.ServeHTTP(w, r)
	})
}

func resolveOpenClawTarget(name string, openClawHandler *handlers.OpenClawHandler) (string, string, bool) {
	targetBase, ok := openClawHandler.TargetBaseForContainer(name)
	if !ok {
		return "", "", false
	}
	return targetBase, strings.TrimSpace(openClawHandler.GatewayTokenForContainer(name)), true
}

func getOpenClawProxyHandler(proxyCache *sync.Map, name, targetBase, token string) (http.Handler, error) {
	cacheKey := fmt.Sprintf("%s:%s:%s", name, targetBase, token)
	if handler, ok := proxyCache.Load(cacheKey); ok {
		return handler.(http.Handler), nil
	}

	headers := map[string]string{}
	if token != "" {
		headers["Authorization"] = "Bearer " + token
		headers["X-OpenClaw-Token"] = token
	}
	stripPrefix := "/embedded/openclaw/" + name
	handler, err := proxy.NewWebSocketAwareHandlerWithHeaders(targetBase, stripPrefix, headers)
	if err != nil {
		return nil, err
	}
	cached, _ := proxyCache.LoadOrStore(cacheKey, handler)
	return cached.(http.Handler), nil
}

func registerOpenClawDisabledRoutes(mux *http.ServeMux) {
	for _, path := range []string{
		"/api/openclaw/status",
		"/api/openclaw/teams",
		"/api/openclaw/workers",
		"/api/openclaw/rooms",
	} {
		mux.HandleFunc(path, staticJSONHandler(`[]`))
	}
	mux.HandleFunc("/api/openclaw/rooms/", jsonErrorHandler(http.StatusServiceUnavailable, `{"error":"OpenClaw feature disabled"}`))
	mux.HandleFunc("/embedded/openclaw/", notConfiguredServiceHandler("OpenClaw", "OPENCLAW_ENABLED", "true"))
}
