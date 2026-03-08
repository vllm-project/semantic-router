package router

import (
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"

	backendapp "github.com/vllm-project/semantic-router/dashboard/backend/app"
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

func registerOpenClawRoutes(mux *http.ServeMux, app *backendapp.App, openClawHandler *handlers.OpenClawHandler) {
	cfg := app.Config
	access := newRouteAccess(app)
	if cfg.OpenClawEnabled && openClawHandler != nil {
		registerOpenClawAPIRoutes(mux, access, openClawHandler)
		registerOpenClawEmbeddedProxy(mux, app, access, openClawHandler)
		log.Printf("OpenClaw API endpoints registered: /api/openclaw/*")
		log.Printf("OpenClaw dynamic proxy configured: /embedded/openclaw/{name}/ (WebSocket enabled)")
		return
	}

	registerOpenClawDisabledRoutes(mux)
	log.Printf("OpenClaw feature disabled")
}

func registerOpenClawAPIRoutes(mux *http.ServeMux, access routeAccess, openClawHandler *handlers.OpenClawHandler) {
	mux.Handle("/api/openclaw/status", access.admin(openClawHandler.StatusHandler()))
	mux.Handle("/api/openclaw/skills", access.admin(openClawHandler.SkillsHandler()))
	mux.Handle("/api/openclaw/teams", access.admin(openClawHandler.TeamsHandler()))
	mux.Handle("/api/openclaw/teams/", access.admin(openClawHandler.TeamByIDHandler()))
	mux.Handle("/api/openclaw/workers", access.admin(openClawHandler.WorkersHandler()))
	mux.Handle("/api/openclaw/workers/", access.admin(openClawHandler.WorkerByIDHandler()))
	mux.Handle("/api/openclaw/rooms", access.admin(openClawHandler.RoomsHandler()))
	mux.Handle("/api/openclaw/rooms/", access.admin(openClawHandler.RoomByIDHandler()))
	mux.Handle("/api/openclaw/provision", access.admin(openClawHandler.ProvisionHandler()))
	mux.Handle("/api/openclaw/start", access.admin(openClawHandler.StartHandler()))
	mux.Handle("/api/openclaw/stop", access.admin(openClawHandler.StopHandler()))
	mux.Handle("/api/openclaw/token", access.admin(openClawHandler.TokenHandler()))
	mux.Handle("/api/openclaw/next-port", access.admin(openClawHandler.NextPortHandler()))
	mux.Handle("/api/openclaw/containers/", access.admin(openClawHandler.DeleteHandler()))
}

func registerOpenClawEmbeddedProxy(mux *http.ServeMux, app *backendapp.App, access routeAccess, openClawHandler *handlers.OpenClawHandler) {
	var proxyCache sync.Map

	mux.Handle("/embedded/openclaw/", access.admin(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rejectCrossOriginProxyAccess(w, r) {
			return
		}
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
		if len(parts) == 1 || strings.TrimSpace(parts[1]) == "" {
			appendEmbeddedServiceAudit(app, r, "openclaw", name)
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
	})))
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
