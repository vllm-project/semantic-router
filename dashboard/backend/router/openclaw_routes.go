package router

import (
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/proxy"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

func newOpenClawHandler(cfg *config.Config, wf *workflowstore.Store) *handlers.OpenClawHandler {
	if !cfg.OpenClawEnabled {
		return nil
	}

	openClawHandler := handlers.NewOpenClawHandler(cfg.OpenClawDataDir, cfg.ReadonlyMode, wf)
	openClawHandler.SetRouterConfigPath(cfg.AbsConfigPath)
	openClawHandler.SetAllowedOrigins(cfg.AllowedOrigins)
	return openClawHandler
}

func registerOpenClawRoutes(
	mux routeRegistrar,
	cfg *config.Config,
	openClawHandler *handlers.OpenClawHandler,
) {
	if cfg.OpenClawEnabled && openClawHandler != nil {
		registerEnabledOpenClawRoutes(mux, openClawHandler)
		log.Printf("OpenClaw API endpoints registered: /api/openclaw/*")
		registerOpenClawProxyRoute(mux, openClawHandler)
		log.Printf("OpenClaw dynamic proxy configured: /embedded/openclaw/{name}/ (WebSocket enabled)")
		return
	}

	registerDisabledOpenClawRoutes(mux)
	log.Printf("OpenClaw feature disabled")
}

func registerEnabledOpenClawRoutes(mux routeRegistrar, openClawHandler *handlers.OpenClawHandler) {
	for _, route := range []struct {
		path    string
		handler http.HandlerFunc
	}{
		{"/api/openclaw/status", openClawHandler.StatusHandler()},
		{"/api/openclaw/skills", openClawHandler.SkillsHandler()},
		{"/api/openclaw/token", openClawHandler.TokenHandler()},
	} {
		registerRouteFunc(mux, auth.ProtectedRoute(route.path, auth.PermOpenClawRead, auth.SensitivitySecret, auth.ResourceOwnerOpenClaw, http.MethodGet), route.handler)
	}
	registerRouteFunc(mux, auth.ProtectedRoute("/api/openclaw/next-port", auth.PermOpenClaw, auth.SensitivitySensitive, auth.ResourceOwnerOpenClaw, http.MethodGet), openClawHandler.NextPortHandler())
	registerOpenClawCollection(mux, "/api/openclaw/teams", "openclaw.team.create", openClawHandler.TeamsHandler())
	registerOpenClawItem(mux, "/api/openclaw/teams/{id}", "openclaw.team", openClawHandler.TeamByIDHandler())
	registerOpenClawCollection(mux, "/api/openclaw/workers", "openclaw.worker.create", openClawHandler.WorkersHandler())
	registerOpenClawItem(mux, "/api/openclaw/workers/{id}", "openclaw.worker", openClawHandler.WorkerByIDHandler())
	registerOpenClawCollection(mux, "/api/openclaw/rooms", "openclaw.room.create", openClawHandler.RoomsHandler())
	roomHandler := openClawHandler.RoomByIDHandler()
	registerRouteFunc(mux, auth.Route("/api/openclaw/rooms/{id}",
		auth.ReadPolicy(http.MethodGet, auth.PermOpenClawRead, auth.SensitivitySensitive, auth.ResourceOwnerOpenClaw),
		auth.MutationPolicy(http.MethodDelete, auth.PermOpenClaw, "openclaw.room.delete", auth.SensitivitySecret, auth.ResourceOwnerOpenClaw, 64<<10),
	), roomHandler)
	registerRouteFunc(mux, auth.Route("/api/openclaw/rooms/{id}/messages",
		auth.ReadPolicy(http.MethodGet, auth.PermOpenClawRead, auth.SensitivitySensitive, auth.ResourceOwnerOpenClaw),
		auth.MutationPolicy(http.MethodPost, auth.PermOpenClawRead, "openclaw.room.message", auth.SensitivitySecret, auth.ResourceOwnerOpenClaw, 2<<20),
	), roomHandler)
	registerRouteFunc(mux, auth.ProtectedRoute("/api/openclaw/rooms/{id}/stream", auth.PermOpenClawRead, auth.SensitivitySensitive, auth.ResourceOwnerOpenClaw, http.MethodGet), roomHandler)
	registerRouteFunc(mux, auth.ProtectedRoute("/api/openclaw/rooms/{id}/ws", auth.PermOpenClawRead, auth.SensitivitySensitive, auth.ResourceOwnerOpenClaw, http.MethodGet), roomHandler)
	for _, route := range []struct {
		path, action string
		handler      http.HandlerFunc
	}{
		{"/api/openclaw/provision", "openclaw.provision", openClawHandler.ProvisionHandler()},
		{"/api/openclaw/start", "openclaw.start", openClawHandler.StartHandler()},
		{"/api/openclaw/stop", "openclaw.stop", openClawHandler.StopHandler()},
	} {
		registerRouteFunc(mux, auth.ProtectedMutationRoute(route.path, auth.PermOpenClaw, route.action, auth.SensitivitySecret, auth.ResourceOwnerOpenClaw, 2<<20, http.MethodPost), route.handler)
	}
	registerRouteFunc(mux, auth.ProtectedMutationRoute("/api/openclaw/containers/{name}", auth.PermOpenClaw, "openclaw.container.delete", auth.SensitivitySecret, auth.ResourceOwnerOpenClaw, 64<<10, http.MethodDelete), openClawHandler.DeleteHandler())
}

func registerOpenClawCollection(mux routeRegistrar, path, action string, handler http.HandlerFunc) {
	registerRouteFunc(mux, auth.Route(path,
		auth.ReadPolicy(http.MethodGet, auth.PermOpenClawRead, auth.SensitivitySensitive, auth.ResourceOwnerOpenClaw),
		auth.MutationPolicy(http.MethodPost, auth.PermOpenClaw, action, auth.SensitivitySecret, auth.ResourceOwnerOpenClaw, 2<<20),
	), handler)
}

func registerOpenClawItem(mux routeRegistrar, path, action string, handler http.HandlerFunc) {
	registerRouteFunc(mux, auth.Route(path,
		auth.ReadPolicy(http.MethodGet, auth.PermOpenClawRead, auth.SensitivitySensitive, auth.ResourceOwnerOpenClaw),
		auth.MutationPolicy(http.MethodPut, auth.PermOpenClaw, action+".update", auth.SensitivitySecret, auth.ResourceOwnerOpenClaw, 2<<20),
		auth.MutationPolicy(http.MethodDelete, auth.PermOpenClaw, action+".delete", auth.SensitivitySecret, auth.ResourceOwnerOpenClaw, 64<<10),
	), handler)
}

func registerOpenClawProxyRoute(mux routeRegistrar, openClawHandler *handlers.OpenClawHandler) {
	var proxyCache sync.Map // map[string]http.Handler
	registerRouteFunc(mux, auth.ProtectedRoute("/embedded/openclaw/", auth.PermOpenClawRead, auth.SensitivitySecret, auth.ResourceOwnerOpenClaw, http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
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

		targetBase, ok := openClawHandler.TargetBaseForContainer(name)
		if !ok {
			http.Error(w, "container not found in registry", http.StatusNotFound)
			return
		}

		token := strings.TrimSpace(openClawHandler.GatewayTokenForContainer(name))
		staticHeaders := map[string]string{}
		if token != "" {
			staticHeaders["Authorization"] = "Bearer " + token
			staticHeaders["X-OpenClaw-Token"] = token
		}

		stripPrefix := "/embedded/openclaw/" + name
		cacheKey := fmt.Sprintf("%s:%s:%s", name, targetBase, token)
		handler, loaded := proxyCache.Load(cacheKey)
		if !loaded {
			h, err := proxy.NewWebSocketAwareHandlerWithHeaders(targetBase, stripPrefix, staticHeaders)
			if err != nil {
				log.Printf("Failed to create proxy for %s: %v", name, err)
				http.Error(w, "proxy error", http.StatusBadGateway)
				return
			}
			handler, _ = proxyCache.LoadOrStore(cacheKey, h)
		}

		roomID := strings.TrimSpace(r.Header.Get("X-OpenClaw-Room-Id"))
		if roomID == "" {
			roomID = strings.TrimSpace(r.URL.Query().Get("roomId"))
		}
		if roomID != "" {
			r.Header.Set("X-OpenClaw-Room-Id", roomID)
		}

		handler.(http.Handler).ServeHTTP(w, r)
	})
}

func registerDisabledOpenClawRoutes(mux routeRegistrar) {
	for _, path := range []string{"/api/openclaw/status", "/api/openclaw/teams", "/api/openclaw/workers", "/api/openclaw/rooms"} {
		registerRouteFunc(mux, auth.ProtectedRoute(path, auth.PermOpenClawRead, auth.SensitivityOperational, auth.ResourceOwnerOpenClaw, http.MethodGet), writeOpenClawArray)
	}
	registerRouteFunc(mux, auth.ProtectedRoute("/api/openclaw/rooms/{id}", auth.PermOpenClawRead, auth.SensitivityOperational, auth.ResourceOwnerOpenClaw, http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		http.Error(w, `{"error":"OpenClaw feature disabled"}`, http.StatusServiceUnavailable)
	})
	registerRouteFunc(mux,
		auth.ProtectedRoute("/embedded/openclaw/", auth.PermOpenClawRead, auth.SensitivityOperational, auth.ResourceOwnerOpenClaw, http.MethodGet),
		serviceUnavailableHTMLHandler("OpenClaw", "OPENCLAW_ENABLED", "true"),
	)
}

func writeOpenClawArray(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`[]`))
}
