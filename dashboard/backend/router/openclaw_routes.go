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

const maxOpenClawBodyBytes = 2 << 20

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
	routes *auth.PolicyMux,
	cfg *config.Config,
	openClawHandler *handlers.OpenClawHandler,
) {
	if cfg.OpenClawEnabled && openClawHandler != nil {
		registerEnabledOpenClawRoutes(routes, openClawHandler)
		log.Printf("OpenClaw API endpoints registered: /api/openclaw/*")
		registerOpenClawProxyRoute(routes, openClawHandler)
		log.Printf("OpenClaw dynamic proxy configured: /embedded/openclaw/{name}/ (WebSocket enabled)")
		return
	}

	registerDisabledOpenClawRoutes(routes)
	log.Printf("OpenClaw feature disabled")
}

func registerEnabledOpenClawRoutes(routes *auth.PolicyMux, openClawHandler *handlers.OpenClawHandler) {
	routes.HandleFunc(openClawReadRoute("/api/openclaw/status"), openClawHandler.StatusHandler())
	routes.HandleFunc(openClawReadRoute("/api/openclaw/skills"), openClawHandler.SkillsHandler())
	routes.HandleFunc(openClawCollectionRoute("/api/openclaw/teams", "openclaw.team.create"), openClawHandler.TeamsHandler())
	routes.HandleFunc(openClawItemRoute("/api/openclaw/teams/{id}", "openclaw.team"), openClawHandler.TeamByIDHandler())
	routes.HandleFunc(openClawCollectionRoute("/api/openclaw/workers", "openclaw.worker.create"), openClawHandler.WorkersHandler())
	routes.HandleFunc(openClawItemRoute("/api/openclaw/workers/{id}", "openclaw.worker"), openClawHandler.WorkerByIDHandler())
	routes.HandleFunc(openClawCollectionRoute("/api/openclaw/rooms", "openclaw.room.create"), openClawHandler.RoomsHandler())
	routes.HandleGroup([]auth.RouteContract{
		auth.Route("/api/openclaw/rooms/{id}",
			auth.ReadPolicy(http.MethodGet, auth.PermOpenClawRead, auth.SensitivitySensitive, auth.ResourceOwnerOpenClaw),
			auth.MutationPolicy(http.MethodDelete, auth.PermOpenClaw, "openclaw.room.delete", auth.SensitivitySecret, auth.ResourceOwnerOpenClaw, auth.NoBodyLimit),
		),
		// Room participation is a read-role capability: messages, stream, and
		// the websocket carry the conversation rather than reconfigure it.
		auth.ProtectedBoundedRoute("/api/openclaw/rooms/{id}/messages", auth.PermOpenClawRead, auth.SensitivitySensitive, auth.ResourceOwnerOpenClaw, maxOpenClawBodyBytes, http.MethodGet, http.MethodPost),
		openClawReadRoute("/api/openclaw/rooms/{id}/stream"),
		openClawReadRoute("/api/openclaw/rooms/{id}/ws"),
	}, openClawHandler.RoomByIDHandler())
	routes.HandleFunc(openClawMutationRoute("/api/openclaw/provision", "openclaw.provision"), openClawHandler.ProvisionHandler())
	routes.HandleFunc(openClawMutationRoute("/api/openclaw/start", "openclaw.start"), openClawHandler.StartHandler())
	routes.HandleFunc(openClawMutationRoute("/api/openclaw/stop", "openclaw.stop"), openClawHandler.StopHandler())
	routes.HandleFunc(openClawReadRoute("/api/openclaw/token"), openClawHandler.TokenHandler())
	routes.HandleFunc(openClawReadRoute("/api/openclaw/next-port"), openClawHandler.NextPortHandler())
	routes.HandleFunc(
		auth.ProtectedMutationRoute("/api/openclaw/containers/{name}", auth.PermOpenClaw, "openclaw.container.delete", auth.SensitivitySecret, auth.ResourceOwnerOpenClaw, auth.NoBodyLimit, http.MethodDelete),
		openClawHandler.DeleteHandler(),
	)
}

func registerOpenClawProxyRoute(routes *auth.PolicyMux, openClawHandler *handlers.OpenClawHandler) {
	var proxyCache sync.Map // map[string]http.Handler
	routes.HandleFunc(openClawEmbeddedRoute(), func(w http.ResponseWriter, r *http.Request) {
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

func registerDisabledOpenClawRoutes(routes *auth.PolicyMux) {
	routes.HandleFunc(openClawReadRoute("/api/openclaw/status"), writeOpenClawArray)
	routes.HandleFunc(openClawReadRoute("/api/openclaw/teams"), writeOpenClawArray)
	routes.HandleFunc(openClawReadRoute("/api/openclaw/workers"), writeOpenClawArray)
	routes.HandleFunc(openClawReadRoute("/api/openclaw/rooms"), writeOpenClawArray)
	routes.HandleFunc(openClawReadRoute("/api/openclaw/rooms/"), func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		http.Error(w, `{"error":"OpenClaw feature disabled"}`, http.StatusServiceUnavailable)
	})
	routes.HandleFunc(
		openClawEmbeddedRoute(),
		serviceUnavailableHTMLHandler("OpenClaw", "OPENCLAW_ENABLED", "true"),
	)
}

func writeOpenClawArray(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`[]`))
}

func openClawReadRoute(pattern string) auth.RouteContract {
	return auth.ProtectedRoute(pattern, auth.PermOpenClawRead, auth.SensitivitySensitive, auth.ResourceOwnerOpenClaw, http.MethodGet)
}

// openClawEmbeddedRoute fronts a proxied gateway UI, so every method the
// embedded application uses passes through under the read permission.
func openClawEmbeddedRoute() auth.RouteContract {
	return auth.ProtectedRoute(
		"/embedded/openclaw/",
		auth.PermOpenClawRead,
		auth.SensitivitySecret,
		auth.ResourceOwnerOpenClaw,
		http.MethodGet, http.MethodHead, http.MethodPost, http.MethodPut, http.MethodPatch, http.MethodDelete,
	)
}

func openClawCollectionRoute(pattern, createAction string) auth.RouteContract {
	return auth.Route(
		pattern,
		auth.ReadPolicy(http.MethodGet, auth.PermOpenClawRead, auth.SensitivitySensitive, auth.ResourceOwnerOpenClaw),
		auth.MutationPolicy(http.MethodPost, auth.PermOpenClaw, createAction, auth.SensitivitySecret, auth.ResourceOwnerOpenClaw, maxOpenClawBodyBytes),
	)
}

func openClawItemRoute(pattern, actionPrefix string) auth.RouteContract {
	return auth.Route(
		pattern,
		auth.ReadPolicy(http.MethodGet, auth.PermOpenClawRead, auth.SensitivitySensitive, auth.ResourceOwnerOpenClaw),
		auth.MutationPolicy(http.MethodPut, auth.PermOpenClaw, actionPrefix+".update", auth.SensitivitySecret, auth.ResourceOwnerOpenClaw, maxOpenClawBodyBytes),
		auth.MutationPolicy(http.MethodPatch, auth.PermOpenClaw, actionPrefix+".update", auth.SensitivitySecret, auth.ResourceOwnerOpenClaw, maxOpenClawBodyBytes),
		auth.MutationPolicy(http.MethodDelete, auth.PermOpenClaw, actionPrefix+".delete", auth.SensitivitySecret, auth.ResourceOwnerOpenClaw, auth.NoBodyLimit),
	)
}

func openClawMutationRoute(pattern, action string) auth.RouteContract {
	return auth.ProtectedMutationRoute(pattern, auth.PermOpenClaw, action, auth.SensitivitySecret, auth.ResourceOwnerOpenClaw, maxOpenClawBodyBytes, http.MethodPost)
}
