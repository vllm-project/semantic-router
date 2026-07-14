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
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

func newOpenClawHandler(cfg *config.Config, wf *workflowstore.Store) *handlers.OpenClawHandler {
	if !cfg.OpenClawEnabled {
		return nil
	}

	openClawHandler := handlers.NewOpenClawHandler(cfg.OpenClawDataDir, cfg.ReadonlyMode, wf)
	openClawHandler.SetRouterConfigPath(cfg.AbsConfigPath)
	return openClawHandler
}

func registerOpenClawRoutes(
	mux *http.ServeMux,
	cfg *config.Config,
	openClawHandler *handlers.OpenClawHandler,
) {
	if cfg.OpenClawEnabled && openClawHandler != nil {
		registerEnabledOpenClawRoutes(mux, openClawHandler)
		log.Printf("OpenClaw API endpoints registered: /api/openclaw/*")
		if cfg.SecurityProfile == config.DashboardSecurityProfileProduction {
			registerProductionOpenClawProxyBoundary(mux)
			log.Printf("OpenClaw embedded UI disabled by the production origin-isolation policy")
		} else {
			registerOpenClawProxyRoute(mux, openClawHandler)
			log.Printf("OpenClaw dynamic proxy configured for trusted development use: /embedded/openclaw/{name}/ (WebSocket enabled)")
		}
		return
	}

	registerDisabledOpenClawRoutes(mux)
	log.Printf("OpenClaw feature disabled")
}

// The OpenClaw image owns active HTML and JavaScript. Serving that application
// beneath the Dashboard origin would give a compromised image the ambient
// HttpOnly Dashboard session. Production therefore fails closed until the UI
// has a separately hosted origin and a short-lived, audience-scoped browser
// capability instead of the Dashboard cookie.
func registerProductionOpenClawProxyBoundary(mux *http.ServeMux) {
	mux.HandleFunc("/embedded/openclaw/", func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "OpenClaw embedded UI requires an isolated origin", http.StatusForbidden)
	})
}

func registerEnabledOpenClawRoutes(mux *http.ServeMux, openClawHandler *handlers.OpenClawHandler) {
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

func registerOpenClawProxyRoute(mux *http.ServeMux, openClawHandler *handlers.OpenClawHandler) {
	var proxyCache sync.Map // map[string]http.Handler
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

		roomID := strings.TrimSpace(r.Header.Get("X-ClawOS-Room-Id"))
		if roomID == "" {
			roomID = strings.TrimSpace(r.URL.Query().Get("roomId"))
		}
		if roomID != "" {
			r.Header.Set("X-ClawOS-Room-Id", roomID)
		}

		handler.(http.Handler).ServeHTTP(w, r)
	})
}

func registerDisabledOpenClawRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/api/openclaw/status", writeOpenClawArray)
	mux.HandleFunc("/api/openclaw/teams", writeOpenClawArray)
	mux.HandleFunc("/api/openclaw/workers", writeOpenClawArray)
	mux.HandleFunc("/api/openclaw/rooms", writeOpenClawArray)
	mux.HandleFunc("/api/openclaw/rooms/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		http.Error(w, `{"error":"OpenClaw feature disabled"}`, http.StatusServiceUnavailable)
	})
	mux.HandleFunc(
		"/embedded/openclaw/",
		serviceUnavailableHTMLHandler("OpenClaw", "OPENCLAW_ENABLED", "true"),
	)
}

func writeOpenClawArray(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`[]`))
}
