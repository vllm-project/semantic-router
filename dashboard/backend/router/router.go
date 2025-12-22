package router

import (
	"log"
	"net/http"
	"net/http/httputil"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/proxy"
)

// Setup configures all routes and returns the configured mux
func Setup(cfg *config.Config) *http.ServeMux {
	mux := http.NewServeMux()

	// Health check endpoint
	mux.HandleFunc("/healthz", handlers.HealthCheck)

	// Config endpoints
	mux.HandleFunc("/api/router/config/all", handlers.ConfigHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/router/config/update", handlers.UpdateConfigHandler(cfg.AbsConfigPath))
	log.Printf("Config API endpoints registered: /api/router/config/all, /api/router/config/update")

	// Tools DB endpoint
	mux.HandleFunc("/api/tools-db", handlers.ToolsDBHandler(cfg.ConfigDir))
	log.Printf("Tools DB API endpoint registered: /api/tools-db")

	// Router API proxy (forward Authorization) - MUST be registered before Grafana
	var routerAPIProxy *httputil.ReverseProxy
	if cfg.RouterAPIURL != "" {
		rp, err := proxy.NewReverseProxy(cfg.RouterAPIURL, "/api/router", true)
		if err != nil {
			log.Fatalf("router API proxy error: %v", err)
		}
		routerAPIProxy = rp
		mux.Handle("/api/router/", rp)
		log.Printf("Router API proxy configured: %s", cfg.RouterAPIURL)
	}

	// Grafana proxy and static assets
	var grafanaStaticProxy *httputil.ReverseProxy
	if cfg.GrafanaURL != "" {
		gp, err := proxy.NewReverseProxy(cfg.GrafanaURL, "/embedded/grafana", false)
		if err != nil {
			log.Fatalf("grafana proxy error: %v", err)
		}
		mux.HandleFunc("/embedded/grafana/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			gp.ServeHTTP(w, r)
		})

		// Proxy for Grafana static assets (no prefix stripping)
		grafanaStaticProxy, err = proxy.NewReverseProxy(cfg.GrafanaURL, "", false)
		if err != nil {
			log.Printf("Warning: failed to create Grafana static proxy: %v", err)
			grafanaStaticProxy = nil
		}
		mux.HandleFunc("/public/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			if grafanaStaticProxy == nil {
				w.Header().Set("Content-Type", "application/json")
				http.Error(w, `{"error":"Service not available","message":"Grafana static proxy not configured"}`, http.StatusBadGateway)
				return
			}
			grafanaStaticProxy.ServeHTTP(w, r)
		})
		mux.HandleFunc("/avatar/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			if grafanaStaticProxy == nil {
				w.Header().Set("Content-Type", "application/json")
				http.Error(w, `{"error":"Service not available","message":"Grafana static proxy not configured"}`, http.StatusBadGateway)
				return
			}
			grafanaStaticProxy.ServeHTTP(w, r)
		})

		if grafanaStaticProxy != nil {
			log.Printf("Grafana proxy configured: %s", cfg.GrafanaURL)
			log.Printf("Grafana static assets proxied: /public/, /avatar/")
		} else {
			log.Printf("Grafana proxy configured: %s (static proxy failed to initialize)", cfg.GrafanaURL)
		}
	} else {
		mux.HandleFunc("/embedded/grafana/", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":"Grafana not configured","message":"TARGET_GRAFANA_URL environment variable is not set"}`))
		})
		log.Printf("Warning: Grafana URL not configured")
	}

	// OpenWebUI static proxy (needs to be set up early for the smart /api/ router below)
	var openwebuiStaticProxy *httputil.ReverseProxy
	if cfg.OpenWebUIURL != "" {
		var err error
		// Use forwardAuth=true to forward Authorization headers for API authentication
		openwebuiStaticProxy, err = proxy.NewReverseProxy(cfg.OpenWebUIURL, "", true)
		if err != nil {
			log.Printf("Warning: failed to create OpenWebUI static proxy: %v", err)
			openwebuiStaticProxy = nil
		}
	}

	// OpenWebUI is not base-path aware; the Playground iframe loads it at /workspace (root),
	// and it may redirect to /auth. These must be proxied to OpenWebUI (not served by the dashboard SPA).
	if openwebuiStaticProxy != nil {
		openwebuiRootProxy := func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			openwebuiStaticProxy.ServeHTTP(w, r)
		}
		mux.HandleFunc("/auth", openwebuiRootProxy)
		mux.HandleFunc("/auth/", openwebuiRootProxy)
		mux.HandleFunc("/workspace", openwebuiRootProxy)
		mux.HandleFunc("/workspace/", openwebuiRootProxy)
	}

	// Jaeger API proxy (needs to be set up early for the smart router below)
	var jaegerAPIProxy *httputil.ReverseProxy
	var jaegerStaticProxy *httputil.ReverseProxy
	if cfg.JaegerURL != "" {
		// Create proxy for Jaeger API (no prefix stripping for /api/*)
		var err error
		jaegerAPIProxy, err = proxy.NewReverseProxy(cfg.JaegerURL, "", false)
		if err != nil {
			log.Printf("Warning: failed to create Jaeger API proxy: %v", err)
			jaegerAPIProxy = nil
		}
		// Create proxy for Jaeger static assets (reused in handlers)
		jaegerStaticProxy, err = proxy.NewReverseProxy(cfg.JaegerURL, "", false)
		if err != nil {
			log.Printf("Warning: failed to create Jaeger static proxy: %v", err)
			jaegerStaticProxy = nil
		}
	}

	// Chat UI proxy (exposed early for smart /api routing and root-level assets)
	// Uses the same approach as Grafana to solve CORS and iframe embedding issues
	var chatUIProxy *httputil.ReverseProxy
	if cfg.ChatUIURL != "" {
		// Root-level proxy (no prefix stripping) for assets and API
		var err error
		chatUIProxy, err = proxy.NewReverseProxy(cfg.ChatUIURL, "", false)
		if err != nil {
			log.Printf("Warning: failed to create ChatUI proxy: %v", err)
			chatUIProxy = nil
		}
		// Main UI under /embedded/chatui with prefix stripping
		cup, err := proxy.NewReverseProxy(cfg.ChatUIURL, "/embedded/chatui", false)
		if err != nil {
			log.Fatalf("chatui proxy error: %v", err)
		}
		mux.HandleFunc("/embedded/chatui", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			cup.ServeHTTP(w, r)
		})
		mux.HandleFunc("/embedded/chatui/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			cup.ServeHTTP(w, r)
		})
		// Note: /_app/ is also used by OpenWebUI, so it's handled by OpenWebUI's handler
		// (registered later) which checks referer and routes to ChatUI if needed
		// SvelteKit static assets
		mux.HandleFunc("/_next/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			if chatUIProxy == nil {
				w.Header().Set("Content-Type", "application/json")
				http.Error(w, `{"error":"Service not available","message":"ChatUI proxy not configured"}`, http.StatusBadGateway)
				return
			}
			log.Printf("Proxying Chat UI Next.js asset: %s", r.URL.Path)
			chatUIProxy.ServeHTTP(w, r)
		})
		// Common web assets
		mux.HandleFunc("/favicon.ico", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			if chatUIProxy == nil {
				w.Header().Set("Content-Type", "application/json")
				http.Error(w, `{"error":"Service not available","message":"ChatUI proxy not configured"}`, http.StatusBadGateway)
				return
			}
			chatUIProxy.ServeHTTP(w, r)
		})
		mux.HandleFunc("/manifest.webmanifest", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			if chatUIProxy == nil {
				w.Header().Set("Content-Type", "application/json")
				http.Error(w, `{"error":"Service not available","message":"ChatUI proxy not configured"}`, http.StatusBadGateway)
				return
			}
			chatUIProxy.ServeHTTP(w, r)
		})
		mux.HandleFunc("/manifest.json", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			if chatUIProxy == nil {
				w.Header().Set("Content-Type", "application/json")
				http.Error(w, `{"error":"Service not available","message":"ChatUI proxy not configured"}`, http.StatusBadGateway)
				return
			}
			chatUIProxy.ServeHTTP(w, r)
		})
		mux.HandleFunc("/robots.txt", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			if chatUIProxy == nil {
				w.Header().Set("Content-Type", "application/json")
				http.Error(w, `{"error":"Service not available","message":"ChatUI proxy not configured"}`, http.StatusBadGateway)
				return
			}
			chatUIProxy.ServeHTTP(w, r)
		})

		// HuggingFace Chat UI API endpoints (these are NOT under /api/)
		// These need to be proxied when the iframe makes requests
		mux.HandleFunc("/conversation", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			if chatUIProxy == nil {
				w.Header().Set("Content-Type", "application/json")
				http.Error(w, `{"error":"Service not available","message":"ChatUI proxy not configured"}`, http.StatusBadGateway)
				return
			}
			log.Printf("Proxying Chat UI conversation API: %s %s", r.Method, r.URL.Path)
			chatUIProxy.ServeHTTP(w, r)
		})
		// Handle /conversation/{id} for sending messages to a specific conversation
		mux.HandleFunc("/conversation/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			if chatUIProxy == nil {
				w.Header().Set("Content-Type", "application/json")
				http.Error(w, `{"error":"Service not available","message":"ChatUI proxy not configured"}`, http.StatusBadGateway)
				return
			}
			log.Printf("Proxying Chat UI conversation API: %s %s", r.Method, r.URL.Path)
			chatUIProxy.ServeHTTP(w, r)
		})
		mux.HandleFunc("/conversations", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			if chatUIProxy == nil {
				w.Header().Set("Content-Type", "application/json")
				http.Error(w, `{"error":"Service not available","message":"ChatUI proxy not configured"}`, http.StatusBadGateway)
				return
			}
			log.Printf("Proxying Chat UI conversations API: %s %s", r.Method, r.URL.Path)
			chatUIProxy.ServeHTTP(w, r)
		})
		mux.HandleFunc("/conversations/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			if chatUIProxy == nil {
				w.Header().Set("Content-Type", "application/json")
				http.Error(w, `{"error":"Service not available","message":"ChatUI proxy not configured"}`, http.StatusBadGateway)
				return
			}
			log.Printf("Proxying Chat UI conversations API: %s %s", r.Method, r.URL.Path)
			chatUIProxy.ServeHTTP(w, r)
		})
		mux.HandleFunc("/settings", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			if chatUIProxy == nil {
				w.Header().Set("Content-Type", "application/json")
				http.Error(w, `{"error":"Service not available","message":"ChatUI proxy not configured"}`, http.StatusBadGateway)
				return
			}
			log.Printf("Proxying Chat UI settings: %s %s", r.Method, r.URL.Path)
			chatUIProxy.ServeHTTP(w, r)
		})
		mux.HandleFunc("/settings/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			if chatUIProxy == nil {
				w.Header().Set("Content-Type", "application/json")
				http.Error(w, `{"error":"Service not available","message":"ChatUI proxy not configured"}`, http.StatusBadGateway)
				return
			}
			log.Printf("Proxying Chat UI settings: %s %s", r.Method, r.URL.Path)
			chatUIProxy.ServeHTTP(w, r)
		})
		mux.HandleFunc("/login", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			// Check if this is a Grafana login request by:
			// 1. Query parameter redirectTo with "goto" (GET requests)
			// 2. Referer header containing "/embedded/grafana" or "/monitoring"
			// 3. Content-Type: application/json (Grafana uses JSON for login POST)
			redirectTo := r.URL.Query().Get("redirectTo")
			referer := r.Header.Get("Referer")
			contentType := r.Header.Get("Content-Type")

			isGrafanaRequest := (redirectTo != "" && strings.Contains(redirectTo, "goto")) ||
				strings.Contains(referer, "/embedded/grafana") ||
				strings.Contains(referer, "/monitoring") ||
				strings.Contains(contentType, "application/json")

			if isGrafanaRequest && grafanaStaticProxy != nil {
				log.Printf("Proxying Grafana login: %s %s (redirectTo=%s, referer=%s, contentType=%s)", r.Method, r.URL.Path, redirectTo, referer, contentType)
				grafanaStaticProxy.ServeHTTP(w, r)
				return
			}
			if chatUIProxy == nil {
				w.Header().Set("Content-Type", "application/json")
				http.Error(w, `{"error":"Service not available","message":"ChatUI proxy not configured"}`, http.StatusBadGateway)
				return
			}
			log.Printf("Proxying Chat UI login: %s %s (contentType=%s)", r.Method, r.URL.Path, contentType)
			chatUIProxy.ServeHTTP(w, r)
		})
		mux.HandleFunc("/logout", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			if chatUIProxy == nil {
				w.Header().Set("Content-Type", "application/json")
				http.Error(w, `{"error":"Service not available","message":"ChatUI proxy not configured"}`, http.StatusBadGateway)
				return
			}
			log.Printf("Proxying Chat UI logout: %s %s", r.Method, r.URL.Path)
			chatUIProxy.ServeHTTP(w, r)
		})
		// Shared conversation routes
		mux.HandleFunc("/r/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			if chatUIProxy == nil {
				w.Header().Set("Content-Type", "application/json")
				http.Error(w, `{"error":"Service not available","message":"ChatUI proxy not configured"}`, http.StatusBadGateway)
				return
			}
			log.Printf("Proxying Chat UI shared conversation: %s %s", r.Method, r.URL.Path)
			chatUIProxy.ServeHTTP(w, r)
		})
		// Chat UI assets folder (logo, images, etc.)
		mux.HandleFunc("/chatui/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			if chatUIProxy == nil {
				w.Header().Set("Content-Type", "application/json")
				http.Error(w, `{"error":"Service not available","message":"ChatUI proxy not configured"}`, http.StatusBadGateway)
				return
			}
			log.Printf("Proxying Chat UI assets: %s", r.URL.Path)
			chatUIProxy.ServeHTTP(w, r)
		})

		log.Printf("HuggingChat proxy configured: %s → /embedded/chatui/", cfg.ChatUIURL)
		log.Printf("HuggingChat API routes: /conversation, /conversations, /settings, /login, /logout, /r/")
		log.Printf("HuggingChat assets proxied at: /_app/, /chatui/, /favicon.ico, /manifest.webmanifest, /robots.txt")
	} else {
		mux.HandleFunc("/embedded/chatui/", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":"HuggingChat not configured","message":"TARGET_CHATUI_URL environment variable is not set"}`))
		})
		log.Printf("Info: HuggingChat not configured (optional)")
	}

	// Smart /api/ router: route to Router API, Jaeger API, Chat UI API, or Grafana API based on path
	mux.HandleFunc("/api/", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		// Log all API requests for debugging
		log.Printf("API request: %s %s (from: %s)", r.Method, r.URL.Path, r.Header.Get("Referer"))

		// If path starts with /api/router/, use Router API proxy
		if strings.HasPrefix(r.URL.Path, "/api/router/") && routerAPIProxy != nil {
			log.Printf("Routing to Router API: %s", r.URL.Path)
			routerAPIProxy.ServeHTTP(w, r)
			return
		}
		// If path is Jaeger API (services, traces, operations, dependencies, etc.), use Jaeger proxy
		if jaegerAPIProxy != nil && (strings.HasPrefix(r.URL.Path, "/api/services") ||
			strings.HasPrefix(r.URL.Path, "/api/traces") ||
			strings.HasPrefix(r.URL.Path, "/api/operations") ||
			strings.HasPrefix(r.URL.Path, "/api/dependencies")) {
			log.Printf("Routing to Jaeger API: %s", r.URL.Path)
			jaegerAPIProxy.ServeHTTP(w, r)
			return
		}
		// Check if request is from OpenWebUI (by referer) and route to OpenWebUI API
		referer := r.Header.Get("Referer")
		if openwebuiStaticProxy != nil && referer != "" && strings.Contains(referer, "/embedded/openwebui") {
			log.Printf("Routing to OpenWebUI API: %s (referer: %s)", r.URL.Path, referer)
			openwebuiStaticProxy.ServeHTTP(w, r)
			return
		}
		// Check if path is a known OpenWebUI API endpoint (even without referer)
		// OpenWebUI uses /api/config for configuration and /api/v1/ for auth/users
		if openwebuiStaticProxy != nil && (strings.HasPrefix(r.URL.Path, "/api/config") ||
			strings.HasPrefix(r.URL.Path, "/api/v1/")) {
			log.Printf("Routing to OpenWebUI API: %s (by path pattern)", r.URL.Path)
			openwebuiStaticProxy.ServeHTTP(w, r)
			return
		}
		// Prefer Chat UI API when available (to avoid returning HTML from other backends)
		if chatUIProxy != nil {
			log.Printf("Routing to Chat UI API: %s", r.URL.Path)
			chatUIProxy.ServeHTTP(w, r)
			return
		}
		// Otherwise, if Grafana is configured, proxy to Grafana API
		if grafanaStaticProxy != nil {
			log.Printf("Routing to Grafana API: %s", r.URL.Path)
			grafanaStaticProxy.ServeHTTP(w, r)
			return
		}
		// No handler available
		log.Printf("No handler available for: %s", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		http.Error(w, `{"error":"Service not available","message":"No API handler configured for this path"}`, http.StatusBadGateway)
	})

	// Router metrics passthrough
	mux.HandleFunc("/metrics/router", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, cfg.RouterMetrics, http.StatusTemporaryRedirect)
	})

	// Prometheus proxy (optional)
	if cfg.PrometheusURL != "" {
		pp, err := proxy.NewReverseProxy(cfg.PrometheusURL, "/embedded/prometheus", false)
		if err != nil {
			log.Fatalf("prometheus proxy error: %v", err)
		}
		mux.HandleFunc("/embedded/prometheus", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			pp.ServeHTTP(w, r)
		})
		mux.HandleFunc("/embedded/prometheus/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			pp.ServeHTTP(w, r)
		})
		log.Printf("Prometheus proxy configured: %s", cfg.PrometheusURL)
	} else {
		mux.HandleFunc("/embedded/prometheus/", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":"Prometheus not configured","message":"TARGET_PROMETHEUS_URL environment variable is not set"}`))
		})
		log.Printf("Warning: Prometheus URL not configured")
	}

	// Jaeger proxy (optional) - expose full UI under /embedded/jaeger and its static assets under /static/
	if cfg.JaegerURL != "" {
		jp, err := proxy.NewReverseProxy(cfg.JaegerURL, "/embedded/jaeger", false)
		if err != nil {
			log.Fatalf("jaeger proxy error: %v", err)
		}
		// Jaeger UI (root UI under /embedded/jaeger)
		mux.HandleFunc("/embedded/jaeger", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			jp.ServeHTTP(w, r)
		})
		mux.HandleFunc("/embedded/jaeger/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			jp.ServeHTTP(w, r)
		})

		// Jaeger static assets are typically served under /static/* from the same origin
		// Note: /static/ is shared with OpenWebUI, so we handle it in OpenWebUI section with referer-based routing

		// Jaeger /dependencies page (accessible directly, not under /embedded/jaeger)
		// Use the pre-created jaegerStaticProxy
		if jaegerStaticProxy != nil {
			mux.HandleFunc("/dependencies", func(w http.ResponseWriter, r *http.Request) {
				if middleware.HandleCORSPreflight(w, r) {
					return
				}
				log.Printf("Proxying Jaeger dependencies page: %s", r.URL.Path)
				jaegerStaticProxy.ServeHTTP(w, r)
			})
		}

		log.Printf("Jaeger proxy configured: %s; static assets proxied at /static/, /dependencies", cfg.JaegerURL)
	} else {
		mux.HandleFunc("/embedded/jaeger/", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":"Jaeger not configured","message":"TARGET_JAEGER_URL environment variable is not set"}`))
		})
		log.Printf("Info: Jaeger URL not configured (optional)")
	}

	// Open WebUI proxy (optional) - MUST handle /static/ with referer-based routing for Jaeger compatibility
	if cfg.OpenWebUIURL != "" {
		op, err := proxy.NewReverseProxy(cfg.OpenWebUIURL, "/embedded/openwebui", true)
		if err != nil {
			log.Fatalf("openwebui proxy error: %v", err)
		}
		mux.HandleFunc("/embedded/openwebui", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			op.ServeHTTP(w, r)
		})
		mux.HandleFunc("/embedded/openwebui/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			op.ServeHTTP(w, r)
		})

		// Static assets for OpenWebUI and Jaeger - route based on referer
		mux.HandleFunc("/static/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			// Check referer to determine if request is from Jaeger or OpenWebUI
			referer := r.Header.Get("Referer")
			if referer != "" && strings.Contains(referer, "/embedded/jaeger") {
				// Route to Jaeger if referer indicates Jaeger
				if jaegerStaticProxy != nil {
					log.Printf("Proxying Jaeger /static/ asset: %s (referer: %s)", r.URL.Path, referer)
					jaegerStaticProxy.ServeHTTP(w, r)
					return
				}
			}
			// Default to OpenWebUI when it's configured
			if openwebuiStaticProxy != nil {
				log.Printf("Proxying OpenWebUI /static/ asset: %s (referer: %s)", r.URL.Path, referer)
				openwebuiStaticProxy.ServeHTTP(w, r)
			} else {
				w.Header().Set("Content-Type", "application/json")
				http.Error(w, `{"error":"Service not available","message":"OpenWebUI static proxy not configured"}`, http.StatusBadGateway)
			}
		})

		// OpenWebUI also uses /_app/ for its main JS/CSS bundles
		mux.HandleFunc("/_app/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			// Check Referer to determine if request is from OpenWebUI or ChatUI
			referer := r.Header.Get("Referer")
			isOpenWebUIRequest := referer != "" && strings.Contains(referer, "/embedded/openwebui")
			isChatUIRequest := referer != "" && strings.Contains(referer, "/embedded/chatui")

			// If referer indicates OpenWebUI, route to OpenWebUI
			if isOpenWebUIRequest && openwebuiStaticProxy != nil {
				log.Printf("Proxying OpenWebUI /_app/ asset: %s (referer: %s)", r.URL.Path, referer)
				openwebuiStaticProxy.ServeHTTP(w, r)
				return
			}
			// If referer indicates ChatUI, route to ChatUI (if configured)
			if isChatUIRequest && chatUIProxy != nil {
				log.Printf("Proxying Chat UI /_app/ asset: %s (referer: %s)", r.URL.Path, referer)
				chatUIProxy.ServeHTTP(w, r)
				return
			}
			// If no referer or unclear, try OpenWebUI first (since it's configured)
			if openwebuiStaticProxy != nil {
				log.Printf("Proxying /_app/ asset to OpenWebUI (no clear referer): %s (referer: %s)", r.URL.Path, referer)
				openwebuiStaticProxy.ServeHTTP(w, r)
			} else {
				w.Header().Set("Content-Type", "application/json")
				http.Error(w, `{"error":"Service not available","message":"No handler available for /_app/"}`, http.StatusBadGateway)
			}
		})

		log.Printf("Open WebUI proxy configured: %s", cfg.OpenWebUIURL)
		if openwebuiStaticProxy != nil {
			log.Printf("Open WebUI static assets proxied at: /static/, /_app/")
		} else {
			log.Printf("Open WebUI static assets proxy failed to initialize")
		}
	} else {
		mux.HandleFunc("/embedded/openwebui/", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":"Open WebUI not configured","message":"TARGET_OPENWEBUI_URL environment variable is not set or empty"}`))
		})
		log.Printf("Info: Open WebUI not configured (optional)")
	}

	// Static frontend - MUST be registered last
	mux.Handle("/", handlers.StaticFileServer(cfg.StaticDir))

	return mux
}
