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
		grafanaStaticProxy, _ = proxy.NewReverseProxy(cfg.GrafanaURL, "", false)
		mux.HandleFunc("/public/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			grafanaStaticProxy.ServeHTTP(w, r)
		})
		mux.HandleFunc("/avatar/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			grafanaStaticProxy.ServeHTTP(w, r)
		})

		log.Printf("Grafana proxy configured: %s", cfg.GrafanaURL)
		log.Printf("Grafana static assets proxied: /public/, /avatar/")
	} else {
		mux.HandleFunc("/embedded/grafana/", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			w.Write([]byte(`{"error":"Grafana not configured","message":"TARGET_GRAFANA_URL environment variable is not set"}`))
		})
		log.Printf("Warning: Grafana URL not configured")
	}

	// Jaeger API proxy (needs to be set up early for the smart router below)
	var jaegerAPIProxy *httputil.ReverseProxy
	if cfg.JaegerURL != "" {
		// Create proxy for Jaeger API (no prefix stripping for /api/*)
		jaegerAPIProxy, _ = proxy.NewReverseProxy(cfg.JaegerURL, "", false)
	}

	// Chat UI proxy (exposed early for smart /api routing and root-level assets)
	// Uses the same approach as Grafana to solve CORS and iframe embedding issues
	var chatUIProxy *httputil.ReverseProxy
	if cfg.ChatUIURL != "" {
		// Root-level proxy (no prefix stripping) for assets and API
		chatUIProxy, _ = proxy.NewReverseProxy(cfg.ChatUIURL, "", false)
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
		// Static assets commonly used by HF Chat UI (SvelteKit/Next)
		mux.HandleFunc("/_app/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			log.Printf("Proxying Chat UI asset: %s", r.URL.Path)
			chatUIProxy.ServeHTTP(w, r)
		})
		// SvelteKit static assets
		mux.HandleFunc("/_next/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
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
			chatUIProxy.ServeHTTP(w, r)
		})
		mux.HandleFunc("/manifest.webmanifest", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			chatUIProxy.ServeHTTP(w, r)
		})
		mux.HandleFunc("/manifest.json", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			chatUIProxy.ServeHTTP(w, r)
		})
		mux.HandleFunc("/robots.txt", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
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
			log.Printf("Proxying Chat UI conversation API: %s %s", r.Method, r.URL.Path)
			chatUIProxy.ServeHTTP(w, r)
		})
		// Handle /conversation/{id} for sending messages to a specific conversation
		mux.HandleFunc("/conversation/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			log.Printf("Proxying Chat UI conversation API: %s %s", r.Method, r.URL.Path)
			chatUIProxy.ServeHTTP(w, r)
		})
		mux.HandleFunc("/conversations", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			log.Printf("Proxying Chat UI conversations API: %s %s", r.Method, r.URL.Path)
			chatUIProxy.ServeHTTP(w, r)
		})
		mux.HandleFunc("/conversations/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			log.Printf("Proxying Chat UI conversations API: %s %s", r.Method, r.URL.Path)
			chatUIProxy.ServeHTTP(w, r)
		})
		mux.HandleFunc("/settings", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			log.Printf("Proxying Chat UI settings: %s %s", r.Method, r.URL.Path)
			chatUIProxy.ServeHTTP(w, r)
		})
		mux.HandleFunc("/settings/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			log.Printf("Proxying Chat UI settings: %s %s", r.Method, r.URL.Path)
			chatUIProxy.ServeHTTP(w, r)
		})
		mux.HandleFunc("/login", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			log.Printf("Proxying Chat UI login: %s %s", r.Method, r.URL.Path)
			chatUIProxy.ServeHTTP(w, r)
		})
		mux.HandleFunc("/logout", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
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
			log.Printf("Proxying Chat UI shared conversation: %s %s", r.Method, r.URL.Path)
			chatUIProxy.ServeHTTP(w, r)
		})
		// Chat UI assets folder (logo, images, etc.)
		mux.HandleFunc("/chatui/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
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
			w.Write([]byte(`{"error":"HuggingChat not configured","message":"TARGET_CHATUI_URL environment variable is not set"}`))
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
		// If path is Jaeger API (services, traces, operations, etc.), use Jaeger proxy
		if jaegerAPIProxy != nil && (strings.HasPrefix(r.URL.Path, "/api/services") ||
			strings.HasPrefix(r.URL.Path, "/api/traces") ||
			strings.HasPrefix(r.URL.Path, "/api/operations")) {
			log.Printf("Routing to Jaeger API: %s", r.URL.Path)
			jaegerAPIProxy.ServeHTTP(w, r)
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
			w.Write([]byte(`{"error":"Prometheus not configured","message":"TARGET_PROMETHEUS_URL environment variable is not set"}`))
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
		// Provide a passthrough proxy without prefix stripping
		jStatic, _ := proxy.NewReverseProxy(cfg.JaegerURL, "", false)
		mux.Handle("/static/", jStatic)

		log.Printf("Jaeger proxy configured: %s; static assets proxied at /static/", cfg.JaegerURL)
	} else {
		mux.HandleFunc("/embedded/jaeger/", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			w.Write([]byte(`{"error":"Jaeger not configured","message":"TARGET_JAEGER_URL environment variable is not set"}`))
		})
		log.Printf("Info: Jaeger URL not configured (optional)")
	}

	// Open WebUI proxy (optional)
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
		log.Printf("Open WebUI proxy configured: %s", cfg.OpenWebUIURL)
	} else {
		mux.HandleFunc("/embedded/openwebui/", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			w.Write([]byte(`{"error":"Open WebUI not configured","message":"TARGET_OPENWEBUI_URL environment variable is not set or empty"}`))
		})
		log.Printf("Info: Open WebUI not configured (optional)")
	}

	// Static frontend - MUST be registered last
	mux.Handle("/", handlers.StaticFileServer(cfg.StaticDir))

	return mux
}
