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

	// Config endpoints - MUST be registered BEFORE proxy to take precedence
	// In Go's ServeMux, exact path matches registered first take precedence over prefix handlers
	mux.HandleFunc("/api/router/config/all", handlers.ConfigHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/router/config/update", handlers.UpdateConfigHandler(cfg.AbsConfigPath))
	log.Printf("Config API endpoints registered: /api/router/config/all, /api/router/config/update")

	// Tools DB endpoint
	mux.HandleFunc("/api/tools-db", handlers.ToolsDBHandler(cfg.ConfigDir))
	log.Printf("Tools DB API endpoint registered: /api/tools-db")

	// Status endpoint - shows service health status (aligns with vllm-sr status)
	mux.HandleFunc("/api/status", handlers.StatusHandler(cfg.RouterAPIURL))
	log.Printf("Status API endpoint registered: /api/status")

	// Logs endpoint - shows service logs (aligns with vllm-sr logs)
	mux.HandleFunc("/api/logs", handlers.LogsHandler(cfg.RouterAPIURL))
	log.Printf("Logs API endpoint registered: /api/logs")

	// Router API proxy (forward Authorization) - MUST be registered before Grafana
	// Use HandleFunc to explicitly exclude config endpoints
	var routerAPIProxy *httputil.ReverseProxy
	if cfg.RouterAPIURL != "" {
		rp, err := proxy.NewReverseProxy(cfg.RouterAPIURL, "/api/router", true)
		if err != nil {
			log.Fatalf("router API proxy error: %v", err)
		}
		routerAPIProxy = rp
		// Explicitly exclude config endpoints from proxy
		mux.HandleFunc("/api/router/", func(w http.ResponseWriter, r *http.Request) {
			if strings.HasPrefix(r.URL.Path, "/api/router/config/") {
				// Config endpoints are handled by specific handlers above
				http.NotFound(w, r)
				return
			}
			rp.ServeHTTP(w, r)
		})
		log.Printf("Router API proxy configured: %s (excluding /api/router/config/*)", cfg.RouterAPIURL)
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
		mux.HandleFunc("/login", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			if grafanaStaticProxy == nil {
				w.Header().Set("Content-Type", "application/json")
				http.Error(w, `{"error":"Service not available","message":"Grafana proxy not configured"}`, http.StatusBadGateway)
				return
			}
			grafanaStaticProxy.ServeHTTP(w, r)
		})

		if grafanaStaticProxy != nil {
			log.Printf("Grafana proxy configured: %s", cfg.GrafanaURL)
			log.Printf("Grafana static assets proxied: /public/, /avatar/, /login")
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

	// Jaeger API proxy
	var jaegerAPIProxy *httputil.ReverseProxy
	var jaegerStaticProxy *httputil.ReverseProxy
	if cfg.JaegerURL != "" {
		var err error
		jaegerAPIProxy, err = proxy.NewReverseProxy(cfg.JaegerURL, "", false)
		if err != nil {
			log.Printf("Warning: failed to create Jaeger API proxy: %v", err)
			jaegerAPIProxy = nil
		}
		jaegerStaticProxy, err = proxy.NewReverseProxy(cfg.JaegerURL, "", false)
		if err != nil {
			log.Printf("Warning: failed to create Jaeger static proxy: %v", err)
			jaegerStaticProxy = nil
		}
	}

	// Smart /api/ router: route to Router API, Jaeger API, or Grafana API based on path
	mux.HandleFunc("/api/", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		// Exclude config endpoints - they're handled by specific handlers registered earlier
		if strings.HasPrefix(r.URL.Path, "/api/router/config/") {
			http.NotFound(w, r)
			return
		}

		log.Printf("API request: %s %s (from: %s)", r.Method, r.URL.Path, r.Header.Get("Referer"))

		// If path starts with /api/router/, use Router API proxy
		if strings.HasPrefix(r.URL.Path, "/api/router/") && routerAPIProxy != nil {
			log.Printf("Routing to Router API: %s", r.URL.Path)
			routerAPIProxy.ServeHTTP(w, r)
			return
		}
		// If path is Jaeger API, use Jaeger proxy
		if jaegerAPIProxy != nil && (strings.HasPrefix(r.URL.Path, "/api/services") ||
			strings.HasPrefix(r.URL.Path, "/api/traces") ||
			strings.HasPrefix(r.URL.Path, "/api/operations") ||
			strings.HasPrefix(r.URL.Path, "/api/dependencies")) {
			log.Printf("Routing to Jaeger API: %s", r.URL.Path)
			jaegerAPIProxy.ServeHTTP(w, r)
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

	// Jaeger proxy (optional)
	if cfg.JaegerURL != "" {
		jp, err := proxy.NewReverseProxy(cfg.JaegerURL, "/embedded/jaeger", false)
		if err != nil {
			log.Fatalf("jaeger proxy error: %v", err)
		}
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

		// Jaeger static assets
		if jaegerStaticProxy != nil {
			mux.HandleFunc("/static/", func(w http.ResponseWriter, r *http.Request) {
				if middleware.HandleCORSPreflight(w, r) {
					return
				}
				log.Printf("Proxying Jaeger /static/ asset: %s", r.URL.Path)
				jaegerStaticProxy.ServeHTTP(w, r)
			})
			mux.HandleFunc("/dependencies", func(w http.ResponseWriter, r *http.Request) {
				if middleware.HandleCORSPreflight(w, r) {
					return
				}
				log.Printf("Proxying Jaeger dependencies page: %s", r.URL.Path)
				jaegerStaticProxy.ServeHTTP(w, r)
			})
		}

		log.Printf("Jaeger proxy configured: %s", cfg.JaegerURL)
	} else {
		mux.HandleFunc("/embedded/jaeger/", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":"Jaeger not configured","message":"TARGET_JAEGER_URL environment variable is not set"}`))
		})
		log.Printf("Info: Jaeger URL not configured (optional)")
	}

	// Static frontend - MUST be registered last
	mux.Handle("/", handlers.StaticFileServer(cfg.StaticDir))

	return mux
}
