package router

import (
	"errors"
	"log"
	"net/http"
	"net/http/httputil"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/proxy"
	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
)

type dashboardProxySet struct {
	grafanaStatic *httputil.ReverseProxy
	jaegerAPI     *httputil.ReverseProxy
	jaegerStatic  *httputil.ReverseProxy
}

func registerProxyRoutes(
	mux *http.ServeMux,
	cfg *config.Config,
	managementSessions routerauth.ManagementSessionProvider,
) {
	proxies := dashboardProxySet{}
	registerRouterManagementProxy(mux, cfg, managementSessions)
	proxies.grafanaStatic = registerGrafanaRoutes(mux, cfg)
	proxies.jaegerAPI, proxies.jaegerStatic = registerJaegerRoutes(mux, cfg)
	registerFleetSimRoutes(mux, cfg)

	registerSmartAPIRouter(mux, proxies)
	registerMetricsRoutes(mux, cfg)
	registerPrometheusRoutes(mux, cfg)
}

func registerRouterManagementProxy(
	mux *http.ServeMux,
	cfg *config.Config,
	managementSessions routerauth.ManagementSessionProvider,
) *httputil.ReverseProxy {
	if cfg.RouterAPIURL == "" {
		return nil
	}

	managementProxy, err := proxy.NewReverseProxy(cfg.RouterAPIURL, "/api/router", true)
	if err != nil {
		log.Fatalf("Router Management proxy error: %v", err)
	}
	mux.HandleFunc("/api/router/management/v1/", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if err := routerauth.RewriteManagementAuthorization(r, managementSessions); err != nil {
			log.Printf("Router Management session acquisition failed: %v", err)
			w.Header().Set("Cache-Control", "no-store")
			var sessionErr *routerauth.ManagementSessionError
			if errors.As(err, &sessionErr) {
				if retryAfter := sessionErr.RetryAfterHeader(); retryAfter != "" {
					w.Header().Set("Retry-After", retryAfter)
				}
				http.Error(w, sessionErr.Error(), sessionErr.HTTPStatus())
				return
			}
			http.Error(w, "Router Management session is unavailable", http.StatusServiceUnavailable)
			return
		}
		managementProxy.ServeHTTP(w, r)
	})
	log.Printf("Router Management BFF configured: %s", cfg.RouterAPIURL)
	return managementProxy
}

func registerGrafanaRoutes(mux *http.ServeMux, cfg *config.Config) *httputil.ReverseProxy {
	if cfg.GrafanaURL == "" {
		mux.HandleFunc(
			"/embedded/grafana/",
			serviceUnavailableHTMLHandler("Grafana", "TARGET_GRAFANA_URL", "http://localhost:3000"),
		)
		log.Printf("Warning: Grafana URL not configured")
		return nil
	}

	grafanaProxy, err := proxy.NewReverseProxy(cfg.GrafanaURL, "/embedded/grafana", false)
	if err != nil {
		log.Fatalf("grafana proxy error: %v", err)
	}
	mux.HandleFunc("/embedded/grafana/", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		grafanaProxy.ServeHTTP(w, r)
	})

	grafanaStaticProxy, err := proxy.NewReverseProxy(cfg.GrafanaURL, "", false)
	if err != nil {
		log.Printf("Warning: failed to create Grafana static proxy: %v", err)
		log.Printf("Grafana proxy configured: %s (static proxy failed to initialize)", cfg.GrafanaURL)
		return nil
	}

	registerStaticProxyRoute(mux, "/public/", grafanaStaticProxy, "Grafana static proxy not configured")
	registerStaticProxyRoute(mux, "/avatar/", grafanaStaticProxy, "Grafana static proxy not configured")
	log.Printf("Grafana proxy configured: %s", cfg.GrafanaURL)
	log.Printf("Grafana static assets proxied: /public/, /avatar/")
	return grafanaStaticProxy
}

func registerStaticProxyRoute(
	mux *http.ServeMux,
	pattern string,
	staticProxy *httputil.ReverseProxy,
	message string,
) {
	mux.HandleFunc(pattern, func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if staticProxy == nil {
			w.Header().Set("Content-Type", "application/json")
			http.Error(w, `{"error":"Service not available","message":"`+message+`"}`, http.StatusBadGateway)
			return
		}
		staticProxy.ServeHTTP(w, r)
	})
}

func registerJaegerRoutes(
	mux *http.ServeMux,
	cfg *config.Config,
) (*httputil.ReverseProxy, *httputil.ReverseProxy) {
	if cfg.JaegerURL == "" {
		mux.HandleFunc(
			"/embedded/jaeger/",
			serviceUnavailableHTMLHandler("Jaeger", "TARGET_JAEGER_URL", "http://localhost:16686"),
		)
		log.Printf("Info: Jaeger URL not configured (optional)")
		return nil, nil
	}

	jaegerAPIProxy, err := proxy.NewReverseProxy(cfg.JaegerURL, "", false)
	if err != nil {
		log.Printf("Warning: failed to create Jaeger API proxy: %v", err)
		jaegerAPIProxy = nil
	}
	jaegerStaticProxy, err := proxy.NewReverseProxy(cfg.JaegerURL, "", false)
	if err != nil {
		log.Printf("Warning: failed to create Jaeger static proxy: %v", err)
		jaegerStaticProxy = nil
	}

	jaegerProxy, err := proxy.NewJaegerProxy(cfg.JaegerURL, "/embedded/jaeger")
	if err != nil {
		log.Fatalf("jaeger proxy error: %v", err)
	}
	mux.HandleFunc("/embedded/jaeger", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		jaegerProxy.ServeHTTP(w, r)
	})
	mux.HandleFunc("/embedded/jaeger/", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		jaegerProxy.ServeHTTP(w, r)
	})

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
	return jaegerAPIProxy, jaegerStaticProxy
}

func registerSmartAPIRouter(mux *http.ServeMux, proxies dashboardProxySet) {
	mux.HandleFunc("/api/", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		log.Printf("API request: %s %s (from: %s)", r.Method, r.URL.Path, r.Header.Get("Referer"))

		if proxies.jaegerAPI != nil && isJaegerAPIPath(r.URL.Path) {
			log.Printf("Routing to Jaeger API: %s", r.URL.Path)
			proxies.jaegerAPI.ServeHTTP(w, r)
			return
		}
		if proxies.grafanaStatic != nil && isGrafanaAPIPath(r.URL.Path) {
			log.Printf("Routing to Grafana API: %s", r.URL.Path)
			proxies.grafanaStatic.ServeHTTP(w, r)
			return
		}

		log.Printf("No handler available for: %s", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":"not_found","message":"API route not found"}`))
	})
}

func isGrafanaAPIPath(path string) bool {
	for _, prefix := range []string{
		"/api/annotations", "/api/dashboards", "/api/datasources", "/api/ds",
		"/api/folders", "/api/frontend-metrics", "/api/health", "/api/live",
		"/api/org", "/api/plugins", "/api/search", "/api/snapshots", "/api/teams",
		"/api/user", "/api/users",
	} {
		if path == prefix || strings.HasPrefix(path, prefix+"/") {
			return true
		}
	}
	return false
}

func isJaegerAPIPath(path string) bool {
	return strings.HasPrefix(path, "/api/services") ||
		strings.HasPrefix(path, "/api/traces") ||
		strings.HasPrefix(path, "/api/operations") ||
		strings.HasPrefix(path, "/api/dependencies")
}

func registerMetricsRoutes(mux *http.ServeMux, cfg *config.Config) {
	mux.HandleFunc("/metrics/router", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, cfg.RouterMetrics, http.StatusTemporaryRedirect)
	})
}

func registerPrometheusRoutes(mux *http.ServeMux, cfg *config.Config) {
	if cfg.PrometheusURL == "" {
		mux.HandleFunc(
			"/embedded/prometheus/",
			serviceUnavailableHTMLHandler("Prometheus", "TARGET_PROMETHEUS_URL", "http://localhost:9090"),
		)
		log.Printf("Warning: Prometheus URL not configured")
		return
	}

	prometheusProxy, err := proxy.NewReverseProxy(cfg.PrometheusURL, "/embedded/prometheus", false)
	if err != nil {
		log.Fatalf("prometheus proxy error: %v", err)
	}
	mux.HandleFunc("/embedded/prometheus", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		prometheusProxy.ServeHTTP(w, r)
	})
	mux.HandleFunc("/embedded/prometheus/", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		prometheusProxy.ServeHTTP(w, r)
	})
	log.Printf("Prometheus proxy configured: %s", cfg.PrometheusURL)
}

func registerFleetSimRoutes(mux *http.ServeMux, cfg *config.Config) {
	if cfg.FleetSimURL == "" {
		mux.HandleFunc("/api/fleet-sim/", func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			w.Header().Set("Content-Type", "application/json")
			http.Error(
				w,
				`{"error":"Service not available","message":"Fleet simulator is not configured"}`,
				http.StatusBadGateway,
			)
		})
		log.Printf("Info: Fleet simulator URL not configured (optional)")
		return
	}

	fleetSimProxy, err := proxy.NewReverseProxy(cfg.FleetSimURL, "/api/fleet-sim", false)
	if err != nil {
		log.Fatalf("fleet simulator proxy error: %v", err)
	}
	originalDirector := fleetSimProxy.Director
	fleetSimProxy.Director = func(r *http.Request) {
		originalDirector(r)
		r.Header.Set("X-Forwarded-Prefix", "/api/fleet-sim")
	}
	mux.HandleFunc("/api/fleet-sim/", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		fleetSimProxy.ServeHTTP(w, r)
	})
	log.Printf("Fleet simulator proxy configured: %s → /api/fleet-sim/*", cfg.FleetSimURL)
}
