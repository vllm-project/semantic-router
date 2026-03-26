package router

import (
	"log"
	"net/http"
	"net/http/httputil"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/proxy"
)

type dashboardProxySet struct {
	envoy         *httputil.ReverseProxy
	routerAPI     *httputil.ReverseProxy
	grafanaStatic *httputil.ReverseProxy
	jaegerAPI     *httputil.ReverseProxy
	jaegerStatic  *httputil.ReverseProxy
}

func registerProxyRoutes(mux *http.ServeMux, cfg *config.Config) {
	proxies := dashboardProxySet{
		envoy: configureEnvoyProxy(cfg),
	}
	proxies.routerAPI = registerRouterAPIProxy(mux, cfg, proxies.envoy)
	proxies.grafanaStatic = registerGrafanaRoutes(mux, cfg)
	proxies.jaegerAPI, proxies.jaegerStatic = registerJaegerRoutes(mux, cfg)
	registerFleetSimRoutes(mux, cfg)

	registerSmartAPIRouter(mux, proxies)
	registerMetricsRoutes(mux, cfg)
	registerPrometheusRoutes(mux, cfg)
	registerWizMapRoutes(mux, cfg)
}

func configureEnvoyProxy(cfg *config.Config) *httputil.ReverseProxy {
	if cfg.EnvoyURL == "" {
		return nil
	}

	envoyProxy, err := proxy.NewReverseProxy(cfg.EnvoyURL, "", false)
	if err != nil {
		log.Fatalf("envoy proxy error: %v", err)
	}
	log.Printf("Envoy proxy configured: %s → /api/router/v1/chat/completions", cfg.EnvoyURL)
	return envoyProxy
}

func registerRouterAPIProxy(
	mux *http.ServeMux,
	cfg *config.Config,
	envoyProxy *httputil.ReverseProxy,
) *httputil.ReverseProxy {
	if cfg.RouterAPIURL == "" {
		return nil
	}

	routerAPIProxy, err := proxy.NewReverseProxy(cfg.RouterAPIURL, "/api/router", true)
	if err != nil {
		log.Fatalf("router API proxy error: %v", err)
	}

	mux.HandleFunc("/api/router/", func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/router/config/") {
			http.NotFound(w, r)
			return
		}
		if routeRouterTrafficToEnvoy(w, r, envoyProxy) {
			return
		}
		routerAPIProxy.ServeHTTP(w, r)
	})
	log.Printf("Router API proxy configured: %s (excluding /api/router/config/*)", cfg.RouterAPIURL)
	return routerAPIProxy
}

func routeRouterTrafficToEnvoy(
	w http.ResponseWriter,
	r *http.Request,
	envoyProxy *httputil.ReverseProxy,
) bool {
	if envoyProxy == nil {
		return false
	}

	if strings.HasPrefix(r.URL.Path, "/api/router/v1/chat/completions") {
		r.URL.Path = strings.TrimPrefix(r.URL.Path, "/api/router")
		log.Printf("Proxying chat completions to Envoy: %s %s", r.Method, r.URL.Path)
		if middleware.HandleCORSPreflight(w, r) {
			return true
		}
		envoyProxy.ServeHTTP(w, r)
		return true
	}
	if strings.HasPrefix(r.URL.Path, "/api/router/v1/router_replay") {
		r.URL.Path = strings.TrimPrefix(r.URL.Path, "/api/router")
		log.Printf("Proxying router_replay to Envoy: %s %s", r.Method, r.URL.Path)
		if middleware.HandleCORSPreflight(w, r) {
			return true
		}
		envoyProxy.ServeHTTP(w, r)
		return true
	}
	return false
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
	registerStaticProxyRoute(mux, "/login", grafanaStaticProxy, "Grafana proxy not configured")
	log.Printf("Grafana proxy configured: %s", cfg.GrafanaURL)
	log.Printf("Grafana static assets proxied: /public/, /avatar/, /login")
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
		if strings.HasPrefix(r.URL.Path, "/api/router/config/") {
			http.NotFound(w, r)
			return
		}

		log.Printf("API request: %s %s (from: %s)", r.Method, r.URL.Path, r.Header.Get("Referer"))

		if strings.HasPrefix(r.URL.Path, "/api/router/") && proxies.routerAPI != nil {
			log.Printf("Routing to Router API: %s", r.URL.Path)
			proxies.routerAPI.ServeHTTP(w, r)
			return
		}
		if proxies.jaegerAPI != nil && isJaegerAPIPath(r.URL.Path) {
			log.Printf("Routing to Jaeger API: %s", r.URL.Path)
			proxies.jaegerAPI.ServeHTTP(w, r)
			return
		}
		if proxies.grafanaStatic != nil {
			log.Printf("Routing to Grafana API: %s", r.URL.Path)
			proxies.grafanaStatic.ServeHTTP(w, r)
			return
		}

		log.Printf("No handler available for: %s", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		http.Error(w, `{"error":"Service not available","message":"No API handler configured for this path"}`, http.StatusBadGateway)
	})
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
