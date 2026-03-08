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

type proxySet struct {
	envoy         *httputil.ReverseProxy
	routerAPI     *httputil.ReverseProxy
	grafanaStatic *httputil.ReverseProxy
	jaegerAPI     *httputil.ReverseProxy
	jaegerStatic  *httputil.ReverseProxy
}

func registerIntegrationRoutes(mux *http.ServeMux, cfg *config.Config) proxySet {
	proxies := proxySet{}
	proxies.envoy = buildReverseProxy(cfg.EnvoyURL, "", false, "envoy")
	if proxies.envoy != nil {
		log.Printf("Envoy proxy configured: %s → /api/router/v1/chat/completions", cfg.EnvoyURL)
	}

	proxies.routerAPI = registerRouterAPIProxy(mux, cfg, proxies.envoy)
	proxies.grafanaStatic = registerGrafanaRoutes(mux, cfg)
	proxies.jaegerAPI, proxies.jaegerStatic = buildJaegerAPIProxies(cfg)
	return proxies
}

func registerRouterAPIProxy(mux *http.ServeMux, cfg *config.Config, envoyProxy *httputil.ReverseProxy) *httputil.ReverseProxy {
	routerAPIProxy := buildReverseProxy(cfg.RouterAPIURL, "/api/router", true, "router API")
	if routerAPIProxy == nil {
		return nil
	}

	mux.HandleFunc("/api/router/", func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/router/config/") {
			http.NotFound(w, r)
			return
		}
		if tryServeEnvoyProxy(w, r, envoyProxy) {
			return
		}
		routerAPIProxy.ServeHTTP(w, r)
	})
	log.Printf("Router API proxy configured: %s (excluding /api/router/config/*)", cfg.RouterAPIURL)
	return routerAPIProxy
}

func tryServeEnvoyProxy(w http.ResponseWriter, r *http.Request, envoyProxy *httputil.ReverseProxy) bool {
	if envoyProxy == nil {
		return false
	}
	if !strings.HasPrefix(r.URL.Path, "/api/router/v1/chat/completions") && !strings.HasPrefix(r.URL.Path, "/api/router/v1/router_replay") {
		return false
	}

	r.URL.Path = strings.TrimPrefix(r.URL.Path, "/api/router")
	log.Printf("Proxying Envoy path: %s %s", r.Method, r.URL.Path)
	if middleware.HandleCORSPreflight(w, r) {
		return true
	}
	envoyProxy.ServeHTTP(w, r)
	return true
}

func registerGrafanaRoutes(mux *http.ServeMux, cfg *config.Config) *httputil.ReverseProxy {
	if cfg.GrafanaURL == "" {
		mux.HandleFunc("/embedded/grafana/", notConfiguredServiceHandler("Grafana", "TARGET_GRAFANA_URL", "http://localhost:3000"))
		log.Printf("Warning: Grafana URL not configured")
		return nil
	}

	embeddedProxy := buildReverseProxy(cfg.GrafanaURL, "/embedded/grafana", false, "grafana")
	mux.HandleFunc("/embedded/grafana/", withCORSPreflight(embeddedProxy))

	staticProxy := buildOptionalReverseProxy(cfg.GrafanaURL)
	registerGrafanaStaticRoutes(mux, staticProxy)
	if staticProxy != nil {
		log.Printf("Grafana proxy configured: %s", cfg.GrafanaURL)
		log.Printf("Grafana static assets proxied: /public/, /avatar/, /login")
		return staticProxy
	}

	log.Printf("Grafana proxy configured: %s (static proxy failed to initialize)", cfg.GrafanaURL)
	return nil
}

func registerGrafanaStaticRoutes(mux *http.ServeMux, grafanaStaticProxy *httputil.ReverseProxy) {
	mux.HandleFunc("/public/", grafanaStaticHandler(grafanaStaticProxy, `{"error":"Service not available","message":"Grafana static proxy not configured"}`))
	mux.HandleFunc("/avatar/", grafanaStaticHandler(grafanaStaticProxy, `{"error":"Service not available","message":"Grafana static proxy not configured"}`))
	mux.HandleFunc("/login", grafanaStaticHandler(grafanaStaticProxy, `{"error":"Service not available","message":"Grafana proxy not configured"}`))
}

func grafanaStaticHandler(grafanaStaticProxy *httputil.ReverseProxy, failureMessage string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if grafanaStaticProxy == nil {
			writeJSONResponse(w, http.StatusBadGateway, failureMessage)
			return
		}
		grafanaStaticProxy.ServeHTTP(w, r)
	}
}

func buildJaegerAPIProxies(cfg *config.Config) (*httputil.ReverseProxy, *httputil.ReverseProxy) {
	if cfg.JaegerURL == "" {
		return nil, nil
	}
	return buildOptionalReverseProxy(cfg.JaegerURL), buildOptionalReverseProxy(cfg.JaegerURL)
}

func registerSmartAPIRoutes(mux *http.ServeMux, proxies proxySet) {
	mux.HandleFunc("/api/", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/router/config/") {
			http.NotFound(w, r)
			return
		}

		log.Printf("API request: %s %s (from: %s)", r.Method, r.URL.Path, r.Header.Get("Referer"))
		switch {
		case strings.HasPrefix(r.URL.Path, "/api/router/") && proxies.routerAPI != nil:
			log.Printf("Routing to Router API: %s", r.URL.Path)
			proxies.routerAPI.ServeHTTP(w, r)
		case isJaegerAPIPath(r.URL.Path) && proxies.jaegerAPI != nil:
			log.Printf("Routing to Jaeger API: %s", r.URL.Path)
			proxies.jaegerAPI.ServeHTTP(w, r)
		case proxies.grafanaStatic != nil:
			log.Printf("Routing to Grafana API: %s", r.URL.Path)
			proxies.grafanaStatic.ServeHTTP(w, r)
		default:
			log.Printf("No handler available for: %s", r.URL.Path)
			writeJSONResponse(w, http.StatusBadGateway, `{"error":"Service not available","message":"No API handler configured for this path"}`)
		}
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
		mux.HandleFunc("/embedded/prometheus/", notConfiguredServiceHandler("Prometheus", "TARGET_PROMETHEUS_URL", "http://localhost:9090"))
		log.Printf("Warning: Prometheus URL not configured")
		return
	}

	prometheusProxy := buildReverseProxy(cfg.PrometheusURL, "/embedded/prometheus", false, "prometheus")
	mux.HandleFunc("/embedded/prometheus", withCORSPreflight(prometheusProxy))
	mux.HandleFunc("/embedded/prometheus/", withCORSPreflight(prometheusProxy))
	log.Printf("Prometheus proxy configured: %s", cfg.PrometheusURL)
}

func registerJaegerRoutes(mux *http.ServeMux, cfg *config.Config, proxies proxySet) {
	if cfg.JaegerURL == "" {
		mux.HandleFunc("/embedded/jaeger/", notConfiguredServiceHandler("Jaeger", "TARGET_JAEGER_URL", "http://localhost:16686"))
		log.Printf("Info: Jaeger URL not configured (optional)")
		return
	}

	jaegerProxy := buildJaegerProxy(cfg.JaegerURL)
	mux.HandleFunc("/embedded/jaeger", withCORSPreflight(jaegerProxy))
	mux.HandleFunc("/embedded/jaeger/", withCORSPreflight(jaegerProxy))
	registerJaegerStaticRoutes(mux, proxies.jaegerStatic)
	log.Printf("Jaeger proxy configured: %s", cfg.JaegerURL)
}

func registerJaegerStaticRoutes(mux *http.ServeMux, jaegerStaticProxy *httputil.ReverseProxy) {
	if jaegerStaticProxy == nil {
		return
	}

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

func buildReverseProxy(target, stripPrefix string, forwardAuth bool, name string) *httputil.ReverseProxy {
	if target == "" {
		return nil
	}
	reverseProxy, err := proxy.NewReverseProxy(target, stripPrefix, forwardAuth)
	if err != nil {
		log.Fatalf("%s proxy error: %v", name, err)
	}
	return reverseProxy
}

func buildOptionalReverseProxy(target string) *httputil.ReverseProxy {
	reverseProxy, err := proxy.NewReverseProxy(target, "", false)
	if err != nil {
		log.Printf("Warning: failed to create proxy for %s: %v", target, err)
		return nil
	}
	return reverseProxy
}

func buildJaegerProxy(target string) *httputil.ReverseProxy {
	jaegerProxy, err := proxy.NewJaegerProxy(target, "/embedded/jaeger")
	if err != nil {
		log.Fatalf("jaeger proxy error: %v", err)
	}
	return jaegerProxy
}
