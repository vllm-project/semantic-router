package router

import (
	"log"
	"net/http"
	"net/http/httputil"
	"strings"

	backendapp "github.com/vllm-project/semantic-router/dashboard/backend/app"
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

func registerIntegrationRoutes(mux *http.ServeMux, app *backendapp.App) proxySet {
	cfg := app.Config
	access := newRouteAccess(app)
	proxies := proxySet{}
	proxies.envoy = buildReverseProxy(cfg.EnvoyURL, "", false, "envoy")
	if proxies.envoy != nil {
		log.Printf("Envoy proxy configured: %s → /api/router/v1/chat/completions", cfg.EnvoyURL)
	}

	proxies.routerAPI = registerRouterAPIProxy(mux, cfg, access, proxies.envoy)
	proxies.grafanaStatic = registerGrafanaRoutes(mux, app, access)
	proxies.jaegerAPI, proxies.jaegerStatic = buildJaegerAPIProxies(cfg)
	return proxies
}

func registerRouterAPIProxy(mux *http.ServeMux, cfg *config.Config, access routeAccess, envoyProxy *httputil.ReverseProxy) *httputil.ReverseProxy {
	routerAPIProxy := buildReverseProxy(cfg.RouterAPIURL, "/api/router", cfg.ProxyForwardAuth, "router API")
	if routerAPIProxy == nil {
		return nil
	}

	mux.Handle("/api/router/", access.viewer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rejectCrossOriginProxyAccess(w, r) {
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/router/config/") {
			http.NotFound(w, r)
			return
		}
		if tryServeEnvoyProxy(w, r, envoyProxy) {
			return
		}
		routerAPIProxy.ServeHTTP(w, r)
	})))
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

func registerGrafanaRoutes(mux *http.ServeMux, app *backendapp.App, access routeAccess) *httputil.ReverseProxy {
	cfg := app.Config
	if cfg.GrafanaURL == "" {
		mux.Handle("/embedded/grafana/", access.viewer(notConfiguredServiceHandler("Grafana", "TARGET_GRAFANA_URL", "http://localhost:3000")))
		log.Printf("Warning: Grafana URL not configured")
		return nil
	}

	embeddedProxy := buildReverseProxy(cfg.GrafanaURL, "/embedded/grafana", false, "grafana")
	mux.Handle("/embedded/grafana/", access.viewer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rejectCrossOriginProxyAccess(w, r) {
			return
		}
		if r.URL.Path == "/embedded/grafana" || r.URL.Path == "/embedded/grafana/" {
			appendEmbeddedServiceAudit(app, r, "grafana", "grafana")
		}
		withCORSPreflight(embeddedProxy).ServeHTTP(w, r)
	})))

	staticProxy := buildOptionalReverseProxy(cfg.GrafanaURL)
	registerGrafanaStaticRoutes(mux, access, staticProxy)
	if staticProxy != nil {
		log.Printf("Grafana proxy configured: %s", cfg.GrafanaURL)
		log.Printf("Grafana static assets proxied: /public/, /avatar/, /login")
		return staticProxy
	}

	log.Printf("Grafana proxy configured: %s (static proxy failed to initialize)", cfg.GrafanaURL)
	return nil
}

func registerGrafanaStaticRoutes(mux *http.ServeMux, access routeAccess, grafanaStaticProxy *httputil.ReverseProxy) {
	mux.Handle("/public/", access.viewer(grafanaStaticHandler(grafanaStaticProxy, `{"error":"Service not available","message":"Grafana static proxy not configured"}`)))
	mux.Handle("/avatar/", access.viewer(grafanaStaticHandler(grafanaStaticProxy, `{"error":"Service not available","message":"Grafana static proxy not configured"}`)))
	mux.Handle("/login", access.viewer(grafanaStaticHandler(grafanaStaticProxy, `{"error":"Service not available","message":"Grafana proxy not configured"}`)))
}

func grafanaStaticHandler(grafanaStaticProxy *httputil.ReverseProxy, failureMessage string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rejectCrossOriginProxyAccess(w, r) {
			return
		}
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if grafanaStaticProxy == nil {
			writeJSONResponse(w, http.StatusBadGateway, failureMessage)
			return
		}
		grafanaStaticProxy.ServeHTTP(w, r)
	})
}

func buildJaegerAPIProxies(cfg *config.Config) (*httputil.ReverseProxy, *httputil.ReverseProxy) {
	if cfg.JaegerURL == "" {
		return nil, nil
	}
	return buildOptionalReverseProxy(cfg.JaegerURL), buildOptionalReverseProxy(cfg.JaegerURL)
}

func registerSmartAPIRoutes(mux *http.ServeMux, access routeAccess, proxies proxySet) {
	mux.Handle("/api/", access.viewer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rejectCrossOriginProxyAccess(w, r) {
			return
		}
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
	})))
}

func isJaegerAPIPath(path string) bool {
	return strings.HasPrefix(path, "/api/services") ||
		strings.HasPrefix(path, "/api/traces") ||
		strings.HasPrefix(path, "/api/operations") ||
		strings.HasPrefix(path, "/api/dependencies")
}

func registerMetricsRoutes(mux *http.ServeMux, access routeAccess, cfg *config.Config) {
	mux.Handle("/metrics/router", access.viewer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, cfg.RouterMetrics, http.StatusTemporaryRedirect)
	})))
}

func registerPrometheusRoutes(mux *http.ServeMux, app *backendapp.App, access routeAccess) {
	cfg := app.Config
	if cfg.PrometheusURL == "" {
		mux.Handle("/embedded/prometheus/", access.viewer(notConfiguredServiceHandler("Prometheus", "TARGET_PROMETHEUS_URL", "http://localhost:9090")))
		log.Printf("Warning: Prometheus URL not configured")
		return
	}

	prometheusProxy := buildReverseProxy(cfg.PrometheusURL, "/embedded/prometheus", false, "prometheus")
	protectedPrometheus := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rejectCrossOriginProxyAccess(w, r) {
			return
		}
		if r.URL.Path == "/embedded/prometheus" || r.URL.Path == "/embedded/prometheus/" {
			appendEmbeddedServiceAudit(app, r, "prometheus", "prometheus")
		}
		withCORSPreflight(prometheusProxy).ServeHTTP(w, r)
	})
	mux.Handle("/embedded/prometheus", access.viewer(protectedPrometheus))
	mux.Handle("/embedded/prometheus/", access.viewer(protectedPrometheus))
	log.Printf("Prometheus proxy configured: %s", cfg.PrometheusURL)
}

func registerJaegerRoutes(mux *http.ServeMux, app *backendapp.App, access routeAccess, proxies proxySet) {
	cfg := app.Config
	if cfg.JaegerURL == "" {
		mux.Handle("/embedded/jaeger/", access.viewer(notConfiguredServiceHandler("Jaeger", "TARGET_JAEGER_URL", "http://localhost:16686")))
		log.Printf("Info: Jaeger URL not configured (optional)")
		return
	}

	jaegerProxy := buildJaegerProxy(cfg.JaegerURL)
	protectedJaeger := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rejectCrossOriginProxyAccess(w, r) {
			return
		}
		if r.URL.Path == "/embedded/jaeger" || r.URL.Path == "/embedded/jaeger/" {
			appendEmbeddedServiceAudit(app, r, "jaeger", "jaeger")
		}
		withCORSPreflight(jaegerProxy).ServeHTTP(w, r)
	})
	mux.Handle("/embedded/jaeger", access.viewer(protectedJaeger))
	mux.Handle("/embedded/jaeger/", access.viewer(protectedJaeger))
	registerJaegerStaticRoutes(mux, access, proxies.jaegerStatic)
	log.Printf("Jaeger proxy configured: %s", cfg.JaegerURL)
}

func registerJaegerStaticRoutes(mux *http.ServeMux, access routeAccess, jaegerStaticProxy *httputil.ReverseProxy) {
	if jaegerStaticProxy == nil {
		return
	}

	mux.Handle("/static/", access.viewer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rejectCrossOriginProxyAccess(w, r) {
			return
		}
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		log.Printf("Proxying Jaeger /static/ asset: %s", r.URL.Path)
		jaegerStaticProxy.ServeHTTP(w, r)
	})))
	mux.Handle("/dependencies", access.viewer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rejectCrossOriginProxyAccess(w, r) {
			return
		}
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		log.Printf("Proxying Jaeger dependencies page: %s", r.URL.Path)
		jaegerStaticProxy.ServeHTTP(w, r)
	})))
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
