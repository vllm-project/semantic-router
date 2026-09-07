package router

import (
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/proxy"
	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
)

// The Referer of a request made from a page whose URL carried ?authToken= holds a live
// session token, so logging it verbatim put a working credential in stdout. See #2465.
func redactCredentialParams(raw string) string {
	if raw == "" {
		return ""
	}

	parsed, err := url.Parse(raw)
	if err != nil {
		// Say so rather than risk logging a credential.
		return "[unparsable]"
	}

	query := parsed.Query()
	changed := false
	for _, name := range []string{"authToken", "token", "access_token"} {
		if query.Has(name) {
			query.Set(name, "[REDACTED]")
			changed = true
		}
	}
	if !changed {
		// Byte for byte: this log line is how proxy routing gets debugged.
		return raw
	}

	parsed.RawQuery = query.Encode()
	return parsed.String()
}

type dashboardProxySet struct {
	envoy         *httputil.ReverseProxy
	grafanaStatic *httputil.ReverseProxy
	jaegerAPI     *httputil.ReverseProxy
	jaegerStatic  *httputil.ReverseProxy
}

func registerProxyRoutes(mux *http.ServeMux, cfg *config.Config, credentialProvider ...routerauth.CredentialProvider) {
	var provider routerauth.CredentialProvider
	if len(credentialProvider) > 0 {
		provider = credentialProvider[0]
	}
	proxies := dashboardProxySet{
		envoy: configureEnvoyProxy(cfg),
	}
	registerRouterAPIProxy(mux, cfg, proxies.envoy, provider)
	proxies.grafanaStatic = registerGrafanaRoutes(mux, cfg)
	proxies.jaegerAPI, proxies.jaegerStatic = registerJaegerRoutes(mux, cfg)

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
	originalDirector := envoyProxy.Director
	envoyProxy.Director = func(request *http.Request) {
		originalDirector(request)
		routerauth.StripBrowserCredentials(request)
		target, targetErr := resolveDynamicEnvoyTarget(cfg.EnvoyURL, cfg.AbsConfigPath)
		if targetErr != nil {
			request.URL.Scheme = ""
			request.URL.Host = ""
			return
		}
		request.URL.Scheme = target.Scheme
		request.URL.Host = target.Host
		request.Host = target.Host
	}
	log.Printf("Envoy proxy configured: %s → /api/router/v1/chat/completions", cfg.EnvoyURL)
	return envoyProxy
}

func registerRouterAPIProxy(
	mux *http.ServeMux,
	cfg *config.Config,
	envoyProxy *httputil.ReverseProxy,
	credentialProvider routerauth.CredentialProvider,
) *httputil.ReverseProxy {
	if cfg.RouterAPIURL == "" {
		return nil
	}

	// Authorization is forwarded only after the dedicated handler below has
	// stripped the browser identity and installed the managed Router identity.
	routerAPIProxy, err := proxy.NewReverseProxy(cfg.RouterAPIURL, "/api/router", true)
	if err != nil {
		log.Fatalf("router API proxy error: %v", err)
	}
	attachRouterReplayResponseRedaction(routerAPIProxy)

	mux.HandleFunc("/api/router/", func(w http.ResponseWriter, r *http.Request) {
		serveRouterAPIProxy(w, r, cfg, envoyProxy, routerAPIProxy, credentialProvider)
	})
	log.Printf("Router API proxy configured: %s (excluding /api/router/config/*)", cfg.RouterAPIURL)
	return routerAPIProxy
}

func serveRouterAPIProxy(
	w http.ResponseWriter,
	r *http.Request,
	cfg *config.Config,
	envoyProxy, routerAPIProxy *httputil.ReverseProxy,
	credentialProvider routerauth.CredentialProvider,
) {
	if cfg.ReadonlyMode && isReadonlyRouterMutation(r) {
		http.Error(w, "dashboard is read-only", http.StatusForbidden)
		return
	}
	if strings.HasPrefix(r.URL.Path, "/api/router/config/") {
		http.NotFound(w, r)
		return
	}
	if routeRouterTrafficToEnvoy(w, r, envoyProxy) || middleware.HandleCORSPreflight(w, r) {
		return
	}
	if !routerManagementProxyRouteAllowed(r.Method, r.URL.Path) {
		writeDisallowedRouterManagementResponse(w, r)
		return
	}
	if strings.HasPrefix(r.URL.Path, "/api/router/v1/router_replay") {
		// Let the proxy transport negotiate decompression so replay JSON can
		// be redacted safely for read-only Dashboard principals.
		r.Header.Del("Accept-Encoding")
	}
	if err := routerauth.RewriteAuthorization(r, credentialProvider); err != nil {
		http.Error(w, "Router management credential is unavailable", http.StatusServiceUnavailable)
		return
	}
	routerAPIProxy.ServeHTTP(w, r)
}

func isReadonlyRouterMutation(r *http.Request) bool {
	return r.Method != http.MethodGet &&
		(strings.HasPrefix(r.URL.Path, "/api/router/api/v1/response-cache/") ||
			strings.HasPrefix(r.URL.Path, "/api/router/api/v1/context-compression/"))
}

func writeDisallowedRouterManagementResponse(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		http.Error(w, "Router management mutation is not exposed by the Dashboard", http.StatusForbidden)
		return
	}
	http.NotFound(w, r)
}

func routerManagementProxyRouteAllowed(method, path string) bool {
	if method == http.MethodGet &&
		(path == "/api/router/v1/router_replay" || strings.HasPrefix(path, "/api/router/v1/router_replay/")) {
		return true
	}
	switch path {
	case "/api/router/v1/models":
		return method == http.MethodGet || method == http.MethodHead
	case "/api/router/v1/router/outcomes":
		return method == http.MethodPost
	case "/api/router/api/v1/response-cache/capabilities",
		"/api/router/api/v1/response-cache/health",
		"/api/router/api/v1/response-cache/stats",
		"/api/router/api/v1/response-cache/audit",
		"/api/router/api/v1/context-compression/capabilities",
		"/api/router/api/v1/context-compression/health",
		"/api/router/api/v1/context-compression/stats":
		return method == http.MethodGet || method == http.MethodHead
	case "/api/router/api/v1/response-cache/test",
		"/api/router/api/v1/response-cache/invalidate",
		"/api/router/api/v1/response-cache/flush",
		"/api/router/api/v1/context-compression/preview",
		"/api/router/api/v1/context-compression/recovery/invalidate":
		return method == http.MethodPost
	default:
		return false
	}
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
		if strings.HasPrefix(r.URL.Path, "/api/router/config/") {
			http.NotFound(w, r)
			return
		}

		log.Printf("API request: %s %s (from: %s)",
			r.Method, r.URL.Path, redactCredentialParams(r.Header.Get("Referer")))

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
