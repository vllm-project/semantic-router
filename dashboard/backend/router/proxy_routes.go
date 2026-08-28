package router

import (
	"log"
	"net/http"
	"net/http/httputil"
	"strings"

	auth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/proxy"
	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
)

type dashboardProxySet struct {
	envoy         *httputil.ReverseProxy
	grafanaStatic *httputil.ReverseProxy
	jaegerAPI     *httputil.ReverseProxy
	jaegerStatic  *httputil.ReverseProxy
}

func registerProxyRoutes(routes *auth.PolicyMux, cfg *config.Config, credentialProvider ...routerauth.CredentialProvider) {
	var provider routerauth.CredentialProvider
	if len(credentialProvider) > 0 {
		provider = credentialProvider[0]
	}
	proxies := dashboardProxySet{
		envoy: configureEnvoyProxy(cfg),
	}
	registerRouterAPIProxy(routes, cfg, proxies.envoy, provider)
	proxies.grafanaStatic = registerGrafanaRoutes(routes, cfg)
	proxies.jaegerAPI, proxies.jaegerStatic = registerJaegerRoutes(routes, cfg)
	registerFleetSimRoutes(routes, cfg)

	registerSmartAPIRouter(routes, proxies)
	registerMetricsRoutes(routes, cfg)
	registerPrometheusRoutes(routes, cfg)
	registerWizMapRoutes(routes, cfg)
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
	routes *auth.PolicyMux,
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

	routerHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		serveRouterAPIProxy(w, r, cfg, routes, envoyProxy, routerAPIProxy, credentialProvider)
	})
	routes.HandleGroup(routerProxyContracts(), routerHandler)
	log.Printf("Router API proxy configured: %s (excluding /api/router/config/*)", cfg.RouterAPIURL)
	return routerAPIProxy
}

func serveRouterAPIProxy(
	w http.ResponseWriter,
	r *http.Request,
	cfg *config.Config,
	resolver auth.RoutePolicyResolver,
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
	policy, lookup := resolver.LookupRoutePolicy(r.Method, r.URL.Path)
	if lookup != auth.RouteFound || !policy.ProxyUpstream {
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

func routerProxyContracts() []auth.RouteContract {
	contracts := []auth.RouteContract{
		auth.ProxyMutationRoute(
			"/api/router/v1/chat/completions",
			auth.PermInferenceRun,
			"inference.chat",
			auth.SensitivitySecret,
			auth.ResourceOwnerInference,
			16<<20,
			http.MethodPost,
		),
		auth.ProxyRoute(
			"/api/router/v1/models",
			auth.PermConfigRead,
			auth.AuditNone,
			"",
			auth.SensitivityOperational,
			auth.ResourceOwnerInference,
			http.MethodGet,
			http.MethodHead,
		),
		auth.ProxyMutationRoute(
			"/api/router/v1/router/outcomes",
			auth.PermFeedbackSubmit,
			"feedback.submit",
			auth.SensitivitySecret,
			auth.ResourceOwnerFeedback,
			2<<20,
			http.MethodPost,
		),
		auth.ProxyRoute(
			"/api/router/v1/router_replay",
			auth.PermReplayRead,
			auth.AuditNone,
			"",
			auth.SensitivitySecret,
			auth.ResourceOwnerReplay,
			http.MethodGet,
		),
		auth.ProxyRoute(
			"/api/router/v1/router_replay/",
			auth.PermReplayRead,
			auth.AuditNone,
			"",
			auth.SensitivitySecret,
			auth.ResourceOwnerReplay,
			http.MethodGet,
		),
	}
	return append(contracts, routerManagementProxyContracts()...)
}

func routerManagementProxyContracts() []auth.RouteContract {
	contracts := make([]auth.RouteContract, 0, 12)
	for _, pattern := range []string{
		"/api/router/api/v1/response-cache/capabilities",
		"/api/router/api/v1/response-cache/health",
		"/api/router/api/v1/response-cache/stats",
		"/api/router/api/v1/response-cache/audit",
		"/api/router/api/v1/context-compression/capabilities",
		"/api/router/api/v1/context-compression/health",
		"/api/router/api/v1/context-compression/stats",
	} {
		contracts = append(contracts, auth.ProxyRoute(
			pattern,
			auth.PermConfigRead,
			auth.AuditNone,
			"",
			auth.SensitivitySensitive,
			auth.ResourceOwnerConfig,
			http.MethodGet,
			http.MethodHead,
		))
	}
	for _, route := range []struct {
		pattern    string
		permission string
		action     string
	}{
		{pattern: "/api/router/api/v1/response-cache/test", permission: auth.PermConfigWrite, action: "response_cache.test"},
		{pattern: "/api/router/api/v1/response-cache/invalidate", permission: auth.PermConfigWrite, action: "response_cache.invalidate"},
		{pattern: "/api/router/api/v1/response-cache/flush", permission: auth.PermConfigWrite, action: "response_cache.flush"},
		{pattern: "/api/router/api/v1/context-compression/preview", permission: auth.PermConfigRead, action: "context_compression.preview"},
		{pattern: "/api/router/api/v1/context-compression/recovery/invalidate", permission: auth.PermConfigWrite, action: "context_compression.invalidate"},
	} {
		contracts = append(contracts, auth.ProxyMutationRoute(
			route.pattern,
			route.permission,
			route.action,
			auth.SensitivitySensitive,
			auth.ResourceOwnerConfig,
			4<<20,
			http.MethodPost,
		))
	}
	return contracts
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

func registerGrafanaRoutes(routes *auth.PolicyMux, cfg *config.Config) *httputil.ReverseProxy {
	if cfg.GrafanaURL == "" {
		routes.HandleFunc(
			auth.ProtectedRoute("/embedded/grafana/", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet),
			serviceUnavailableHTMLHandler("Grafana", "TARGET_GRAFANA_URL", "http://localhost:3000"),
		)
		log.Printf("Warning: Grafana URL not configured")
		return nil
	}

	grafanaProxy, err := proxy.NewReverseProxy(cfg.GrafanaURL, "/embedded/grafana", false)
	if err != nil {
		log.Fatalf("grafana proxy error: %v", err)
	}
	routes.HandleFunc(auth.ProtectedRoute("/embedded/grafana/", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
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

	registerStaticProxyRoute(routes, "/public/", grafanaStaticProxy, "Grafana static proxy not configured")
	registerStaticProxyRoute(routes, "/avatar/", grafanaStaticProxy, "Grafana static proxy not configured")
	log.Printf("Grafana proxy configured: %s", cfg.GrafanaURL)
	log.Printf("Grafana static assets proxied: /public/, /avatar/")
	return grafanaStaticProxy
}

func registerStaticProxyRoute(
	routes *auth.PolicyMux,
	pattern string,
	staticProxy *httputil.ReverseProxy,
	message string,
) {
	routes.HandleFunc(auth.PublicRoute(pattern, http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
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
	routes *auth.PolicyMux,
	cfg *config.Config,
) (*httputil.ReverseProxy, *httputil.ReverseProxy) {
	if cfg.JaegerURL == "" {
		routes.HandleFunc(
			auth.ProtectedRoute("/embedded/jaeger/", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet),
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
	routes.HandleFunc(auth.ProtectedRoute("/embedded/jaeger", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		jaegerProxy.ServeHTTP(w, r)
	})
	routes.HandleFunc(auth.ProtectedRoute("/embedded/jaeger/", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		jaegerProxy.ServeHTTP(w, r)
	})

	if jaegerStaticProxy != nil {
		routes.HandleFunc(auth.PublicRoute("/static/", http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			log.Printf("Proxying Jaeger /static/ asset: %s", r.URL.Path)
			jaegerStaticProxy.ServeHTTP(w, r)
		})
		routes.HandleFunc(auth.PublicRoute("/dependencies", http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
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

func registerSmartAPIRouter(routes *auth.PolicyMux, proxies dashboardProxySet) {
	contracts := make([]auth.RouteContract, 0, 8)
	for _, pattern := range []string{"/api/services", "/api/traces", "/api/operations", "/api/dependencies"} {
		contracts = append(contracts, auth.ProtectedRoute(
			pattern,
			auth.PermLogsRead,
			auth.SensitivitySensitive,
			auth.ResourceOwnerObservability,
			http.MethodGet,
		))
		contracts = append(contracts, auth.ProtectedRoute(
			pattern+"/",
			auth.PermLogsRead,
			auth.SensitivitySensitive,
			auth.ResourceOwnerObservability,
			http.MethodGet,
		))
	}
	routes.HandleGroup(contracts, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/router/config/") {
			http.NotFound(w, r)
			return
		}

		log.Printf("API request: %s %s (from: %s)", r.Method, r.URL.Path, r.Header.Get("Referer"))

		if proxies.jaegerAPI != nil && isJaegerAPIPath(r.URL.Path) {
			log.Printf("Routing to Jaeger API: %s", r.URL.Path)
			proxies.jaegerAPI.ServeHTTP(w, r)
			return
		}
		log.Printf("No handler available for: %s", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		http.Error(w, `{"error":"Service not available","message":"No API handler configured for this path"}`, http.StatusBadGateway)
	}))
}

func isJaegerAPIPath(path string) bool {
	return strings.HasPrefix(path, "/api/services") ||
		strings.HasPrefix(path, "/api/traces") ||
		strings.HasPrefix(path, "/api/operations") ||
		strings.HasPrefix(path, "/api/dependencies")
}

func registerMetricsRoutes(routes *auth.PolicyMux, cfg *config.Config) {
	routes.HandleFunc(auth.PublicRoute("/metrics/router", http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, cfg.RouterMetrics, http.StatusTemporaryRedirect)
	})
}

func registerPrometheusRoutes(routes *auth.PolicyMux, cfg *config.Config) {
	if cfg.PrometheusURL == "" {
		routes.HandleFunc(
			auth.ProtectedRoute("/embedded/prometheus/", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet),
			serviceUnavailableHTMLHandler("Prometheus", "TARGET_PROMETHEUS_URL", "http://localhost:9090"),
		)
		log.Printf("Warning: Prometheus URL not configured")
		return
	}

	prometheusProxy, err := proxy.NewReverseProxy(cfg.PrometheusURL, "/embedded/prometheus", false)
	if err != nil {
		log.Fatalf("prometheus proxy error: %v", err)
	}
	routes.HandleFunc(auth.ProtectedRoute("/embedded/prometheus", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		prometheusProxy.ServeHTTP(w, r)
	})
	routes.HandleFunc(auth.ProtectedRoute("/embedded/prometheus/", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		prometheusProxy.ServeHTTP(w, r)
	})
	log.Printf("Prometheus proxy configured: %s", cfg.PrometheusURL)
}

func registerFleetSimRoutes(routes *auth.PolicyMux, cfg *config.Config) {
	fleetContract := auth.Route(
		"/api/fleet-sim/",
		auth.ReadPolicy(http.MethodGet, auth.PermConfigRead, auth.SensitivitySensitive, auth.ResourceOwnerConfig),
		auth.MutationPolicy(http.MethodPost, auth.PermConfigWrite, "fleet_sim.create", auth.SensitivitySensitive, auth.ResourceOwnerConfig, 4<<20),
		auth.MutationPolicy(http.MethodPut, auth.PermConfigWrite, "fleet_sim.update", auth.SensitivitySensitive, auth.ResourceOwnerConfig, 4<<20),
		auth.MutationPolicy(http.MethodDelete, auth.PermConfigWrite, "fleet_sim.delete", auth.SensitivitySensitive, auth.ResourceOwnerConfig, auth.NoBodyLimit),
	)
	if cfg.FleetSimURL == "" {
		routes.HandleFunc(fleetContract, func(w http.ResponseWriter, r *http.Request) {
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
	routes.HandleFunc(fleetContract, func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		fleetSimProxy.ServeHTTP(w, r)
	})
	log.Printf("Fleet simulator proxy configured: %s → /api/fleet-sim/*", cfg.FleetSimURL)
}
