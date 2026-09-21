package router

import (
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/observability"
	"github.com/vllm-project/semantic-router/dashboard/backend/proxy"
	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
	"github.com/vllm-project/semantic-router/dashboard/backend/routercontract"
)

const (
	chatCompletionsPath   = "/api/router/v1/chat/completions"
	maxInferenceBodyBytes = 16 << 20
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

func registerProxyRoutes(
	routes *auth.PolicyMux,
	cfg *config.Config,
	feedbackStore playgroundFeedbackStore,
	credentialProvider ...routerauth.CredentialProvider,
) {
	var provider routerauth.CredentialProvider
	if len(credentialProvider) > 0 {
		provider = credentialProvider[0]
	}
	proxies := dashboardProxySet{
		envoy: configureEnvoyProxy(cfg),
	}
	attachPlaygroundReplayTracking(proxies.envoy, feedbackStore)
	registerRouterAPIProxy(routes, cfg, proxies.envoy, feedbackStore, provider)
	proxies.grafanaStatic = registerGrafanaRoutes(routes, cfg)
	proxies.jaegerAPI, proxies.jaegerStatic = registerJaegerRoutes(routes, cfg)

	registerObservabilityAPIRoutes(routes, proxies)
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
	log.Printf("Envoy proxy configured: %s → %s", cfg.EnvoyURL, chatCompletionsPath)
	return envoyProxy
}

func registerRouterAPIProxy(
	routes *auth.PolicyMux,
	cfg *config.Config,
	envoyProxy *httputil.ReverseProxy,
	feedbackStore playgroundFeedbackStore,
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

	routes.HandleGroup(routerGatewayContracts(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		serveRouterAPIProxy(w, r, cfg, envoyProxy, routerAPIProxy, feedbackStore, credentialProvider)
	}))
	log.Printf("Router API proxy configured: %s (excluding /api/router/config/*)", cfg.RouterAPIURL)
	return routerAPIProxy
}

// routerGatewayContracts is the exact Router surface the Dashboard exposes:
// inference through Envoy under its own permission, plus every management
// route in the shared allowlist except knowledge-base storage, which the
// classifier proxy owns.
func routerGatewayContracts() []auth.RouteContract {
	contracts := []auth.RouteContract{
		auth.ProtectedBoundedRoute(chatCompletionsPath, auth.PermInferenceRun, auth.SensitivitySecret, auth.ResourceOwnerInference, maxInferenceBodyBytes, http.MethodPost),
	}
	return append(contracts, routerManagementContracts(isGatewayProxyPath)...)
}

func serveRouterAPIProxy(
	w http.ResponseWriter,
	r *http.Request,
	cfg *config.Config,
	envoyProxy, routerAPIProxy *httputil.ReverseProxy,
	feedbackStore playgroundFeedbackStore,
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
	if r.Method == http.MethodPost && r.URL.Path == "/api/router/api/v1/observability/outcomes" && feedbackStore != nil {
		servePlaygroundOutcome(w, r, cfg.RouterAPIURL, routerAPIProxy, feedbackStore, credentialProvider)
		return
	}
	if strings.HasPrefix(r.URL.Path, "/api/router/api/v1/observability/replays") {
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
	policy, ok := routercontract.LookupManagement(r.Method, r.URL.Path)
	return ok && policy.Mutation
}

func writeDisallowedRouterManagementResponse(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		http.Error(w, "Router management mutation is not exposed by the Dashboard", http.StatusForbidden)
		return
	}
	http.NotFound(w, r)
}

func routerManagementProxyRouteAllowed(method, path string) bool {
	_, ok := routercontract.LookupManagement(method, path)
	return ok
}

func routeRouterTrafficToEnvoy(
	w http.ResponseWriter,
	r *http.Request,
	envoyProxy *httputil.ReverseProxy,
) bool {
	if envoyProxy == nil {
		return false
	}

	if r.URL.Path == chatCompletionsPath && (r.Method == http.MethodPost || r.Method == http.MethodOptions) {
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

// embeddedObservabilityRoute fronts a proxied observability UI whose own
// frontend issues reads and writes; all of them stay behind logs.read and the
// session CSRF policy.
func embeddedObservabilityRoute(pattern string) auth.RouteContract {
	return auth.ProtectedRoute(
		pattern,
		auth.PermLogsRead,
		auth.SensitivitySensitive,
		auth.ResourceOwnerObservability,
		http.MethodGet, http.MethodHead, http.MethodPost, http.MethodPut, http.MethodPatch, http.MethodDelete,
	)
}

func registerGrafanaRoutes(routes *auth.PolicyMux, cfg *config.Config) *httputil.ReverseProxy {
	if cfg.GrafanaURL == "" {
		routes.HandleFunc(
			embeddedObservabilityRoute("/embedded/grafana/"),
			serviceUnavailableHTMLHandler("Grafana", "TARGET_GRAFANA_URL", "http://localhost:3000"),
		)
		log.Printf("Warning: Grafana URL not configured")
		return nil
	}

	grafanaProxy, err := proxy.NewGrafanaProxy(cfg.GrafanaURL)
	if err != nil {
		log.Fatalf("grafana proxy error: %v", err)
	}
	routes.HandleFunc(
		auth.ProtectedRoute(proxy.GrafanaAuthScriptPath, auth.PermLogsRead, auth.SensitivityOperational, auth.ResourceOwnerObservability, http.MethodGet, http.MethodHead),
		proxy.GrafanaAuthScriptHandler,
	)
	routes.HandleFunc(embeddedObservabilityRoute("/embedded/grafana/"), func(w http.ResponseWriter, r *http.Request) {
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

// Static assets outside the API namespace carry no session data.
func registerStaticProxyRoute(
	routes *auth.PolicyMux,
	pattern string,
	staticProxy *httputil.ReverseProxy,
	message string,
) {
	routes.HandleFunc(auth.PublicRoute(pattern, http.MethodGet, http.MethodHead), func(w http.ResponseWriter, r *http.Request) {
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
			embeddedObservabilityRoute("/embedded/jaeger/"),
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
	routes.HandleGroup([]auth.RouteContract{
		embeddedObservabilityRoute("/embedded/jaeger"),
		embeddedObservabilityRoute("/embedded/jaeger/"),
	}, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		jaegerProxy.ServeHTTP(w, r)
	}))

	if jaegerStaticProxy != nil {
		routes.HandleFunc(auth.PublicRoute("/static/", http.MethodGet, http.MethodHead), func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			log.Printf("Proxying Jaeger /static/ asset: %s", r.URL.Path)
			jaegerStaticProxy.ServeHTTP(w, r)
		})
		routes.HandleFunc(auth.PublicRoute("/dependencies", http.MethodGet, http.MethodHead), func(w http.ResponseWriter, r *http.Request) {
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

// registerObservabilityAPIRoutes exposes the root-relative API calls the
// embedded Grafana and Jaeger frontends make: Grafana's data-source query and
// the Jaeger query API. Nothing else under /api/ reaches an upstream.
func registerObservabilityAPIRoutes(routes *auth.PolicyMux, proxies dashboardProxySet) {
	observabilityRead := func(pattern string) auth.RouteContract {
		return auth.ProtectedRoute(pattern, auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet)
	}
	jaegerContracts := make([]auth.RouteContract, 0, 8)
	for _, pattern := range []string{"/api/services", "/api/traces", "/api/operations", "/api/dependencies"} {
		jaegerContracts = append(jaegerContracts, observabilityRead(pattern), observabilityRead(pattern+"/"))
	}
	routes.HandleGroup(jaegerContracts, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		log.Printf("API request: %s %s (from: %s)",
			r.Method, r.URL.Path, redactCredentialParams(r.Header.Get("Referer")))
		if proxies.jaegerAPI == nil || !observability.IsJaegerAPIPath(r.URL.Path) {
			writeNoUpstreamResponse(w)
			return
		}
		log.Printf("Routing to Jaeger API: %s", r.URL.Path)
		proxies.jaegerAPI.ServeHTTP(w, r)
	}))

	routes.HandleFunc(
		auth.ProtectedBoundedRoute("/api/ds/query", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, maxRouterProbeBodyBytes, http.MethodPost),
		func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			if proxies.grafanaStatic == nil || !observability.IsGrafanaQueryPath(r.URL.Path) {
				writeNoUpstreamResponse(w)
				return
			}
			log.Printf("Routing to Grafana API: %s", r.URL.Path)
			proxies.grafanaStatic.ServeHTTP(w, r)
		},
	)
}

func writeNoUpstreamResponse(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	http.Error(w, `{"error":"Service not available","message":"No API handler configured for this path"}`, http.StatusBadGateway)
}

func registerMetricsRoutes(routes *auth.PolicyMux, cfg *config.Config) {
	routes.HandleFunc(auth.PublicRoute("/metrics/router", http.MethodGet, http.MethodHead), func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, cfg.RouterMetrics, http.StatusTemporaryRedirect)
	})
}

func registerPrometheusRoutes(routes *auth.PolicyMux, cfg *config.Config) {
	if cfg.PrometheusURL == "" {
		routes.HandleFunc(
			embeddedObservabilityRoute("/embedded/prometheus/"),
			serviceUnavailableHTMLHandler("Prometheus", "TARGET_PROMETHEUS_URL", "http://localhost:9090"),
		)
		log.Printf("Warning: Prometheus URL not configured")
		return
	}

	prometheusProxy, err := proxy.NewReverseProxy(cfg.PrometheusURL, "/embedded/prometheus", false)
	if err != nil {
		log.Fatalf("prometheus proxy error: %v", err)
	}
	routes.HandleGroup([]auth.RouteContract{
		embeddedObservabilityRoute("/embedded/prometheus"),
		embeddedObservabilityRoute("/embedded/prometheus/"),
	}, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		prometheusProxy.ServeHTTP(w, r)
	}))
	log.Printf("Prometheus proxy configured: %s", cfg.PrometheusURL)
}
