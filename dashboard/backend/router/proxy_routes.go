package router

import (
	"context"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/observability"
	"github.com/vllm-project/semantic-router/dashboard/backend/proxy"
	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
	"github.com/vllm-project/semantic-router/dashboard/backend/routercontract"
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
	mux routeRegistrar,
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
	registerRouterAPIProxy(mux, cfg, proxies.envoy, feedbackStore, provider)
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
	mux routeRegistrar,
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

	routerHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		serveRouterAPIProxy(w, r, cfg, envoyProxy, routerAPIProxy, feedbackStore, credentialProvider)
	})
	contracts := managementRouteContracts(false)
	contracts = append(contracts, auth.ProxyMutationRoute("/api/router/v1/chat/completions", auth.PermInferenceRun, "inference.chat", auth.SensitivitySecret, auth.ResourceOwnerInference, 16<<20, http.MethodPost))
	registerRouteGroup(mux, contracts, routerHandler)
	log.Printf("Router API proxy configured: %s (excluding /api/router/config/*)", cfg.RouterAPIURL)
	return routerAPIProxy
}

func serveRouterAPIProxy(
	w http.ResponseWriter,
	r *http.Request,
	cfg *config.Config,
	envoyProxy, routerAPIProxy *httputil.ReverseProxy,
	feedbackStore playgroundFeedbackStore,
	credentialProvider routerauth.CredentialProvider,
) {
	if policy, ok := auth.RoutePolicyFromContext(r); ok && policy.Revalidate && auth.RejectRevokedMutation(w, r) {
		return
	}
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

	if r.URL.Path == "/api/router/v1/chat/completions" && (r.Method == http.MethodPost || r.Method == http.MethodOptions) {
		r.URL.Path = strings.TrimPrefix(r.URL.Path, "/api/router")
		log.Printf("Proxying chat completions to Envoy: %s %s", r.Method, r.URL.Path)
		if middleware.HandleCORSPreflight(w, r) {
			return true
		}
		if policy, ok := auth.RoutePolicyFromContext(r); ok && policy.ProxyUpstream && policy.Permission == auth.PermInferenceRun {
			proxy.ServeWithLiveAuthorization(w, r, envoyProxy, func(ctx context.Context) error {
				return auth.RevalidateRequest(r.WithContext(ctx))
			}, 250*time.Millisecond)
		} else {
			envoyProxy.ServeHTTP(w, r)
		}
		return true
	}
	return false
}

func registerGrafanaRoutes(mux routeRegistrar, cfg *config.Config) *httputil.ReverseProxy {
	if cfg.GrafanaURL == "" {
		registerRouteFunc(mux,
			auth.ProtectedRoute("/embedded/grafana/", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet),
			serviceUnavailableHTMLHandler("Grafana", "TARGET_GRAFANA_URL", "http://localhost:3000"),
		)
		log.Printf("Warning: Grafana URL not configured")
		registerRouteFunc(mux, auth.ProtectedBoundedRoute("/embedded/grafana/api/ds/query", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, 2<<20, http.MethodPost), serviceUnavailableHTMLHandler("Grafana", "TARGET_GRAFANA_URL", "http://localhost:3000"))
		return nil
	}

	grafanaProxy, err := proxy.NewGrafanaProxy(cfg.GrafanaURL)
	if err != nil {
		log.Fatalf("grafana proxy error: %v", err)
	}
	registerRouteFunc(mux, auth.ProtectedRoute(proxy.GrafanaAuthScriptPath, auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet), proxy.GrafanaAuthScriptHandler)
	grafanaHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		grafanaProxy.ServeHTTP(w, r)
	})
	registerRouteFunc(mux, auth.ProtectedRoute("/embedded/grafana/", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet), grafanaHandler)
	registerRouteFunc(mux, auth.ProtectedBoundedRoute("/embedded/grafana/api/ds/query", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, 2<<20, http.MethodPost), grafanaHandler)

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
	mux routeRegistrar,
	pattern string,
	staticProxy *httputil.ReverseProxy,
	message string,
) {
	registerRouteFunc(mux, auth.PublicRoute(pattern, http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
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
	mux routeRegistrar,
	cfg *config.Config,
) (*httputil.ReverseProxy, *httputil.ReverseProxy) {
	if cfg.JaegerURL == "" {
		registerRouteFunc(mux,
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
	registerRouteFunc(mux, auth.ProtectedRoute("/embedded/jaeger", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		jaegerProxy.ServeHTTP(w, r)
	})
	registerRouteFunc(mux, auth.ProtectedRoute("/embedded/jaeger/", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		jaegerProxy.ServeHTTP(w, r)
	})

	if jaegerStaticProxy != nil {
		registerRouteFunc(mux, auth.PublicRoute("/static/", http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
			if middleware.HandleCORSPreflight(w, r) {
				return
			}
			log.Printf("Proxying Jaeger /static/ asset: %s", r.URL.Path)
			jaegerStaticProxy.ServeHTTP(w, r)
		})
		registerRouteFunc(mux, auth.PublicRoute("/dependencies", http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
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

func registerSmartAPIRouter(mux routeRegistrar, proxies dashboardProxySet) {
	registerRouteFunc(mux, auth.ProtectedBoundedRoute("/api/ds/query", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, 2<<20, http.MethodPost), func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if proxies.grafanaStatic != nil {
			proxies.grafanaStatic.ServeHTTP(w, r)
			return
		}
		http.Error(w, "Grafana proxy is unavailable", http.StatusBadGateway)
	})
	for _, prefix := range []string{"/api/services", "/api/traces", "/api/operations", "/api/dependencies"} {
		for _, path := range []string{prefix, prefix + "/"} {
			registerRouteFunc(mux, auth.ProtectedRoute(path, auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
				if middleware.HandleCORSPreflight(w, r) {
					return
				}
				if proxies.jaegerAPI == nil || !observability.IsJaegerAPIPath(r.URL.Path) {
					http.Error(w, "Jaeger proxy is unavailable", http.StatusBadGateway)
					return
				}
				proxies.jaegerAPI.ServeHTTP(w, r)
			})
		}
	}
}

func registerMetricsRoutes(mux routeRegistrar, cfg *config.Config) {
	registerRouteFunc(mux, auth.PublicRoute("/metrics/router", http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, cfg.RouterMetrics, http.StatusTemporaryRedirect)
	})
}

func registerPrometheusRoutes(mux routeRegistrar, cfg *config.Config) {
	if cfg.PrometheusURL == "" {
		registerRouteFunc(mux,
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
	registerRouteFunc(mux, auth.ProtectedRoute("/embedded/prometheus", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		prometheusProxy.ServeHTTP(w, r)
	})
	registerRouteFunc(mux, auth.ProtectedRoute("/embedded/prometheus/", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		prometheusProxy.ServeHTTP(w, r)
	})
	log.Printf("Prometheus proxy configured: %s", cfg.PrometheusURL)
}
