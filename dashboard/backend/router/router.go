package router

import (
	"log"
	"net/http"
	"net/http/httputil"
	"path/filepath"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/evaluation"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/proxy"
)

// serviceNotConfiguredHTML generates a user-friendly HTML page for unconfigured services
func serviceNotConfiguredHTML(serviceName, envVar, exampleValue string) string {
	return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>` + serviceName + ` Not Configured</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 40px;
            max-width: 480px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        .icon {
            width: 64px;
            height: 64px;
            margin: 0 auto 24px;
            background: rgba(245, 158, 11, 0.15);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .icon svg {
            width: 32px;
            height: 32px;
            stroke: #f59e0b;
        }
        h1 {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 12px;
        }
        p {
            color: #a0a0a0;
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 24px;
        }
        .config-box {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 20px;
            text-align: left;
        }
        .config-box h2 {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .config-box .hint {
            font-size: 13px;
            color: #808080;
            margin-bottom: 12px;
        }
        code {
            display: block;
            background: rgba(0, 0, 0, 0.3);
            padding: 10px 14px;
            border-radius: 6px;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 14px;
            color: #60a5fa;
            word-break: break-all;
        }
        .example {
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px dashed rgba(255, 255, 255, 0.1);
        }
        .example-label {
            font-size: 12px;
            color: #606060;
            margin-bottom: 6px;
        }
        .example code {
            font-size: 12px;
            color: #808080;
        }
        .docs-link {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 24px;
            padding: 12px 24px;
            background: rgba(96, 165, 250, 0.1);
            border: 1px solid rgba(96, 165, 250, 0.3);
            border-radius: 8px;
            color: #60a5fa;
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
        }
        .docs-link:hover {
            background: rgba(96, 165, 250, 0.2);
            border-color: rgba(96, 165, 250, 0.5);
            transform: translateY(-2px);
        }
        .docs-link svg {
            width: 16px;
            height: 16px;
        }
    </style>
</head>
<body>
    <div class="card">
        <div class="icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
        </div>
        <h1>` + serviceName + ` Not Configured</h1>
        <p>` + serviceName + ` is not configured for this dashboard. Please set the required environment variable to enable this service.</p>
        <div class="config-box">
            <h2>Configuration Required</h2>
            <p class="hint">Set the following environment variable:</p>
            <code>` + envVar + `</code>
            <div class="example">
                <p class="example-label">Example:</p>
                <code>` + envVar + `=` + exampleValue + `</code>
            </div>
        </div>
        <a href="https://vllm-semantic-router.com/docs/tutorials/observability/dashboard" target="_blank" rel="noopener noreferrer" class="docs-link">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                <polyline points="15 3 21 3 21 9"/>
                <line x1="10" y1="14" x2="21" y2="3"/>
            </svg>
            View Documentation
        </a>
    </div>
</body>
</html>`
}

// Setup configures all routes and returns the configured mux
func Setup(cfg *config.Config) *http.ServeMux {
	mux := http.NewServeMux()

	// Health check endpoint
	mux.HandleFunc("/healthz", handlers.HealthCheck)

	// Settings endpoint for frontend (readonly mode, etc.)
	mux.HandleFunc("/api/settings", handlers.SettingsHandler(cfg))

	// Config endpoints - MUST be registered BEFORE proxy to take precedence
	// In Go's ServeMux, exact path matches registered first take precedence over prefix handlers
	mux.HandleFunc("/api/router/config/all", handlers.ConfigHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/router/config/update", handlers.UpdateConfigHandler(cfg.AbsConfigPath, cfg.ReadonlyMode))
	log.Printf("Config API endpoints registered: /api/router/config/all, /api/router/config/update")

	// Router defaults endpoints (for .vllm-sr/router-defaults.yaml)
	mux.HandleFunc("/api/router/config/defaults", handlers.RouterDefaultsHandler(cfg.ConfigDir))
	mux.HandleFunc("/api/router/config/defaults/update", handlers.UpdateRouterDefaultsHandler(cfg.ConfigDir, cfg.ReadonlyMode))
	log.Printf("Router defaults API endpoints registered: /api/router/config/defaults, /api/router/config/defaults/update")

	// Tools DB endpoint
	mux.HandleFunc("/api/tools-db", handlers.ToolsDBHandler(cfg.ConfigDir))
	log.Printf("Tools DB API endpoint registered: /api/tools-db")

	// Web Search endpoint for Playground tool execution
	mux.HandleFunc("/api/tools/web-search", handlers.WebSearchHandler())
	log.Printf("Web Search API endpoint registered: /api/tools/web-search")

	// Open Web endpoint for Playground tool execution (避免CORS问题)
	mux.HandleFunc("/api/tools/open-web", handlers.OpenWebHandler())
	log.Printf("Open Web API endpoint registered: /api/tools/open-web")

	// Status endpoint - shows service health status (aligns with vllm-sr status)
	mux.HandleFunc("/api/status", handlers.StatusHandler(cfg.RouterAPIURL))
	log.Printf("Status API endpoint registered: /api/status")

	// Logs endpoint - shows service logs (aligns with vllm-sr logs)
	mux.HandleFunc("/api/logs", handlers.LogsHandler(cfg.RouterAPIURL))
	log.Printf("Logs API endpoint registered: /api/logs")

	// Topology Test Query endpoint - for testing routing decisions
	// dry-run mode calls real Router API, simulate mode uses local config
	mux.HandleFunc("/api/topology/test-query", handlers.TopologyTestQueryHandler(cfg.AbsConfigPath, cfg.RouterAPIURL))
	log.Printf("Topology Test Query API endpoint registered: /api/topology/test-query (Router API: %s)", cfg.RouterAPIURL)

	// Evaluation endpoints (if enabled)
	if cfg.EvaluationEnabled {
		// Get project root (one level up from config dir, e.g., /path/to/semantic-router-fork/config -> /path/to/semantic-router-fork)
		projectRoot := filepath.Dir(cfg.ConfigDir)

		// Initialize evaluation database
		evalDB, err := evaluation.NewDB(cfg.EvaluationDBPath)
		if err != nil {
			log.Printf("Warning: failed to initialize evaluation database: %v", err)
		} else {
			// Initialize evaluation runner
			runner := evaluation.NewRunner(evaluation.RunnerConfig{
				DB:            evalDB,
				ProjectRoot:   projectRoot,
				PythonPath:    cfg.PythonPath,
				ResultsDir:    cfg.EvaluationResultsDir,
				MaxConcurrent: 3,
			})

			// Create evaluation handler
			evalHandler := handlers.NewEvaluationHandler(evalDB, runner, cfg.ReadonlyMode)

			// Register evaluation endpoints
			// /api/evaluation/tasks - GET for list, POST for create
			mux.HandleFunc("/api/evaluation/tasks", func(w http.ResponseWriter, r *http.Request) {
				if middleware.HandleCORSPreflight(w, r) {
					return
				}
				switch r.Method {
				case http.MethodGet:
					evalHandler.ListTasksHandler().ServeHTTP(w, r)
				case http.MethodPost:
					evalHandler.CreateTaskHandler().ServeHTTP(w, r)
				default:
					http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
				}
			})
			// /api/evaluation/tasks/{id} - GET for details, DELETE for remove
			mux.HandleFunc("/api/evaluation/tasks/", func(w http.ResponseWriter, r *http.Request) {
				if middleware.HandleCORSPreflight(w, r) {
					return
				}
				switch r.Method {
				case http.MethodGet:
					evalHandler.GetTaskHandler().ServeHTTP(w, r)
				case http.MethodDelete:
					evalHandler.DeleteTaskHandler().ServeHTTP(w, r)
				default:
					http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
				}
			})
			mux.HandleFunc("/api/evaluation/run", evalHandler.RunTaskHandler())
			mux.HandleFunc("/api/evaluation/cancel/", evalHandler.CancelTaskHandler())
			mux.HandleFunc("/api/evaluation/stream/", evalHandler.StreamProgressHandler())
			mux.HandleFunc("/api/evaluation/results/", evalHandler.GetResultsHandler())
			mux.HandleFunc("/api/evaluation/export/", evalHandler.ExportResultsHandler())
			mux.HandleFunc("/api/evaluation/datasets", evalHandler.GetDatasetsHandler())
			mux.HandleFunc("/api/evaluation/history", evalHandler.GetHistoryHandler())
			log.Printf("Evaluation API endpoints registered: /api/evaluation/*")
		}
	} else {
		log.Printf("Evaluation feature disabled")
	}

	// Envoy proxy for chat completions (if configured)
	// Chat completions must go through Envoy's ext_proc pipeline
	var envoyProxy *httputil.ReverseProxy
	if cfg.EnvoyURL != "" {
		ep, err := proxy.NewReverseProxy(cfg.EnvoyURL, "", false)
		if err != nil {
			log.Fatalf("envoy proxy error: %v", err)
		}
		envoyProxy = ep
		log.Printf("Envoy proxy configured: %s → /api/router/v1/chat/completions", cfg.EnvoyURL)
	}

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
		// Route chat completions to Envoy if configured
		mux.HandleFunc("/api/router/", func(w http.ResponseWriter, r *http.Request) {
			if strings.HasPrefix(r.URL.Path, "/api/router/config/") {
				// Config endpoints are handled by specific handlers above
				http.NotFound(w, r)
				return
			}
			// Route chat completions to Envoy proxy
			if envoyProxy != nil && strings.HasPrefix(r.URL.Path, "/api/router/v1/chat/completions") {
				// Strip /api/router prefix and forward to Envoy
				r.URL.Path = strings.TrimPrefix(r.URL.Path, "/api/router")
				log.Printf("Proxying chat completions to Envoy: %s %s", r.Method, r.URL.Path)
				if middleware.HandleCORSPreflight(w, r) {
					return
				}
				envoyProxy.ServeHTTP(w, r)
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
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(serviceNotConfiguredHTML("Grafana", "TARGET_GRAFANA_URL", "http://localhost:3000")))
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
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(serviceNotConfiguredHTML("Prometheus", "TARGET_PROMETHEUS_URL", "http://localhost:9090")))
		})
		log.Printf("Warning: Prometheus URL not configured")
	}

	// Jaeger proxy (optional) - use NewJaegerProxy for dark theme injection
	if cfg.JaegerURL != "" {
		jp, err := proxy.NewJaegerProxy(cfg.JaegerURL, "/embedded/jaeger")
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
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(serviceNotConfiguredHTML("Jaeger", "TARGET_JAEGER_URL", "http://localhost:16686")))
		})
		log.Printf("Info: Jaeger URL not configured (optional)")
	}

	// Static frontend - MUST be registered last
	mux.Handle("/", handlers.StaticFileServer(cfg.StaticDir))

	return mux
}
