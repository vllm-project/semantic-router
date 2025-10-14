package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"

	yaml "gopkg.in/yaml.v3"
)

// env returns the env var or default
func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// configHandler reads and serves the config.yaml file as JSON
func configHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Only allow GET requests
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Read the config file
		data, err := os.ReadFile(configPath)
		if err != nil {
			log.Printf("Error reading config file: %v", err)
			http.Error(w, fmt.Sprintf("Failed to read config file: %v", err), http.StatusInternalServerError)
			return
		}

		// Parse YAML
		var config interface{}
		if err := yaml.Unmarshal(data, &config); err != nil {
			log.Printf("Error parsing config YAML: %v", err)
			http.Error(w, fmt.Sprintf("Failed to parse config: %v", err), http.StatusInternalServerError)
			return
		}

		// Convert to JSON and send response
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(config); err != nil {
			log.Printf("Error encoding config to JSON: %v", err)
			http.Error(w, fmt.Sprintf("Failed to encode config: %v", err), http.StatusInternalServerError)
			return
		}
	}
}

// updateConfigHandler updates the config.yaml file
func updateConfigHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Only allow POST/PUT requests
		if r.Method != http.MethodPost && r.Method != http.MethodPut {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Read the request body
		var configData map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&configData); err != nil {
			log.Printf("Error decoding request body: %v", err)
			http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
			return
		}

		// Convert to YAML
		yamlData, err := yaml.Marshal(configData)
		if err != nil {
			log.Printf("Error marshaling config to YAML: %v", err)
			http.Error(w, fmt.Sprintf("Failed to convert config to YAML: %v", err), http.StatusInternalServerError)
			return
		}

		// Write to file
		if err := os.WriteFile(configPath, yamlData, 0644); err != nil {
			log.Printf("Error writing config file: %v", err)
			http.Error(w, fmt.Sprintf("Failed to write config file: %v", err), http.StatusInternalServerError)
			return
		}

		log.Printf("Configuration updated successfully")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "success", "message": "Configuration updated successfully"})
	}
}

// toolsDBHandler reads and serves the tools_db.json file
func toolsDBHandler(configDir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Only allow GET requests
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Construct the tools_db.json path
		toolsDBPath := filepath.Join(configDir, "tools_db.json")

		// Read the tools database file
		data, err := os.ReadFile(toolsDBPath)
		if err != nil {
			log.Printf("Error reading tools_db.json: %v", err)
			http.Error(w, fmt.Sprintf("Failed to read tools database: %v", err), http.StatusInternalServerError)
			return
		}

		// Parse JSON to validate it
		var tools interface{}
		if err := json.Unmarshal(data, &tools); err != nil {
			log.Printf("Error parsing tools_db.json: %v", err)
			http.Error(w, fmt.Sprintf("Failed to parse tools database: %v", err), http.StatusInternalServerError)
			return
		}

		// Send response
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(tools); err != nil {
			log.Printf("Error encoding tools to JSON: %v", err)
			http.Error(w, fmt.Sprintf("Failed to encode tools: %v", err), http.StatusInternalServerError)
			return
		}
	}
}

// newReverseProxy creates a reverse proxy to targetBase and strips the given prefix from the incoming path
func newReverseProxy(targetBase, stripPrefix string, forwardAuth bool) (*httputil.ReverseProxy, error) {
	targetURL, err := url.Parse(targetBase)
	if err != nil {
		return nil, fmt.Errorf("invalid target URL %q: %w", targetBase, err)
	}

	proxy := httputil.NewSingleHostReverseProxy(targetURL)

	// Customize the director to rewrite the request
	origDirector := proxy.Director
	proxy.Director = func(r *http.Request) {
		origDirector(r)
		// Preserve original path then strip prefix
		p := r.URL.Path
		if strings.HasPrefix(p, stripPrefix) {
			p = strings.TrimPrefix(p, stripPrefix)
		}
		// Ensure leading slash
		if !strings.HasPrefix(p, "/") {
			p = "/" + p
		}
		r.URL.Path = p
		r.Host = targetURL.Host

		// Set Origin header to match the target URL for Grafana embedding
		// This is required for Grafana to accept the iframe embedding
		r.Header.Set("Origin", targetURL.Scheme+"://"+targetURL.Host)

		// Optionally forward Authorization header
		if !forwardAuth {
			r.Header.Del("Authorization")
		}

		// Log the proxied request for debugging
		log.Printf("Proxying: %s %s -> %s://%s%s", r.Method, stripPrefix, targetURL.Scheme, targetURL.Host, p)
	}

	// Add error handler for proxy failures
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		log.Printf("Proxy error for %s: %v", r.URL.Path, err)
		http.Error(w, fmt.Sprintf("Bad Gateway: %v", err), http.StatusBadGateway)
	}

	// Sanitize response headers for iframe embedding
	proxy.ModifyResponse = func(resp *http.Response) error {
		// Remove frame-busting headers
		resp.Header.Del("X-Frame-Options")
		// Allow iframe from self (dashboard origin)
		// If CSP exists, adjust frame-ancestors; otherwise set a permissive one for self
		csp := resp.Header.Get("Content-Security-Policy")
		if csp == "" {
			resp.Header.Set("Content-Security-Policy", "frame-ancestors 'self'")
		} else {
			// Naive replacement of frame-ancestors directive
			// If frame-ancestors exists, replace its value with 'self'
			// Otherwise append directive
			lower := strings.ToLower(csp)
			if strings.Contains(lower, "frame-ancestors") {
				// Split directives by ';'
				parts := strings.Split(csp, ";")
				for i, d := range parts {
					if strings.Contains(strings.ToLower(d), "frame-ancestors") {
						parts[i] = "frame-ancestors 'self'"
					}
				}
				resp.Header.Set("Content-Security-Policy", strings.Join(parts, ";"))
			} else {
				resp.Header.Set("Content-Security-Policy", csp+"; frame-ancestors 'self'")
			}
		}
		return nil
	}

	return proxy, nil
}

func staticFileServer(staticDir string) http.Handler {
	fs := http.FileServer(http.Dir(staticDir))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Never serve index.html for API or embedded proxy routes
		// These should be handled by their respective handlers
		p := r.URL.Path
		if strings.HasPrefix(p, "/api/") || strings.HasPrefix(p, "/embedded/") ||
			strings.HasPrefix(p, "/metrics/") || strings.HasPrefix(p, "/public/") ||
			strings.HasPrefix(p, "/avatar/") {
			// These paths should have been handled by other handlers
			// If we reach here, it means the proxy failed or route not found
			http.Error(w, "Service not available", http.StatusBadGateway)
			return
		}

		full := path.Join(staticDir, path.Clean(p))

		// Check if file exists
		info, err := os.Stat(full)
		if err == nil {
			// File exists
			if !info.IsDir() {
				// It's a file, serve it
				fs.ServeHTTP(w, r)
				return
			}
			// It's a directory, try index.html
			indexPath := path.Join(full, "index.html")
			if _, err := os.Stat(indexPath); err == nil {
				http.ServeFile(w, r, indexPath)
				return
			}
		}

		// File doesn't exist or is directory without index.html
		// For SPA routing: serve index.html for routes without file extension
		if !strings.Contains(path.Base(p), ".") {
			http.ServeFile(w, r, path.Join(staticDir, "index.html"))
			return
		}

		// Otherwise let the file server handle it (will return 404)
		fs.ServeHTTP(w, r)
	})
}

func main() {
	// Flags/env for configuration
	port := flag.String("port", env("DASHBOARD_PORT", "8700"), "dashboard port")
	staticDir := flag.String("static", env("DASHBOARD_STATIC_DIR", "../frontend"), "static assets directory")
	configFile := flag.String("config", env("ROUTER_CONFIG_PATH", "../../config/config.yaml"), "path to config.yaml")

	// Upstream targets
	grafanaURL := flag.String("grafana", env("TARGET_GRAFANA_URL", ""), "Grafana base URL")
	promURL := flag.String("prometheus", env("TARGET_PROMETHEUS_URL", ""), "Prometheus base URL")
	routerAPI := flag.String("router_api", env("TARGET_ROUTER_API_URL", "http://localhost:8080"), "Router API base URL")
	routerMetrics := flag.String("router_metrics", env("TARGET_ROUTER_METRICS_URL", "http://localhost:9190/metrics"), "Router metrics URL")
	openwebuiURL := flag.String("openwebui", env("TARGET_OPENWEBUI_URL", ""), "Open WebUI base URL")

	flag.Parse()

	// Resolve config file path to absolute path
	absConfigPath, err := filepath.Abs(*configFile)
	if err != nil {
		log.Fatalf("Failed to resolve config path: %v", err)
	}
	log.Printf("Config file path: %s", absConfigPath)

	mux := http.NewServeMux()

	// Health check endpoint
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"healthy","service":"semantic-router-dashboard"}`))
	})

	// Config endpoint - serve the config.yaml as JSON
	mux.HandleFunc("/api/router/config/all", configHandler(absConfigPath))
	log.Printf("Config API endpoint registered: /api/router/config/all")

	// Config update endpoint - update the config.yaml file
	mux.HandleFunc("/api/router/config/update", updateConfigHandler(absConfigPath))
	log.Printf("Config update API endpoint registered: /api/router/config/update")

	// Tools DB endpoint - serve the tools_db.json
	configDir := filepath.Dir(absConfigPath)
	mux.HandleFunc("/api/tools-db", toolsDBHandler(configDir))
	log.Printf("Tools DB API endpoint registered: /api/tools-db")

	// Router API proxy (forward Authorization) - MUST be registered before Grafana
	var routerAPIProxy *httputil.ReverseProxy
	if *routerAPI != "" {
		rp, err := newReverseProxy(*routerAPI, "/api/router", true)
		if err != nil {
			log.Fatalf("router API proxy error: %v", err)
		}
		routerAPIProxy = rp
		mux.Handle("/api/router/", rp)
		log.Printf("Router API proxy configured: %s", *routerAPI)
	}

	// Grafana proxy and static assets
	var grafanaStaticProxy *httputil.ReverseProxy
	if *grafanaURL != "" {
		gp, err := newReverseProxy(*grafanaURL, "/embedded/grafana", false)
		if err != nil {
			log.Fatalf("grafana proxy error: %v", err)
		}
		mux.Handle("/embedded/grafana/", gp)

		// Proxy for Grafana static assets (no prefix stripping)
		grafanaStaticProxy, _ = newReverseProxy(*grafanaURL, "", false)
		mux.Handle("/public/", grafanaStaticProxy)
		mux.Handle("/avatar/", grafanaStaticProxy)

		log.Printf("Grafana proxy configured: %s", *grafanaURL)
		log.Printf("Grafana static assets proxied: /public/, /avatar/")
	} else {
		mux.HandleFunc("/embedded/grafana/", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			w.Write([]byte(`{"error":"Grafana not configured","message":"TARGET_GRAFANA_URL environment variable is not set"}`))
		})
		log.Printf("Warning: Grafana URL not configured")
	}

	// Smart /api/ router: route to Router API or Grafana API based on path
	mux.HandleFunc("/api/", func(w http.ResponseWriter, r *http.Request) {
		// If path starts with /api/router/, use Router API proxy
		if strings.HasPrefix(r.URL.Path, "/api/router/") && routerAPIProxy != nil {
			routerAPIProxy.ServeHTTP(w, r)
			return
		}
		// Otherwise, if Grafana is configured, proxy to Grafana API
		if grafanaStaticProxy != nil {
			grafanaStaticProxy.ServeHTTP(w, r)
			return
		}
		// No handler available
		http.Error(w, "Service not available", http.StatusBadGateway)
	})

	// Router metrics passthrough
	mux.HandleFunc("/metrics/router", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, *routerMetrics, http.StatusTemporaryRedirect)
	})

	// Static frontend - MUST be registered last
	mux.Handle("/", staticFileServer(*staticDir))

	// Prometheus proxy (optional)
	if *promURL != "" {
		pp, err := newReverseProxy(*promURL, "/embedded/prometheus", false)
		if err != nil {
			log.Fatalf("prometheus proxy error: %v", err)
		}
		mux.Handle("/embedded/prometheus", pp)
		mux.Handle("/embedded/prometheus/", pp)
		log.Printf("Prometheus proxy configured: %s", *promURL)
	} else {
		mux.HandleFunc("/embedded/prometheus/", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			w.Write([]byte(`{"error":"Prometheus not configured","message":"TARGET_PROMETHEUS_URL environment variable is not set"}`))
		})
		log.Printf("Warning: Prometheus URL not configured")
	}

	// Open WebUI proxy (optional)
	if *openwebuiURL != "" {
		op, err := newReverseProxy(*openwebuiURL, "/embedded/openwebui", true)
		if err != nil {
			log.Fatalf("openwebui proxy error: %v", err)
		}
		mux.Handle("/embedded/openwebui", op)
		mux.Handle("/embedded/openwebui/", op)
		log.Printf("Open WebUI proxy configured: %s", *openwebuiURL)
	} else {
		mux.HandleFunc("/embedded/openwebui/", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			w.Write([]byte(`{"error":"Open WebUI not configured","message":"TARGET_OPENWEBUI_URL environment variable is not set or empty"}`))
		})
		log.Printf("Info: Open WebUI not configured (optional)")
	}

	addr := ":" + *port
	log.Printf("Semantic Router Dashboard listening on %s", addr)
	log.Printf("Static dir: %s", *staticDir)
	if *grafanaURL != "" {
		log.Printf("Grafana: %s → /embedded/grafana/", *grafanaURL)
	}
	if *promURL != "" {
		log.Printf("Prometheus: %s → /embedded/prometheus/", *promURL)
	}
	if *openwebuiURL != "" {
		log.Printf("OpenWebUI: %s → /embedded/openwebui/", *openwebuiURL)
	}
	log.Printf("Router API: %s → /api/router/*", *routerAPI)
	log.Printf("Router Metrics: %s → /metrics/router", *routerMetrics)

	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
