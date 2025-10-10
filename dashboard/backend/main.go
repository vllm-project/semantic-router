package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path"
	"strings"
)

// env returns the env var or default
func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
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

		// Optionally forward Authorization header
		if !forwardAuth {
			r.Header.Del("Authorization")
		}
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
		// Serve index.html for root and for unknown routes (SPA)
		p := r.URL.Path
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

	// Upstream targets
	grafanaURL := flag.String("grafana", env("TARGET_GRAFANA_URL", ""), "Grafana base URL")
	promURL := flag.String("prometheus", env("TARGET_PROMETHEUS_URL", ""), "Prometheus base URL")
	routerAPI := flag.String("router_api", env("TARGET_ROUTER_API_URL", "http://localhost:8080"), "Router API base URL")
	routerMetrics := flag.String("router_metrics", env("TARGET_ROUTER_METRICS_URL", "http://localhost:9190/metrics"), "Router metrics URL")
	openwebuiURL := flag.String("openwebui", env("TARGET_OPENWEBUI_URL", ""), "Open WebUI base URL")

	flag.Parse()

	mux := http.NewServeMux()

	// Health check endpoint
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"healthy","service":"semantic-router-dashboard"}`))
	})

	// Static frontend
	mux.Handle("/", staticFileServer(*staticDir)) // Router API proxy (forward Authorization)
	if *routerAPI != "" {
		rp, err := newReverseProxy(*routerAPI, "/api/router", true)
		if err != nil {
			log.Fatalf("router API proxy error: %v", err)
		}
		mux.Handle("/api/router", rp)
		mux.Handle("/api/router/", rp)
	}

	// Router metrics passthrough (no rewrite, simple redirect/proxy)
	mux.HandleFunc("/metrics/router", func(w http.ResponseWriter, r *http.Request) {
		// Simple 302 redirect for now to let Prometheus UI open directly
		http.Redirect(w, r, *routerMetrics, http.StatusTemporaryRedirect)
	})

	// Grafana proxy
	if *grafanaURL != "" {
		gp, err := newReverseProxy(*grafanaURL, "/embedded/grafana", false)
		if err != nil {
			log.Fatalf("grafana proxy error: %v", err)
		}
		mux.Handle("/embedded/grafana", gp)
		mux.Handle("/embedded/grafana/", gp)
	}

	// Prometheus proxy (optional)
	if *promURL != "" {
		pp, err := newReverseProxy(*promURL, "/embedded/prometheus", false)
		if err != nil {
			log.Fatalf("prometheus proxy error: %v", err)
		}
		mux.Handle("/embedded/prometheus", pp)
		mux.Handle("/embedded/prometheus/", pp)
	}

	// Open WebUI proxy (optional)
	if *openwebuiURL != "" {
		op, err := newReverseProxy(*openwebuiURL, "/embedded/openwebui", true)
		if err != nil {
			log.Fatalf("openwebui proxy error: %v", err)
		}
		mux.Handle("/embedded/openwebui", op)
		mux.Handle("/embedded/openwebui/", op)
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
