package proxy

import (
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
)

// NewReverseProxy creates a reverse proxy to targetBase and strips the given prefix from the incoming path
// It also handles CORS, iframe embedding, and other security headers
func NewReverseProxy(targetBase, stripPrefix string, forwardAuth bool) (*httputil.ReverseProxy, error) {
	targetURL, err := url.Parse(targetBase)
	if err != nil {
		return nil, fmt.Errorf("invalid target URL %q: %w", targetBase, err)
	}

	proxy := httputil.NewSingleHostReverseProxy(targetURL)

	// Enable streaming responses (critical for SSE/ChatUI)
	// FlushInterval = 0 means flush immediately, supporting real-time streaming
	proxy.FlushInterval = -1 // -1 means flush immediately after each write

	// Optional behavior: override Origin header to target for non-idempotent requests
	// This helps when the upstream enforces strict Origin checking (e.g., CSRF protections)
	overrideOrigin := strings.EqualFold(os.Getenv("PROXY_OVERRIDE_ORIGIN"), "true")

	// Customize the director to rewrite the request
	origDirector := proxy.Director
	proxy.Director = func(r *http.Request) {
		origDirector(r)
		// Preserve original path then strip prefix
		p := r.URL.Path
		p = strings.TrimPrefix(p, stripPrefix)
		// Ensure leading slash
		if !strings.HasPrefix(p, "/") {
			p = "/" + p
		}
		r.URL.Path = p
		r.Host = targetURL.Host

		// Capture incoming Origin for downstream CORS decisions
		incomingOrigin := r.Header.Get("Origin")
		if overrideOrigin && (r.Method == http.MethodPost || r.Method == http.MethodPut || r.Method == http.MethodPatch || r.Method == http.MethodDelete) {
			// Force Origin to target to satisfy upstream Origin/CSRF checks for write requests
			r.Header.Set("Origin", targetURL.Scheme+"://"+targetURL.Host)
		} else if incomingOrigin == "" {
			// If no Origin present, set to target origin to avoid empty Origin edge cases
			r.Header.Set("Origin", targetURL.Scheme+"://"+targetURL.Host)
		}

		// Forward the original Origin (prior to any override) for response CORS handling
		if incomingOrigin != "" {
			r.Header.Set("X-Forwarded-Origin", incomingOrigin)
		} else {
			r.Header.Set("X-Forwarded-Origin", targetURL.Scheme+"://"+targetURL.Host)
		}

		// Set Origin header to match the target URL for iframe embedding
		// This is required for services like Grafana and Chat UI to accept the iframe embedding
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

	// Sanitize response headers for iframe embedding and enable CORS
	// This approach is based on the Grafana proxy implementation
	proxy.ModifyResponse = func(resp *http.Response) error {
		// Remove frame-busting headers that prevent iframe embedding
		resp.Header.Del("X-Frame-Options")

		// Handle Content-Security-Policy for iframe embedding
		// Allow iframe from self (dashboard origin)
		csp := resp.Header.Get("Content-Security-Policy")
		if csp == "" {
			// If no CSP exists, set a permissive one for self
			resp.Header.Set("Content-Security-Policy", "frame-ancestors 'self'")
		} else {
			// If CSP exists, modify frame-ancestors directive
			// This ensures the embedded service (like Chat UI) can be displayed in an iframe
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
				// Append frame-ancestors directive
				resp.Header.Set("Content-Security-Policy", csp+"; frame-ancestors 'self'")
			}
		}

		// Add permissive CORS headers for proxied responses
		// This allows the frontend to make API calls through the proxy
		if resp.Header.Get("Access-Control-Allow-Origin") == "" {
			resp.Header.Set("Access-Control-Allow-Origin", "*")
		}
		if resp.Header.Get("Access-Control-Allow-Methods") == "" {
			resp.Header.Set("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
		}
		if resp.Header.Get("Access-Control-Allow-Headers") == "" {
			resp.Header.Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
		}

		return nil
	}

	return proxy, nil
}
