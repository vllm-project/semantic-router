package proxy

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net"
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
		// This is required for services like Grafana, Chat UI, and OpenWebUI to accept the iframe embedding
		// and pass CSRF/Origin validation checks. The original Origin is preserved in X-Forwarded-Origin
		// for CORS response handling. This override is intentional and necessary for iframe embedding to work.
		r.Header.Set("Origin", targetURL.Scheme+"://"+targetURL.Host)

		// Set X-Forwarded-* headers to preserve client information
		// These headers should reflect the original client request, not the target service
		r.Header.Set("X-Forwarded-Host", r.Host)

		// Determine the original protocol (http or https)
		proto := "http"
		if r.TLS != nil {
			proto = "https"
		}
		// Also check X-Forwarded-Proto from upstream (if we're behind another proxy)
		if forwardedProto := r.Header.Get("X-Forwarded-Proto"); forwardedProto != "" {
			proto = forwardedProto
		}
		r.Header.Set("X-Forwarded-Proto", proto)

		// Extract client IP from RemoteAddr (strip port if present)
		var clientIP string
		if r.RemoteAddr != "" {
			ip, _, err := net.SplitHostPort(r.RemoteAddr)
			if err != nil {
				// If SplitHostPort fails, RemoteAddr might not have a port
				clientIP = r.RemoteAddr
			} else {
				clientIP = ip
			}
		}

		// Append to existing X-Forwarded-For if present (we might be behind another proxy)
		if clientIP != "" {
			if existing := r.Header.Get("X-Forwarded-For"); existing != "" {
				r.Header.Set("X-Forwarded-For", existing+", "+clientIP)
			} else {
				r.Header.Set("X-Forwarded-For", clientIP)
			}
		}

		// Set Host header to match target (some services check this)
		r.Host = targetURL.Host

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
		// Always override CORS headers to ensure iframe embedding works correctly

		// Get the original request's Origin header from X-Forwarded-Origin
		origin := resp.Request.Header.Get("X-Forwarded-Origin")
		if origin != "" {
			// If we have an origin, echo it back and allow credentials
			resp.Header.Set("Access-Control-Allow-Origin", origin)
			resp.Header.Set("Access-Control-Allow-Credentials", "true")
			resp.Header.Set("Vary", "Origin")
		} else {
			// If no origin, use wildcard (but can't use credentials with wildcard)
			resp.Header.Set("Access-Control-Allow-Origin", "*")
		}

		resp.Header.Set("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
		resp.Header.Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With, Accept, Origin")
		resp.Header.Set("Access-Control-Expose-Headers", "Content-Length, Content-Range")

		// Rewrite URLs in HTML/CSS/JS responses to fix hardcoded internal URLs
		// This is necessary for Chat UI which may have hardcoded https://chat-ui:3000 URLs
		contentType := resp.Header.Get("Content-Type")
		if strings.Contains(contentType, "text/html") ||
			strings.Contains(contentType, "text/css") ||
			strings.Contains(contentType, "application/javascript") ||
			strings.Contains(contentType, "text/javascript") {

			// Read the response body
			bodyBytes, err := io.ReadAll(resp.Body)
			if err != nil {
				log.Printf("Error reading response body: %v", err)
				return err
			}
			resp.Body.Close()

			// Replace hardcoded internal URLs with proxy paths
			bodyStr := string(bodyBytes)

			// Replace various forms of the internal URL
			// Note: /chatui/ assets should stay as /chatui/ (not /embedded/chatui/)
			// because the backend has a separate route for /chatui/ assets
			replacements := map[string]string{
				"https://chat-ui:3000/chatui/": "/chatui/",
				"https://chat-ui:3000/chatui":  "/chatui",
				"http://chat-ui:3000/chatui/":  "/chatui/",
				"http://chat-ui:3000/chatui":   "/chatui",
				"https://chat-ui:3000/":        "/embedded/chatui/",
				"https://chat-ui:3000":         "/embedded/chatui",
				"http://chat-ui:3000/":         "/embedded/chatui/",
				"http://chat-ui:3000":          "/embedded/chatui",
			}

			for old, new := range replacements {
				bodyStr = strings.ReplaceAll(bodyStr, old, new)
			}

			// Create new response body
			newBody := []byte(bodyStr)
			resp.Body = io.NopCloser(bytes.NewReader(newBody))
			resp.ContentLength = int64(len(newBody))
			resp.Header.Set("Content-Length", fmt.Sprintf("%d", len(newBody)))
		}

		return nil
	}

	return proxy, nil
}
