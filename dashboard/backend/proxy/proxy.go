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

	// Enable streaming responses (critical for SSE)
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

		// If no Origin header, try to extract from Referer (for iframe requests)
		if incomingOrigin == "" {
			if referer := r.Header.Get("Referer"); referer != "" {
				if refererURL, err := url.Parse(referer); err == nil {
					incomingOrigin = refererURL.Scheme + "://" + refererURL.Host
				}
			}
		}

		// Forward the original Origin for response CORS handling
		// This is critical: we must preserve the actual client origin, not the target URL
		if incomingOrigin != "" {
			r.Header.Set("X-Forwarded-Origin", incomingOrigin)
		}
		// If still no origin, don't set X-Forwarded-Origin (will use wildcard in response)

		// Set Origin header to match the target URL for iframe embedding
		// This is required for embedded services to accept iframe embedding
		// and pass CSRF/Origin validation checks.
		if overrideOrigin && (r.Method == http.MethodPost || r.Method == http.MethodPut || r.Method == http.MethodPatch || r.Method == http.MethodDelete) {
			// Force Origin to target to satisfy upstream Origin/CSRF checks for write requests
			r.Header.Set("Origin", targetURL.Scheme+"://"+targetURL.Host)
		} else if r.Header.Get("Origin") == "" {
			// If no Origin present, set to target origin to avoid empty Origin edge cases
			r.Header.Set("Origin", targetURL.Scheme+"://"+targetURL.Host)
		}

		// Preserve Access-Control-Request-Private-Network header for PNA preflight requests
		// This header is sent by Chrome when making preflight requests to private network resources
		// from public pages. We need to forward it to the upstream service.
		if pnaHeader := r.Header.Get("Access-Control-Request-Private-Network"); pnaHeader != "" {
			log.Printf("PNA preflight request detected: %s", pnaHeader)
		}

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
			// This ensures the embedded service can be displayed in an iframe
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

		// Add Private Network Access (PNA) headers to allow public pages to access private network resources
		// This is required when accessing the dashboard via a public domain that proxies to local services
		// Chrome's Private Network Access policy requires these headers for cross-origin requests to private networks
		// See: https://developer.chrome.com/blog/private-network-access-preflight/
		resp.Header.Set("Access-Control-Allow-Private-Network", "true")

		return nil
	}

	return proxy, nil
}

// NewJaegerProxy creates a reverse proxy specifically for Jaeger UI with dark theme injection
func NewJaegerProxy(targetBase, stripPrefix string) (*httputil.ReverseProxy, error) {
	proxy, err := NewReverseProxy(targetBase, stripPrefix, false)
	if err != nil {
		return nil, err
	}

	// Override ModifyResponse to inject dark theme script into HTML responses
	originalModifyResponse := proxy.ModifyResponse
	proxy.ModifyResponse = func(resp *http.Response) error {
		// First apply the original response modifications (CORS, CSP, etc.)
		if originalModifyResponse != nil {
			if err := originalModifyResponse(resp); err != nil {
				return err
			}
		}

		// Only inject script into HTML responses
		contentType := resp.Header.Get("Content-Type")
		if !strings.Contains(contentType, "text/html") {
			return nil
		}

		// Read the response body
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return err
		}
		resp.Body.Close()

		// Inject light theme script to ensure Jaeger displays consistently in light mode
		// This avoids theme conflicts with the dashboard
		themeScript := `<script>
(function() {
  try {
    // Force Jaeger UI to use light theme for consistent appearance
    localStorage.setItem('jaeger-ui-theme', 'light');
    localStorage.setItem('theme', 'light');

    // Set data-theme attribute on document element
    if (document.documentElement) {
      document.documentElement.setAttribute('data-theme', 'light');
      document.documentElement.setAttribute('data-bs-theme', 'light');
      document.documentElement.style.colorScheme = 'light';
    }

    // Also set it after DOM is ready
    document.addEventListener('DOMContentLoaded', function() {
      if (document.documentElement) {
        document.documentElement.setAttribute('data-theme', 'light');
        document.documentElement.setAttribute('data-bs-theme', 'light');
        document.documentElement.style.colorScheme = 'light';
      }
    });
  } catch (e) {
    console.error('Failed to set Jaeger theme:', e);
  }
})();
</script>`

		// Try to inject before </head>, otherwise before </body>
		modifiedBody := string(body)
		if strings.Contains(modifiedBody, "</head>") {
			modifiedBody = strings.Replace(modifiedBody, "</head>", themeScript+"</head>", 1)
		} else if strings.Contains(modifiedBody, "<body") {
			// Find the end of the <body> tag and inject after it
			bodyTagEnd := strings.Index(modifiedBody, ">")
			if bodyTagEnd != -1 {
				modifiedBody = modifiedBody[:bodyTagEnd+1] + themeScript + modifiedBody[bodyTagEnd+1:]
			}
		}

		// Create new response body
		newBody := []byte(modifiedBody)
		resp.Body = io.NopCloser(bytes.NewReader(newBody))
		resp.ContentLength = int64(len(newBody))
		resp.Header.Set("Content-Length", fmt.Sprintf("%d", len(newBody)))

		return nil
	}

	return proxy, nil
}
