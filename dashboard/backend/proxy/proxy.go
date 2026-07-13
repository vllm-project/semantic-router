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

	"github.com/vllm-project/semantic-router/dashboard/backend/browsersecurity"
)

// NewReverseProxy creates a reverse proxy to targetBase and strips the given prefix from the incoming path
// It also handles CORS, iframe embedding, and other security headers
func NewReverseProxy(targetBase, stripPrefix string, forwardAuthorization bool) (*httputil.ReverseProxy, error) {
	targetURL, err := url.Parse(targetBase)
	if err != nil {
		return nil, fmt.Errorf("invalid target URL %q: %w", targetBase, err)
	}

	proxy := httputil.NewSingleHostReverseProxy(targetURL)
	proxy.FlushInterval = -1 // -1 means flush immediately after each write
	director := reverseProxyDirector{
		original:             proxy.Director,
		target:               targetURL,
		stripPrefix:          stripPrefix,
		forwardAuthorization: forwardAuthorization,
		overrideOrigin:       strings.EqualFold(os.Getenv("PROXY_OVERRIDE_ORIGIN"), "true"),
	}
	proxy.Director = director.direct
	proxy.ErrorHandler = handleReverseProxyError
	proxy.ModifyResponse = reverseProxyResponseModifier{
		embeddedContent: isEmbeddedProxyPrefix(stripPrefix),
	}.modify

	return proxy, nil
}

type reverseProxyDirector struct {
	original             func(*http.Request)
	target               *url.URL
	stripPrefix          string
	forwardAuthorization bool
	overrideOrigin       bool
}

func (d reverseProxyDirector) direct(r *http.Request) {
	incomingOrigin := r.Header.Get("Origin")
	originAllowed := incomingOrigin != "" && browsersecurity.ValidOrigin(r)
	d.original(r)
	stripDashboardCredentials(r, d.forwardAuthorization)

	path := stripProxyPath(r.URL.Path, d.stripPrefix)
	r.URL.Path = path
	setForwardedOrigin(r.Header, incomingOrigin, originAllowed)
	setTargetOrigin(r, d.target, d.overrideOrigin)
	if pnaHeader := r.Header.Get("Access-Control-Request-Private-Network"); pnaHeader != "" {
		log.Printf("PNA preflight request detected: %s", pnaHeader)
	}
	setProxyForwardingHeaders(r)
	r.Host = d.target.Host
	log.Printf("Proxying: %s %s -> %s://%s%s", r.Method, d.stripPrefix, d.target.Scheme, d.target.Host, path)
}

func stripProxyPath(path, stripPrefix string) string {
	path = strings.TrimPrefix(path, stripPrefix)
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	return path
}

func setForwardedOrigin(header http.Header, origin string, allowed bool) {
	if allowed {
		header.Set("X-Forwarded-Origin", origin)
		return
	}
	header.Del("X-Forwarded-Origin")
}

func setTargetOrigin(r *http.Request, target *url.URL, override bool) {
	if !override && r.Header.Get("Origin") != "" {
		return
	}
	if override && !isProxyWriteMethod(r.Method) && r.Header.Get("Origin") != "" {
		return
	}
	r.Header.Set("Origin", target.Scheme+"://"+target.Host)
}

func isProxyWriteMethod(method string) bool {
	switch method {
	case http.MethodPost, http.MethodPut, http.MethodPatch, http.MethodDelete:
		return true
	default:
		return false
	}
}

func setProxyForwardingHeaders(r *http.Request) {
	r.Header.Set("X-Forwarded-Host", r.Host)
	proto := "http"
	if r.TLS != nil {
		proto = "https"
	}
	if forwardedProto := r.Header.Get("X-Forwarded-Proto"); forwardedProto != "" {
		proto = forwardedProto
	}
	r.Header.Set("X-Forwarded-Proto", proto)

	clientIP := proxyClientIP(r.RemoteAddr)
	if clientIP == "" {
		return
	}
	if existing := r.Header.Get("X-Forwarded-For"); existing != "" {
		clientIP = existing + ", " + clientIP
	}
	r.Header.Set("X-Forwarded-For", clientIP)
}

func proxyClientIP(remoteAddr string) string {
	if remoteAddr == "" {
		return ""
	}
	ip, _, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		return remoteAddr
	}
	return ip
}

func handleReverseProxyError(w http.ResponseWriter, r *http.Request, err error) {
	log.Printf("Proxy error for %s: %v", r.URL.Path, err)
	http.Error(w, "Bad Gateway", http.StatusBadGateway)
}

type reverseProxyResponseModifier struct {
	embeddedContent bool
}

func (m reverseProxyResponseModifier) modify(resp *http.Response) error {
	stripDashboardSessionSetCookies(resp.Header)
	resp.Header.Del("Service-Worker-Allowed")
	resp.Header.Del("Access-Control-Allow-Origin")
	resp.Header.Del("Access-Control-Allow-Credentials")
	resp.Header.Del("Access-Control-Allow-Private-Network")
	resp.Header.Del("X-Frame-Options")
	transformCSPHeaders(resp.Header, m.embeddedContent)
	setProxyCORSHeaders(resp)
	return nil
}

func setProxyCORSHeaders(resp *http.Response) {
	origin := resp.Request.Header.Get("X-Forwarded-Origin")
	if origin != "" {
		resp.Header.Set("Access-Control-Allow-Origin", origin)
		resp.Header.Set("Access-Control-Allow-Credentials", "true")
		resp.Header.Set("Vary", "Origin")
	}
	resp.Header.Set("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
	resp.Header.Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With, Accept, Origin")
	resp.Header.Set("Access-Control-Expose-Headers", "Content-Length, Content-Range")
	if origin != "" && resp.Request.Header.Get("Access-Control-Request-Private-Network") == "true" {
		resp.Header.Set("Access-Control-Allow-Private-Network", "true")
	}
}

func isEmbeddedProxyPrefix(stripPrefix string) bool {
	prefix := strings.TrimRight(strings.TrimSpace(stripPrefix), "/")
	return prefix == "/embedded" || strings.HasPrefix(prefix, "/embedded/")
}

func setCSPDirective(policy, name, value string) string {
	directive := name + " " + value
	parts := strings.Split(policy, ";")
	found := false
	for i, part := range parts {
		fields := strings.Fields(part)
		if len(fields) > 0 && strings.EqualFold(fields[0], name) {
			parts[i] = directive
			found = true
		}
	}
	if found {
		return strings.Join(parts, ";")
	}
	policy = strings.TrimSpace(policy)
	if policy == "" {
		return directive
	}
	return strings.TrimRight(policy, "; ") + "; " + directive
}

func transformCSPHeaders(header http.Header, embeddedContent bool) {
	const headerName = "Content-Security-Policy"
	policies := header.Values(headerName)
	if len(policies) == 0 {
		policies = []string{""}
	}

	transformed := make([]string, len(policies))
	for i, policyList := range policies {
		// A field may contain a comma-delimited policy list after an
		// intermediary combines repeated CSP fields. Every policy remains an
		// independent enforcement layer and must be transformed independently.
		policyParts := strings.Split(policyList, ",")
		for j, policy := range policyParts {
			policy = setCSPDirective(policy, "frame-ancestors", "'self'")
			if embeddedContent {
				policy = setCSPDirective(policy, "worker-src", "'none'")
			}
			policyParts[j] = policy
		}
		transformed[i] = strings.Join(policyParts, ", ")
	}
	header[http.CanonicalHeaderKey(headerName)] = transformed
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
