package handlers

import (
	"context"
	"crypto/tls"
	"errors"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"path"
	"strings"
	"time"
)

const (
	routerClassifierProxyMaxRequestBodyBytes = 52 << 20
	routerClassifierProxyTimeout             = 2 * time.Minute
)

var routerClassifierProxyHopHeaders = [...]string{
	"Connection",
	"Keep-Alive",
	"Proxy-Authenticate",
	"Proxy-Authorization",
	"Proxy-Connection",
	"Te",
	"Trailer",
	"Transfer-Encoding",
	"Upgrade",
}

// RouterClassifierProxyHandler forwards knowledge-base management traffic to
// the fixed internal router API. Dashboard credentials and hop-by-hop headers
// never cross this trust boundary.
func RouterClassifierProxyHandler(routerAPIURL string, readonlyMode bool) http.HandlerFunc {
	baseURL, parseErr := parseRouterClassifierProxyBaseURL(routerAPIURL)
	return routerClassifierProxyHandler(baseURL, parseErr, readonlyMode, newRouterClassifierProxyClient())
}

func routerClassifierProxyHandler(
	baseURL *url.URL,
	configurationErr error,
	readonlyMode bool,
	client *http.Client,
) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		if configurationErr != nil || baseURL == nil || client == nil {
			http.Error(w, "Router API URL is not configured", http.StatusBadGateway)
			return
		}
		if readonlyMode && r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.Error(w, "Dashboard is in read-only mode. Configuration editing is disabled.", http.StatusForbidden)
			return
		}

		targetURL, err := routerClassifierProxyTarget(baseURL, r.URL)
		if err != nil {
			http.Error(w, "Invalid router API path", http.StatusBadRequest)
			return
		}
		if r.ContentLength > routerClassifierProxyMaxRequestBodyBytes {
			http.Error(w, "Request body too large", http.StatusRequestEntityTooLarge)
			return
		}

		var body io.ReadCloser = http.NoBody
		if r.Body != nil {
			r.Body = http.MaxBytesReader(w, r.Body, routerClassifierProxyMaxRequestBodyBytes)
			body = r.Body
		}
		proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL.String(), body)
		if err != nil {
			http.Error(w, "Failed to build router API request", http.StatusInternalServerError)
			return
		}
		proxyReq.ContentLength = r.ContentLength
		copyRouterClassifierProxyHeaders(proxyReq.Header, r.Header)
		stripRouterClassifierProxyCredentials(proxyReq.Header)
		proxyReq.Host = baseURL.Host

		resp, err := client.Do(proxyReq)
		if err != nil {
			var maxBytesErr *http.MaxBytesError
			if errors.As(err, &maxBytesErr) {
				http.Error(w, "Request body too large", http.StatusRequestEntityTooLarge)
				return
			}
			http.Error(w, "Router API request failed", http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode >= http.StatusMultipleChoices && resp.StatusCode < http.StatusBadRequest {
			http.Error(w, "Router API redirect rejected", http.StatusBadGateway)
			return
		}
		copyRouterClassifierProxyHeaders(w.Header(), resp.Header)
		w.Header().Del("Set-Cookie")
		w.Header().Del("Clear-Site-Data")
		w.WriteHeader(resp.StatusCode)
		if _, err := io.Copy(w, resp.Body); err != nil && !errors.Is(err, context.Canceled) {
			log.Printf("router classifier proxy: response copy failed")
		}
	}
}

func parseRouterClassifierProxyBaseURL(raw string) (*url.URL, error) {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed.Opaque != "" || parsed.Host == "" || parsed.Hostname() == "" || parsed.User != nil || parsed.Fragment != "" {
		return nil, errors.New("invalid router API URL")
	}
	parsed.Scheme = strings.ToLower(parsed.Scheme)
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, errors.New("invalid router API URL scheme")
	}
	parsed.RawQuery = ""
	return parsed, nil
}

func routerClassifierProxyTarget(baseURL, requestURL *url.URL) (*url.URL, error) {
	if baseURL == nil || requestURL == nil {
		return nil, errors.New("missing URL")
	}
	suffix, ok := strings.CutPrefix(requestURL.Path, "/api/router")
	if !ok || (suffix != "/config/kbs" && !strings.HasPrefix(suffix, "/config/kbs/")) {
		return nil, errors.New("path outside classifier API")
	}
	canonicalSuffix := path.Clean(suffix)
	if strings.HasSuffix(suffix, "/") && canonicalSuffix != "/" {
		canonicalSuffix += "/"
	}
	if canonicalSuffix != suffix {
		return nil, errors.New("non-canonical classifier API path")
	}
	target := *baseURL
	target.Path = strings.TrimRight(baseURL.Path, "/") + suffix
	target.RawPath = ""
	target.RawQuery = requestURL.RawQuery
	target.Fragment = ""
	return &target, nil
}

func newRouterClassifierProxyClient() *http.Client {
	dialer := &net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}
	transport := &http.Transport{
		Proxy:                  nil,
		DialContext:            dialer.DialContext,
		ForceAttemptHTTP2:      true,
		MaxIdleConns:           20,
		MaxIdleConnsPerHost:    10,
		IdleConnTimeout:        90 * time.Second,
		TLSHandshakeTimeout:    10 * time.Second,
		ResponseHeaderTimeout:  30 * time.Second,
		ExpectContinueTimeout:  time.Second,
		MaxResponseHeaderBytes: outboundMaxResponseHeaderBytes,
		TLSClientConfig:        &tls.Config{MinVersion: tls.VersionTLS12},
	}
	return &http.Client{
		Transport: transport,
		Timeout:   routerClassifierProxyTimeout,
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
}

func copyRouterClassifierProxyHeaders(dst, src http.Header) {
	connectionHeaders := make(map[string]struct{})
	for _, value := range src.Values("Connection") {
		for _, name := range strings.Split(value, ",") {
			if canonical := http.CanonicalHeaderKey(strings.TrimSpace(name)); canonical != "" {
				connectionHeaders[canonical] = struct{}{}
			}
		}
	}
	for key, values := range src {
		canonical := http.CanonicalHeaderKey(key)
		if _, blocked := connectionHeaders[canonical]; blocked || isRouterClassifierProxyHopHeader(canonical) {
			continue
		}
		dst.Del(canonical)
		for _, value := range values {
			dst.Add(canonical, value)
		}
	}
}

func isRouterClassifierProxyHopHeader(header string) bool {
	for _, blocked := range routerClassifierProxyHopHeaders {
		if strings.EqualFold(header, blocked) {
			return true
		}
	}
	return false
}

func stripRouterClassifierProxyCredentials(header http.Header) {
	for _, name := range [...]string{
		"Authorization",
		"Cookie",
		"Forwarded",
		"X-Csrf-Token",
		"X-Forwarded-For",
		"X-Forwarded-Host",
		"X-Forwarded-Proto",
		"X-Vsr-Auth-Mode",
	} {
		header.Del(name)
	}
}
