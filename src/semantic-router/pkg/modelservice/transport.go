package modelservice

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	unixBaseURL         = "http://model-runtime"
	maxIdleConnsPerHost = 64
	dialTimeout         = 2 * time.Second
	idleConnTimeout     = 90 * time.Second
)

// newHTTPClient returns the base URL and an HTTP client for an endpoint:
// unix:///path/to.sock, http://host:port or https://host:port.
func newHTTPClient(endpoint string) (string, *http.Client, error) {
	parsed, err := url.Parse(endpoint)
	if err != nil {
		return "", nil, fmt.Errorf("model runtime endpoint: %w", err)
	}
	transport := &http.Transport{
		MaxIdleConns:        maxIdleConnsPerHost,
		MaxIdleConnsPerHost: maxIdleConnsPerHost,
		IdleConnTimeout:     idleConnTimeout,
		ForceAttemptHTTP2:   false,
	}
	switch parsed.Scheme {
	case "unix":
		socket := parsed.Path
		dialer := &net.Dialer{Timeout: dialTimeout}
		transport.DialContext = func(ctx context.Context, _, _ string) (net.Conn, error) {
			return dialer.DialContext(ctx, "unix", socket)
		}
		return unixBaseURL, &http.Client{Transport: transport}, nil
	case "http", "https":
		transport.Proxy = http.ProxyFromEnvironment
		transport.DialContext = (&net.Dialer{Timeout: dialTimeout, KeepAlive: 30 * time.Second}).DialContext
		return strings.TrimRight(endpoint, "/"), &http.Client{Transport: transport}, nil
	default:
		return "", nil, fmt.Errorf("model runtime endpoint must use unix://, http:// or https://")
	}
}
