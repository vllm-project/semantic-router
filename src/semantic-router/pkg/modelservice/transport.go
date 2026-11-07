package modelservice

import (
	"context"
	"crypto/tls"
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

// authorizingTransport sets an attached runtime's headers and credential on
// each request.
type authorizingTransport struct {
	next      http.RoundTripper
	authorize func(context.Context, *http.Request) error
}

func (t authorizingTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	request = request.Clone(request.Context())
	if err := t.authorize(request.Context(), request); err != nil {
		return nil, err
	}
	return t.next.RoundTrip(request)
}

// newHTTPClient returns the base URL and an HTTP client for an endpoint:
// unix:///path/to.sock, http://host:port or https://host:port.
func newHTTPClient(endpoint string, tlsConfig *tls.Config) (string, *http.Client, error) {
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
		if parsed.Scheme == "https" {
			transport.TLSClientConfig = tlsConfig
		}
		transport.Proxy = http.ProxyFromEnvironment
		transport.DialContext = (&net.Dialer{Timeout: dialTimeout, KeepAlive: 30 * time.Second}).DialContext
		return strings.TrimRight(endpoint, "/"), &http.Client{Transport: transport}, nil
	default:
		return "", nil, fmt.Errorf("model runtime endpoint must use unix://, http:// or https://")
	}
}
