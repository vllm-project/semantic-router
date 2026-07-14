package proxy

import (
	"crypto/tls"
	"net"
	"net/http"
	"time"
)

const (
	proxyDialTimeout         = 10 * time.Second
	proxyKeepAlive           = 30 * time.Second
	proxyTLSHandshakeTimeout = 10 * time.Second
	// Router inference can legitimately queue or cold-start before emitting
	// response headers. Keep this bounded while matching the default five-minute
	// model request contract instead of imposing an interactive HTTP timeout.
	proxyResponseHeaderTimeout  = 5 * time.Minute
	proxyExpectContinueTimeout  = time.Second
	proxyIdleConnTimeout        = 90 * time.Second
	proxyMaxResponseHeaderBytes = 1 << 20
	proxyMaxIdleConns           = 100
	proxyMaxIdleConnsPerHost    = 16
)

// dashboardProxyTransport is process-owned and shared by all fixed-target
// reverse proxies. In particular, Proxy remains nil: these east-west requests
// may contain prompts, upstream credentials, or control-plane data and must not
// inherit ambient HTTP_PROXY/HTTPS_PROXY settings.
var dashboardProxyTransport = newDashboardProxyTransport()

func newDashboardProxyTransport() *http.Transport {
	dialer := &net.Dialer{
		Timeout:   proxyDialTimeout,
		KeepAlive: proxyKeepAlive,
	}
	return &http.Transport{
		Proxy:                  nil,
		DialContext:            dialer.DialContext,
		ForceAttemptHTTP2:      true,
		MaxIdleConns:           proxyMaxIdleConns,
		MaxIdleConnsPerHost:    proxyMaxIdleConnsPerHost,
		IdleConnTimeout:        proxyIdleConnTimeout,
		TLSHandshakeTimeout:    proxyTLSHandshakeTimeout,
		ResponseHeaderTimeout:  proxyResponseHeaderTimeout,
		ExpectContinueTimeout:  proxyExpectContinueTimeout,
		MaxResponseHeaderBytes: proxyMaxResponseHeaderBytes,
		TLSClientConfig: &tls.Config{
			MinVersion: tls.VersionTLS12,
		},
	}
}
