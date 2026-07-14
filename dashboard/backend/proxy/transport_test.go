package proxy

import (
	"context"
	"crypto/tls"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestReverseProxyUsesBoundedDirectTransport(t *testing.T) {
	proxyHandler, err := NewReverseProxy("https://upstream.example", "/api/router", true)
	if err != nil {
		t.Fatalf("NewReverseProxy: %v", err)
	}
	transport, ok := proxyHandler.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("transport type = %T, want *http.Transport", proxyHandler.Transport)
	}
	if transport.Proxy != nil {
		t.Fatal("reverse proxy transport must not inherit ambient HTTP proxies")
	}
	if transport.DialContext == nil || transport.ResponseHeaderTimeout != 5*time.Minute ||
		transport.TLSHandshakeTimeout <= 0 || transport.IdleConnTimeout <= 0 ||
		transport.MaxResponseHeaderBytes <= 0 || transport.MaxIdleConns <= 0 ||
		transport.MaxIdleConnsPerHost <= 0 {
		t.Fatalf("reverse proxy transport is missing a resource bound: %#v", transport)
	}
	if transport.TLSClientConfig == nil || transport.TLSClientConfig.MinVersion < tls.VersionTLS12 {
		t.Fatalf("TLS minimum = %#v, want TLS 1.2 or newer", transport.TLSClientConfig)
	}
}

func TestReverseProxyIgnoresAmbientHTTPProxy(t *testing.T) {
	var attackerRequests atomic.Int32
	attacker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		attackerRequests.Add(1)
		http.Error(w, "intercepted", http.StatusBadGateway)
	}))
	defer attacker.Close()

	receivedBody := make(chan string, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read upstream request: %v", err)
		}
		receivedBody <- string(body)
		w.WriteHeader(http.StatusNoContent)
	}))
	defer upstream.Close()

	t.Setenv("HTTP_PROXY", attacker.URL)
	t.Setenv("HTTPS_PROXY", attacker.URL)
	t.Setenv("NO_PROXY", "")

	proxyHandler, err := NewReverseProxy("http://router.internal.test", "/api/router", true)
	if err != nil {
		t.Fatalf("NewReverseProxy: %v", err)
	}
	transport, ok := proxyHandler.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("transport type = %T, want *http.Transport", proxyHandler.Transport)
	}
	testTransport := transport.Clone()
	upstreamURL, err := url.Parse(upstream.URL)
	if err != nil {
		t.Fatalf("parse upstream URL: %v", err)
	}
	attackerURL, err := url.Parse(attacker.URL)
	if err != nil {
		t.Fatalf("parse attacker URL: %v", err)
	}
	dialer := &net.Dialer{}
	testTransport.DialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		switch address {
		case "router.internal.test:80":
			return dialer.DialContext(ctx, network, upstreamURL.Host)
		case attackerURL.Host:
			return dialer.DialContext(ctx, network, attackerURL.Host)
		default:
			return nil, &net.DNSError{Err: "unexpected test destination", Name: address}
		}
	}
	proxyHandler.Transport = testTransport

	request := httptest.NewRequest(
		http.MethodPost,
		"http://dashboard.test/api/router/v1/chat/completions",
		strings.NewReader("sensitive prompt"),
	)
	request.Header.Set("Authorization", "Bearer upstream-secret")
	recorder := httptest.NewRecorder()
	proxyHandler.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusNoContent {
		t.Fatalf("proxy status = %d, want %d", recorder.Code, http.StatusNoContent)
	}
	if got := <-receivedBody; got != "sensitive prompt" {
		t.Fatalf("upstream body = %q", got)
	}
	if got := attackerRequests.Load(); got != 0 {
		t.Fatalf("ambient HTTP proxy received %d requests, want 0", got)
	}
}
