package proxy

import (
	"bufio"
	"context"
	"crypto/tls"
	"crypto/x509"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

func TestWebSocketTargetAddressHandlesDefaultPortsAndIPv6(t *testing.T) {
	t.Parallel()

	tests := []struct {
		rawURL string
		want   string
	}{
		{rawURL: "http://example.com", want: "example.com:80"},
		{rawURL: "https://example.com", want: "example.com:443"},
		{rawURL: "http://[::1]", want: "[::1]:80"},
		{rawURL: "https://[2001:db8::1]:9443", want: "[2001:db8::1]:9443"},
	}
	for _, test := range tests {
		t.Run(test.rawURL, func(t *testing.T) {
			t.Parallel()
			target, err := url.Parse(test.rawURL)
			if err != nil {
				t.Fatalf("url.Parse: %v", err)
			}
			got, err := webSocketTargetAddress(target)
			if err != nil {
				t.Fatalf("webSocketTargetAddress: %v", err)
			}
			if got != test.want {
				t.Fatalf("address = %q, want %q", got, test.want)
			}
		})
	}
}

func TestWebSocketTargetRejectsUnsupportedOrCredentialedURLs(t *testing.T) {
	t.Parallel()

	for _, rawURL := range []string{
		"ws://example.com",
		"wss://example.com",
		"ftp://example.com",
		"http:///missing-host",
		"https://user:pass@example.com",
		"https://example.com/base?token=ambiguous",
		"https://example.com/base#fragment",
	} {
		target, err := url.Parse(rawURL)
		if err == nil {
			err = validateWebSocketTarget(target)
		}
		if err == nil {
			t.Fatalf("target %q must be rejected", rawURL)
		}
	}
}

func TestDialWebSocketTargetUsesTLSForHTTPS(t *testing.T) {
	t.Parallel()

	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "secure")
	}))
	defer server.Close()
	target, err := url.Parse(server.URL)
	if err != nil {
		t.Fatalf("url.Parse: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if conn, dialErr := dialWebSocketTarget(ctx, target, time.Second); dialErr == nil {
		_ = conn.Close()
		t.Fatal("default TLS verification accepted an untrusted self-signed certificate")
	}

	roots := x509.NewCertPool()
	roots.AddCert(server.Certificate())
	tlsConfig := &tls.Config{RootCAs: roots, MinVersion: tls.VersionTLS12}
	conn, err := dialWebSocketTargetWithTLSConfig(ctx, target, time.Second, tlsConfig)
	if err != nil {
		t.Fatalf("dialWebSocketTargetWithTLSConfig: %v", err)
	}
	defer conn.Close()
	if _, ok := conn.(*tls.Conn); !ok {
		t.Fatalf("connection type = %T, want *tls.Conn", conn)
	}

	if _, writeErr := io.WriteString(conn, "GET / HTTP/1.1\r\nHost: "+target.Host+"\r\nConnection: close\r\n\r\n"); writeErr != nil {
		t.Fatalf("write TLS request: %v", writeErr)
	}
	response, err := http.ReadResponse(bufio.NewReader(conn), nil)
	if err != nil {
		t.Fatalf("read TLS response: %v", err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatalf("read response body: %v", err)
	}
	if !strings.Contains(string(body), "secure") {
		t.Fatalf("response body = %q, want secure", body)
	}
}

func TestDialWebSocketTargetRejectsHostnameMismatch(t *testing.T) {
	t.Parallel()

	server := httptest.NewTLSServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer server.Close()
	target, err := url.Parse(server.URL)
	if err != nil {
		t.Fatalf("url.Parse: %v", err)
	}
	roots := x509.NewCertPool()
	roots.AddCert(server.Certificate())
	tlsConfig := &tls.Config{
		RootCAs:    roots,
		ServerName: "hostname-mismatch.invalid",
		MinVersion: tls.VersionTLS12,
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if conn, err := dialWebSocketTargetWithTLSConfig(ctx, target, time.Second, tlsConfig); err == nil {
		_ = conn.Close()
		t.Fatal("TLS verification accepted a certificate for a different hostname")
	}
}
