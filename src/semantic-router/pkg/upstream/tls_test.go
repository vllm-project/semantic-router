package upstream

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// tlsBackend starts an HTTPS test server whose certificate names
// example.com, and reports the SNI each handshake carried.
func tlsBackend(t *testing.T) (*httptest.Server, *x509.CertPool, chan string) {
	t.Helper()
	sni := make(chan string, 8)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, r.Proto)
	}))
	server.TLS = &tls.Config{GetConfigForClient: func(hello *tls.ClientHelloInfo) (*tls.Config, error) {
		sni <- hello.ServerName
		return nil, nil
	}}
	// Rejected handshakes are the point of some tests; keep them out of the log.
	server.Config.ErrorLog = log.New(io.Discard, "", 0)
	server.StartTLS()
	t.Cleanup(server.Close)
	roots := x509.NewCertPool()
	roots.AddCert(server.Certificate())
	return server, roots, sni
}

// dialTo sends every connection to address, whatever name was dialed.
func dialTo(address string) DialFunc {
	return func(ctx context.Context, network, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, address)
	}
}

func TestDoVerifiesTheUpstreamCertificateAgainstSNI(t *testing.T) {
	server, roots, sni := tlsBackend(t)
	ep := EndpointSpec{Name: "secure", Scheme: "https", Host: "example.com", Port: 443, Weight: 1}
	spec := clusterOf("secure", ep)
	spec.TLS = &TLSSpec{ServerName: "example.com"}
	set := newSet(t, Options{RootCAs: roots, Dial: dialTo(server.Listener.Addr().String())}, spec)

	resp, err := set.Do(t.Context(), post("secure"))
	if err != nil {
		t.Fatal(err)
	}
	body, err := readAll(t, resp)
	if err != nil || body != "HTTP/1.1" {
		t.Fatalf("body = %q, err = %v; want an HTTP/1.1 exchange", body, err)
	}
	if got := <-sni; got != "example.com" {
		t.Fatalf("SNI = %q, want example.com", got)
	}
}

func TestDoRejectsACertificateForAnotherName(t *testing.T) {
	server, roots, _ := tlsBackend(t)
	ep := EndpointSpec{Name: "wrong", Scheme: "https", Host: "api.example.test", Port: 443, Weight: 1}
	spec := clusterOf("wrong-name", ep)
	spec.TLS = &TLSSpec{ServerName: "api.example.test"}
	set := newSet(t, Options{RootCAs: roots, Dial: dialTo(server.Listener.Addr().String())}, spec)

	resp, err := set.Do(t.Context(), post("wrong-name"))
	if err != nil {
		t.Fatal(err)
	}
	if KindOf(resp.Local) != KindConnectFailure {
		t.Fatalf("failure = %v, want a connect failure", resp.Local)
	}
	var verifyErr *tls.CertificateVerificationError
	if !errorsAs(error(resp.Local), &verifyErr) {
		t.Fatalf("failure = %v, want a certificate verification failure", resp.Local)
	}
	body, _ := readAll(t, resp)
	if !strings.HasPrefix(body, "upstream connect error or disconnect/reset before headers. reset reason: "+
		"remote connection failure, transport failure reason: TLS_error:|") {
		t.Fatalf("local reply = %q", body)
	}
}

func TestDoRejectsAnUntrustedCertificate(t *testing.T) {
	server, _, _ := tlsBackend(t)
	ep := EndpointSpec{Name: "untrusted", Scheme: "https", Host: "example.com", Port: 443, Weight: 1}
	spec := clusterOf("untrusted", ep)
	spec.TLS = &TLSSpec{ServerName: "example.com"}
	set := newSet(t, Options{RootCAs: x509.NewCertPool(), Dial: dialTo(server.Listener.Addr().String())}, spec)
	resp, err := set.Do(t.Context(), post("untrusted"))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = readAll(t, resp)
	if KindOf(resp.Local) != KindConnectFailure {
		t.Fatalf("failure = %v, want a connect failure", resp.Local)
	}
}
