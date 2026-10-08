package gateway

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"io"
	"math/big"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"golang.org/x/net/http2"
)

func serveForTest(t *testing.T, handler http.Handler, opts ServerOptions) (string, func()) {
	t.Helper()
	ln, err := Listen("127.0.0.1:0", opts)
	if err != nil {
		t.Fatal(err)
	}
	server := NewServer(handler, opts)
	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = Serve(server, ln)
	}()
	return ln.Addr().String(), func() {
		_ = server.Close()
		<-done
	}
}

func protoHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, r.Proto)
	})
}

func get(t *testing.T, client *http.Client, url string) string {
	t.Helper()
	resp, err := client.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return string(body)
}

// writeCertificate writes a self-signed certificate for 127.0.0.1 and its key.
func writeCertificate(t *testing.T) (certFile, keyFile string, pool *x509.CertPool) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	template := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "gateway-test"},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
		IPAddresses:  []net.IP{net.ParseIP("127.0.0.1")},
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	certFile, keyFile = filepath.Join(dir, "tls.crt"), filepath.Join(dir, "tls.key")
	if err = os.WriteFile(certFile, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o600); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(keyFile, pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}), 0o600); err != nil {
		t.Fatal(err)
	}
	leaf, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatal(err)
	}
	pool = x509.NewCertPool()
	pool.AddCert(leaf)
	return certFile, keyFile, pool
}

func TestServerSpeaksHTTP1AndH2C(t *testing.T) {
	addr, stop := serveForTest(t, protoHandler(), ServerOptions{})
	defer stop()

	if got := get(t, &http.Client{}, "http://"+addr); got != "HTTP/1.1" {
		t.Fatalf("HTTP/1.1 client got %q", got)
	}
	h2c := &http.Client{Transport: &http2.Transport{
		AllowHTTP: true,
		DialTLS: func(network, addr string, _ *tls.Config) (net.Conn, error) {
			return net.Dial(network, addr)
		},
	}}
	if got := get(t, h2c, "http://"+addr); got != "HTTP/2.0" {
		t.Fatalf("h2c client got %q", got)
	}
}

func TestServerServesTLSWithALPN(t *testing.T) {
	certFile, keyFile, pool := writeCertificate(t)
	tlsConfig, err := LoadTLS(certFile, keyFile, nil)
	if err != nil {
		t.Fatal(err)
	}
	addr, stop := serveForTest(t, protoHandler(), ServerOptions{TLS: tlsConfig})
	defer stop()

	h2 := &http.Client{Transport: &http.Transport{TLSClientConfig: &tls.Config{RootCAs: pool}, ForceAttemptHTTP2: true}}
	if got := get(t, h2, "https://"+addr); got != "HTTP/2.0" {
		t.Fatalf("an HTTP/2 client over TLS got %q", got)
	}
	h1 := &http.Client{Transport: &http.Transport{
		TLSClientConfig: &tls.Config{RootCAs: pool},
		TLSNextProto:    map[string]func(string, *tls.Conn) http.RoundTripper{},
	}}
	if got := get(t, h1, "https://"+addr); got != "HTTP/1.1" {
		t.Fatalf("an HTTP/1.1 client over TLS got %q", got)
	}
	if resp, err := (&http.Client{}).Get("http://" + addr); err == nil {
		resp.Body.Close()
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("a cleartext request to a TLS listener got %d", resp.StatusCode)
		}
	}
	old := &tls.Config{RootCAs: pool, MaxVersion: tls.VersionTLS11}
	if resp, err := (&http.Client{Transport: &http.Transport{TLSClientConfig: old}}).Get("https://" + addr); err == nil {
		resp.Body.Close()
		t.Fatal("TLS 1.1 must be refused, as Envoy refuses it")
	}
}

func TestLoadTLSRejectsAMismatchedKey(t *testing.T) {
	certFile, _, _ := writeCertificate(t)
	_, otherKey, _ := writeCertificate(t)
	if _, err := LoadTLS(certFile, otherKey, nil); err == nil {
		t.Fatal("a key that does not match the certificate must be rejected")
	}
	if _, err := LoadTLS(filepath.Join(t.TempDir(), "missing.crt"), otherKey, nil); err == nil {
		t.Fatal("a missing certificate must be rejected")
	}
}

func TestListenBoundsConcurrentConnections(t *testing.T) {
	addr, stop := serveForTest(t, protoHandler(), ServerOptions{MaxConnections: 1})
	defer stop()

	first, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = io.WriteString(first, "GET / HTTP/1.1\r\nHost: x\r\n\r\n"); err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 64)
	if _, err = first.Read(buf); err != nil {
		t.Fatalf("the first connection must be served: %v", err)
	}

	second, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	if _, err := io.WriteString(second, "GET / HTTP/1.1\r\nHost: x\r\n\r\n"); err != nil {
		t.Fatal(err)
	}
	_ = second.SetReadDeadline(time.Now().Add(200 * time.Millisecond))
	if _, err := second.Read(buf); err == nil {
		t.Fatal("a connection over the limit must wait for a free slot")
	}

	_ = first.Close()
	_ = second.SetReadDeadline(time.Now().Add(5 * time.Second))
	if _, err := second.Read(buf); err != nil {
		t.Fatalf("the waiting connection must be served once a slot frees: %v", err)
	}
}
