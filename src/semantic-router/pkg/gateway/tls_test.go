package gateway

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func handshake(addr string, pool *x509.CertPool) error {
	conn, err := tls.Dial("tcp", addr, &tls.Config{RootCAs: pool, MinVersion: tls.VersionTLS12})
	if err != nil {
		return err
	}
	return conn.Close()
}

func copyPair(t *testing.T, certFile, keyFile, liveCert, liveKey string, at time.Time) {
	t.Helper()
	for from, to := range map[string]string{certFile: liveCert, keyFile: liveKey} {
		data, err := os.ReadFile(from)
		if err != nil {
			t.Fatal(err)
		}
		// Keep the file sizes the same across renewals even if the encoded
		// signatures are different lengths.
		data = append(data, bytes.Repeat([]byte{'\n'}, 2048-len(data))...)
		if err := os.WriteFile(to, data, 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(to, at, at); err != nil {
			t.Fatal(err)
		}
	}
}

func TestLoadTLSServesARotatedCertificateToNewConnections(t *testing.T) {
	previous := keyPairCheckInterval
	keyPairCheckInterval = 0
	t.Cleanup(func() { keyPairCheckInterval = previous })

	certA, keyA, poolA := writeCertificate(t)
	certB, keyB, poolB := writeCertificate(t)
	live := t.TempDir()
	liveCert, liveKey := filepath.Join(live, "tls.crt"), filepath.Join(live, "tls.key")
	start := time.Now().Add(-time.Hour)
	copyPair(t, certA, keyA, liveCert, liveKey, start)

	var mu sync.Mutex
	var reloads []error
	reported := func() []error {
		mu.Lock()
		defer mu.Unlock()
		return append([]error(nil), reloads...)
	}
	tlsConfig, err := LoadTLS(liveCert, liveKey, func(err error) {
		mu.Lock()
		defer mu.Unlock()
		reloads = append(reloads, err)
	})
	if err != nil {
		t.Fatal(err)
	}
	addr, stop := serveForTest(t, protoHandler(), ServerOptions{TLS: tlsConfig})
	defer stop()

	if err := handshake(addr, poolA); err != nil {
		t.Fatalf("the first certificate: %v", err)
	}
	if err := handshake(addr, poolB); err == nil {
		t.Fatal("the second certificate must not serve before it is written")
	}

	copyPair(t, certB, keyB, liveCert+".new", liveKey+".new", start)
	if err := os.Rename(liveCert+".new", liveCert); err != nil {
		t.Fatal(err)
	}
	if err := handshake(addr, poolA); err != nil {
		t.Fatalf("an incomplete rotation must leave the previous pair serving: %v", err)
	}
	if err := os.Rename(liveKey+".new", liveKey); err != nil {
		t.Fatal(err)
	}
	if err := handshake(addr, poolB); err != nil {
		t.Fatalf("a new connection after an atomic rotation with equal size and mtime: %v", err)
	}
	if got := reported(); len(got) != 2 || got[0] == nil || got[1] != nil {
		t.Fatalf("reloads = %v, want the incomplete rotation followed by a success", got)
	}

	if err := os.WriteFile(liveCert, []byte("not a certificate"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := handshake(addr, poolB); err != nil {
		t.Fatalf("a pair that fails to load must leave the previous one serving: %v", err)
	}
	if err := os.Remove(liveCert); err != nil {
		t.Fatal(err)
	}
	for range 2 {
		if err := handshake(addr, poolB); err != nil {
			t.Fatalf("a missing file must leave the previous pair serving: %v", err)
		}
	}
	if got := reported(); len(got) != 4 || got[2] == nil || got[3] == nil {
		t.Fatalf("reloads = %v, want the failed load and the missing file reported once each", got)
	}
}
