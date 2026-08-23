package managementauth

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"net/url"
	"testing"
	"time"
)

func TestVerifiedMTLSEvidenceRequiresListenerVerifiedLeaf(t *testing.T) {
	leaf := &x509.Certificate{Raw: []byte("leaf"), RawSubject: []byte("subject"), NotAfter: time.Now().Add(time.Hour)}
	if _, err := VerifiedMTLSEvidenceFromConnection(&tls.ConnectionState{
		PeerCertificates: []*x509.Certificate{leaf},
	}); !errors.Is(err, ErrAuthenticationDenied) {
		t.Fatalf("unverified peer certificate error = %v", err)
	}

	different := &x509.Certificate{Raw: []byte("different"), RawSubject: leaf.RawSubject, NotAfter: leaf.NotAfter}
	if _, err := VerifiedMTLSEvidenceFromConnection(&tls.ConnectionState{
		PeerCertificates: []*x509.Certificate{leaf}, VerifiedChains: [][]*x509.Certificate{{different}},
	}); !errors.Is(err, ErrAuthenticationDenied) {
		t.Fatalf("different verified leaf error = %v", err)
	}
}

func TestVerifiedMTLSEvidenceUsesExactSelectorPrecedence(t *testing.T) {
	spiffe := mustMTLSURI(t, "spiffe://cluster.example/workload/router")
	leaf := &x509.Certificate{
		Raw: []byte("leaf"), URIs: []*url.URL{spiffe}, DNSNames: []string{"IGNORED.EXAMPLE."},
		NotAfter: time.Date(2026, 9, 1, 2, 3, 4, 0, time.UTC),
	}
	verifiedCopy := *leaf
	evidence, err := VerifiedMTLSEvidenceFromConnection(&tls.ConnectionState{
		PeerCertificates: []*x509.Certificate{leaf}, VerifiedChains: [][]*x509.Certificate{{&verifiedCopy}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if evidence.MatcherKind != "spiffe_id" || evidence.MatcherValue != spiffe.String() ||
		!evidence.CertificateNotAfter.Equal(leaf.NotAfter) {
		t.Fatalf("evidence = %#v", evidence)
	}
}

func TestVerifiedMTLSEvidenceRejectsAmbiguousHighestPrioritySelector(t *testing.T) {
	leaf := &x509.Certificate{
		Raw: []byte("leaf"), URIs: []*url.URL{
			mustMTLSURI(t, "spiffe://cluster.example/workload/one"),
			mustMTLSURI(t, "spiffe://cluster.example/workload/two"),
		},
		DNSNames: []string{"fallback.example"}, NotAfter: time.Now().Add(time.Hour),
	}
	if _, err := VerifiedMTLSEvidenceFromConnection(&tls.ConnectionState{
		PeerCertificates: []*x509.Certificate{leaf}, VerifiedChains: [][]*x509.Certificate{{leaf}},
	}); !errors.Is(err, ErrAuthenticationDenied) {
		t.Fatalf("ambiguous SPIFFE selector error = %v", err)
	}
}

func mustMTLSURI(t *testing.T, value string) *url.URL {
	t.Helper()
	parsed, err := url.Parse(value)
	if err != nil {
		t.Fatal(err)
	}
	return parsed
}
