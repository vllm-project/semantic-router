package backendegress

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"math/big"
	"net/netip"
	"strings"
	"testing"
	"time"
)

const testPolicy = `version: v1
schemes: [https, http]
hosts:
  - host: api.example.com
    ports: [443]
  - host: "*.models.example.com"
    ports: [443]
  - host: model.internal
    ports: [8000]
    allow_cidrs: [10.20.0.0/16]
`

func TestPolicyNormalizesAndAuthorizesExactTargets(t *testing.T) {
	policy, err := Parse([]byte(testPolicy))
	if err != nil {
		t.Fatal(err)
	}
	for _, origin := range []string{"https://api.example.com", "https://tenant.models.example.com"} {
		if _, err := policy.AuthorizeOrigin(origin); err != nil {
			t.Fatalf("AuthorizeOrigin(%q): %v", origin, err)
		}
	}
	for _, origin := range []string{
		"https://models.example.com", "https://api.example.com:8443",
		"https://other.example.com", "HTTPS://api.example.com",
	} {
		if _, err := policy.AuthorizeOrigin(origin); err == nil {
			t.Fatalf("AuthorizeOrigin(%q) succeeded", origin)
		}
	}
}

func TestGuardRejectsMixedDNSRebindingAnswers(t *testing.T) {
	policy, _ := Parse([]byte(testPolicy))
	guard := Guard{Policy: policy, Resolver: resolverStub{addresses: map[string][]netip.Addr{
		"api.example.com": {netip.MustParseAddr("203.0.113.10"), netip.MustParseAddr("127.0.0.1")},
	}}}
	if _, err := guard.Resolve(context.Background(), "https://api.example.com"); err == nil ||
		!strings.Contains(err.Error(), "denied address") {
		t.Fatalf("mixed DNS response error = %v", err)
	}
}

func TestGuardAllowsOnlyExplicitPrivateCIDR(t *testing.T) {
	policy, _ := Parse([]byte(testPolicy))
	guard := Guard{Policy: policy, Resolver: resolverStub{addresses: map[string][]netip.Addr{
		"model.internal": {netip.MustParseAddr("10.20.1.4")},
	}}}
	resolved, err := guard.Resolve(context.Background(), "http://model.internal:8000")
	if err != nil || len(resolved.Addresses) != 1 {
		t.Fatalf("explicit private target = %+v, %v", resolved, err)
	}
	guard.Resolver = resolverStub{addresses: map[string][]netip.Addr{
		"model.internal": {netip.MustParseAddr("10.21.1.4")},
	}}
	if _, err := guard.Resolve(context.Background(), "http://model.internal:8000"); err == nil {
		t.Fatal("unlisted private CIDR was allowed")
	}
}

func TestPolicyRejectsUnknownFieldsAndUnsafeCIDRs(t *testing.T) {
	for _, document := range []string{
		"version: v1\nschemes: [https]\nhosts: [{host: api.example.com, ports: [443], typo: true}]\n",
		"version: v1\nschemes: [https]\nhosts: [{host: api.example.com, ports: [443], allow_cidrs: [169.254.0.0/16]}]\n",
		"version: v1\nschemes: [ftp]\nhosts: [{host: api.example.com, ports: [21]}]\n",
	} {
		if _, err := Parse([]byte(document)); err == nil {
			t.Fatalf("unsafe policy succeeded:\n%s", document)
		}
	}
}

func TestPolicyAllowsExactContainerServiceLabelsButNotWildcardPatterns(t *testing.T) {
	policy, err := Parse([]byte(`version: v1
schemes: [https]
hosts:
  - {host: team_a-dashboard, ports: [8743], allow_cidrs: [172.24.0.0/16]}
  - {host: _team.internal, ports: [8743], allow_cidrs: [172.24.0.0/16]}
  - {host: team_.internal, ports: [8743], allow_cidrs: [172.24.0.0/16]}
  - {host: team_.blue-vllm-sr-dashboard-container, ports: [8743], allow_cidrs: [172.24.0.0/16]}
`))
	if err != nil {
		t.Fatal(err)
	}
	for _, origin := range []string{
		"https://team_a-dashboard:8743",
		"https://_team.internal:8743",
		"https://team_.internal:8743",
		"https://team_.blue-vllm-sr-dashboard-container:8743",
	} {
		if _, err := policy.AuthorizeOrigin(origin); err != nil {
			t.Fatalf("exact container service label %q = %v", origin, err)
		}
	}
	if _, err := Parse([]byte(`version: v1
schemes: [https]
hosts:
  - {host: "*.team_a.internal", ports: [443]}
`)); err == nil {
		t.Fatal("wildcard policy accepted a non-DNS service label")
	}
}

func TestExactContainerServiceLabelMatchesTLSIdentity(t *testing.T) {
	const hostname = "team_a-vllm-sr-dashboard-container"
	policy, err := Parse([]byte(`version: v1
schemes: [https]
hosts:
  - {host: team_a-vllm-sr-dashboard-container, ports: [8743], allow_cidrs: [172.24.0.0/16]}
`))
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := (Guard{
		Policy: policy,
		Resolver: resolverStub{addresses: map[string][]netip.Addr{
			hostname: {netip.MustParseAddr("172.24.0.4")},
		}},
	}).Resolve(context.Background(), "https://"+hostname+":8743")
	if err != nil || resolved.ServerName != hostname || len(resolved.Addresses) != 1 {
		t.Fatalf("resolved exact container service = %+v, %v", resolved, err)
	}
	publicKey, privateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	template := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		NotBefore:    now.Add(-time.Minute),
		NotAfter:     now.Add(time.Hour),
		DNSNames:     []string{hostname},
	}
	encoded, err := x509.CreateCertificate(rand.Reader, template, template, publicKey, privateKey)
	if err != nil {
		t.Fatal(err)
	}
	certificate, err := x509.ParseCertificate(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if err := certificate.VerifyHostname(resolved.ServerName); err != nil {
		t.Fatalf("exact container TLS identity = %v", err)
	}
	if err := certificate.VerifyHostname("other-service"); err == nil {
		t.Fatal("container TLS identity matched another service")
	}
}

func TestOverlayReplacesStalePrivateHostException(t *testing.T) {
	base, err := Parse([]byte(`version: v1
schemes: [https]
hosts:
  - {host: api.example.com, ports: [443]}
  - {host: dashboard.internal, ports: [8743], allow_cidrs: [172.24.0.0/16]}
`))
	if err != nil {
		t.Fatal(err)
	}
	overrides, err := Parse([]byte(`version: v1
schemes: [https]
hosts:
  - {host: dashboard.internal, ports: [8743], allow_cidrs: [172.31.0.0/16]}
`))
	if err != nil {
		t.Fatal(err)
	}
	policy, err := Overlay(base, overrides)
	if err != nil {
		t.Fatal(err)
	}
	guard := Guard{Policy: policy, Resolver: resolverStub{addresses: map[string][]netip.Addr{
		"dashboard.internal": {netip.MustParseAddr("172.31.0.4")},
	}}}
	if _, err := guard.Resolve(context.Background(), "https://dashboard.internal:8743"); err != nil {
		t.Fatalf("current private issuer = %v", err)
	}
	guard.Resolver = resolverStub{addresses: map[string][]netip.Addr{
		"dashboard.internal": {netip.MustParseAddr("172.24.0.4")},
	}}
	if _, err := guard.Resolve(context.Background(), "https://dashboard.internal:8743"); err == nil {
		t.Fatal("stale private issuer range remained authorized")
	}
	if _, err := policy.AuthorizeOrigin("https://api.example.com"); err != nil {
		t.Fatalf("base public origin was lost: %v", err)
	}
}

func TestOverlayCannotExpandBaseSchemes(t *testing.T) {
	base, _ := Parse([]byte("version: v1\nschemes: [https]\nhosts: [{host: api.example.com, ports: [443]}]\n"))
	overrides, _ := Parse([]byte("version: v1\nschemes: [http]\nhosts: [{host: issuer.internal, ports: [80], allow_cidrs: [10.20.0.0/16]}]\n"))
	if _, err := Overlay(base, overrides); err == nil || !strings.Contains(err.Error(), "not allowed") {
		t.Fatalf("scheme-expanding overlay error = %v", err)
	}
}

func TestOverlayRejectsWildcardOverrides(t *testing.T) {
	base, _ := Parse([]byte("version: v1\nschemes: [https]\nhosts: [{host: api.example.com, ports: [443]}]\n"))
	overrides, _ := Parse([]byte("version: v1\nschemes: [https]\nhosts: [{host: '*.internal.example.com', ports: [443]}]\n"))
	if _, err := Overlay(base, overrides); err == nil || !strings.Contains(err.Error(), "exact host identities") {
		t.Fatalf("wildcard overlay error = %v", err)
	}
}

type resolverStub struct{ addresses map[string][]netip.Addr }

func (r resolverStub) LookupNetIP(_ context.Context, _, host string) ([]netip.Addr, error) {
	return r.addresses[host], nil
}
