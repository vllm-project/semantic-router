package backendegress

import (
	"context"
	"net/netip"
	"strings"
	"testing"
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

type resolverStub struct{ addresses map[string][]netip.Addr }

func (r resolverStub) LookupNetIP(_ context.Context, _, host string) ([]netip.Addr, error) {
	return r.addresses[host], nil
}
