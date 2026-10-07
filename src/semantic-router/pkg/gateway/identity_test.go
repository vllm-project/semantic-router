package gateway

import (
	"io"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// identitySeen sends one request with identity headers through a listener
// with trust and returns the identity headers the routing engine received.
func identitySeen(t *testing.T, trust IdentityTrust) (user, groups, configured string) {
	t.Helper()
	p := &fakeProcessor{}
	u := &fakeUpstream{resp: okUpstream(`{}`, http.Header{"Content-Type": {"application/json"}})}
	h, err := NewHandler(Options{
		Serving: Static(Serving{
			Engine:          routing.NewEngine(p, routing.DefaultOptions),
			Upstream:        u,
			IdentityHeaders: []string{"x-user"},
			TrustIdentity:   trust,
		}),
		Listener: "http-8899",
	})
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(h)
	defer server.Close()

	req, _ := http.NewRequest(http.MethodPost, server.URL+"/v1/chat/completions", strings.NewReader(`{"model":"m"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Authz-User-Id", "alice")
	req.Header.Set("X-Authz-User-Groups", "admins")
	req.Header.Set("X-User", "alice")
	// A client can claim any forwarded address; trust reads the TCP peer.
	req.Header.Set("X-Forwarded-For", "10.9.9.9")
	resp, err := server.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	header := p.headers[0]
	return header.Get("x-authz-user-id"), header.Get("x-authz-user-groups"), header.Get("x-user")
}

func TestListenerKeepsIdentityHeadersOnlyWhenItTrustsTheRequest(t *testing.T) {
	loopback := netip.MustParsePrefix("127.0.0.0/8")
	elsewhere := netip.MustParsePrefix("10.9.9.0/24")
	for _, tt := range []struct {
		name  string
		trust IdentityTrust
		kept  bool
	}{
		{name: "default drops them", trust: IdentityTrust{}, kept: false},
		{name: "trusted headers from any peer", trust: IdentityTrust{Headers: true}, kept: true},
		{name: "trusted peer network", trust: IdentityTrust{Headers: true, Peers: []netip.Prefix{elsewhere, loopback}}, kept: true},
		{name: "peer outside the networks, whatever X-Forwarded-For says", trust: IdentityTrust{Headers: true, Peers: []netip.Prefix{elsewhere}}, kept: false},
		{name: "peers without trusted headers", trust: IdentityTrust{Peers: []netip.Prefix{loopback}}, kept: false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			user, groups, configured := identitySeen(t, tt.trust)
			if kept := user == "alice" && groups == "admins" && configured == "alice"; kept != tt.kept {
				t.Fatalf("identity headers kept = %v (%q, %q, %q), want %v", kept, user, groups, configured, tt.kept)
			}
			if !tt.kept && (user != "" || groups != "" || configured != "") {
				t.Fatalf("an untrusted request carried identity headers: %q, %q, %q", user, groups, configured)
			}
		})
	}
}
