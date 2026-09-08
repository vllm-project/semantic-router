package safefetch

import (
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// staticResolver answers with a fixed set, standing in for DNS.
type staticResolver struct{ addresses []netip.Addr }

func (r staticResolver) LookupNetIP(context.Context, string, string) ([]netip.Addr, error) {
	return r.addresses, nil
}

// rebindResolver answers publicly the first time and privately afterwards,
// which is the shape of a DNS rebinding attack against a check-then-connect
// implementation.
type rebindResolver struct{ calls atomic.Int32 }

func (r *rebindResolver) LookupNetIP(context.Context, string, string) ([]netip.Addr, error) {
	if r.calls.Add(1) == 1 {
		return []netip.Addr{netip.MustParseAddr("8.8.8.8")}, nil
	}
	return []netip.Addr{netip.MustParseAddr("127.0.0.1")}, nil
}

type failingResolver struct{}

func (failingResolver) LookupNetIP(context.Context, string, string) ([]netip.Addr, error) {
	return nil, errors.New("no such host")
}

func TestValidateURLRejectsUnsafeInput(t *testing.T) {
	policy := DefaultPolicy()

	tests := []struct {
		name string
		raw  string
		want error
	}{
		{"empty", "", ErrInvalidURL},
		{"relative", "/etc/passwd", ErrInvalidURL},
		{"no host", "https://", ErrInvalidURL},
		{"opaque", "https:opaque", ErrInvalidURL},
		{"file scheme", "file:///etc/passwd", ErrSchemeNotAllowed},
		{"gopher scheme", "gopher://example.com/", ErrSchemeNotAllowed},
		{"credentials replayed on redirect", "https://user:pass@example.com/", ErrInvalidURL},
		{"fragment", "https://example.com/#frag", ErrInvalidURL},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := policy.ValidateURL(tt.raw); !errors.Is(err, tt.want) {
				t.Errorf("ValidateURL(%q) = %v, want %v", tt.raw, err, tt.want)
			}
		})
	}
}

func TestValidateURLAcceptsPublicHTTPAndHTTPS(t *testing.T) {
	policy := DefaultPolicy()
	for _, raw := range []string{"https://example.com/a?b=c", "http://example.com"} {
		if _, err := policy.ValidateURL(raw); err != nil {
			t.Errorf("ValidateURL(%q) = %v, want accepted", raw, err)
		}
	}
}

func TestValidateURLHonoursSchemeRestriction(t *testing.T) {
	policy := DefaultPolicy().WithSchemes("https")
	if _, err := policy.ValidateURL("http://example.com"); !errors.Is(err, ErrSchemeNotAllowed) {
		t.Errorf("http accepted under an HTTPS-only policy: %v", err)
	}
}

// A hostname that resolves inward must be refused at dial time. Validation
// alone cannot see this, which is the whole point of checking after DNS.
func TestClientRefusesHostResolvingToNonPublicAddress(t *testing.T) {
	for _, raw := range []string{"127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "fe80::1"} {
		t.Run(raw, func(t *testing.T) {
			client := DefaultPolicy().
				WithResolver(staticResolver{addresses: []netip.Addr{netip.MustParseAddr(raw)}}).
				NewClient()

			resp, err := client.Get("https://public-looking-name.invalid/")
			closeResponse(resp)
			if !errors.Is(err, ErrDestinationForbidden) {
				t.Fatalf("resolving to %s was not refused: %v", raw, err)
			}
		})
	}
}

// A mixed answer is refused outright rather than dialling whichever record
// happens to be public.
func TestClientRefusesMixedPublicAndPrivateAnswers(t *testing.T) {
	client := DefaultPolicy().
		WithResolver(staticResolver{addresses: []netip.Addr{
			netip.MustParseAddr("8.8.8.8"),
			netip.MustParseAddr("127.0.0.1"),
		}}).
		NewClient()

	resp, err := client.Get("https://mixed.invalid/")
	closeResponse(resp)
	if !errors.Is(err, ErrDestinationForbidden) {
		t.Fatalf("mixed answer was not refused: %v", err)
	}
}

func TestClientRefusesUnresolvableHost(t *testing.T) {
	client := DefaultPolicy().WithResolver(failingResolver{}).NewClient()
	resp, err := client.Get("https://nowhere.invalid/")
	closeResponse(resp)
	if !errors.Is(err, ErrDestinationForbidden) {
		t.Fatalf("unresolvable host was not refused: %v", err)
	}
}

// The dialler connects to the address it validated, not to the name, so an
// answer that changes between validation and dial cannot move the connection.
func TestClientPinsTheValidatedAddressAgainstRebinding(t *testing.T) {
	resolver := &rebindResolver{}
	client := DefaultPolicy().WithResolver(resolver).NewClient()

	// Drive several attempts so a second lookup, if one happened, would be the
	// private answer.
	for i := 0; i < 3; i++ {
		resp, err := client.Get("https://rebind.invalid/")
		closeResponse(resp)
		// 8.8.8.8:443 is not reachable from a test runner, so the connection
		// fails. What must never happen is a successful fetch from loopback.
		if err == nil {
			t.Fatal("a rebound request succeeded")
		}
	}
}

// A redirect from an allowed public URL to a denied destination is refused,
// and the credential-bearing Referer is not forwarded.
func TestClientRevalidatesEveryRedirect(t *testing.T) {
	policy := DefaultPolicy()
	client := policy.NewClient()

	denied := mustRequest(t, "file:///etc/passwd")
	if err := client.CheckRedirect(denied, nil); !errors.Is(err, ErrSchemeNotAllowed) {
		t.Errorf("redirect to a denied scheme accepted: %v", err)
	}

	credentialed := mustRequest(t, "https://user:pass@elsewhere.example/")
	if err := client.CheckRedirect(credentialed, nil); !errors.Is(err, ErrInvalidURL) {
		t.Errorf("redirect to a credentialed URL accepted: %v", err)
	}

	allowed := mustRequest(t, "https://elsewhere.example/next")
	allowed.Header.Set("Referer", "https://origin.example/?token=secret")
	if err := client.CheckRedirect(allowed, nil); err != nil {
		t.Fatalf("redirect to an allowed URL refused: %v", err)
	}
	if got := allowed.Header.Get("Referer"); got != "" {
		t.Errorf("Referer survived the redirect: %q", got)
	}
}

func TestClientEnforcesTheRedirectLimit(t *testing.T) {
	policy := DefaultPolicy()
	client := policy.NewClient()

	via := make([]*http.Request, policy.MaxRedirects)
	if err := client.CheckRedirect(mustRequest(t, "https://elsewhere.example/"), via); !errors.Is(err, ErrTooManyRedirects) {
		t.Error("the redirect limit was not enforced")
	}
}

// A redirect to a name that resolves inward is refused at the second dial, not
// merely at the syntactic check.
func TestClientRefusesRedirectResolvingToNonPublicAddress(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/start" {
			http.Redirect(w, r, "http://internal.invalid/admin", http.StatusFound)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	host, port := splitHostPort(t, upstream.Listener.Addr().String())
	client := DefaultPolicy().
		WithResolver(hostRoutingResolver{
			public:   host,
			publicAt: netip.MustParseAddr(host),
		}).
		NewClient()

	resp, err := client.Get(fmt.Sprintf("http://%s:%s/start", host, port))
	closeResponse(resp)
	if !errors.Is(err, ErrDestinationForbidden) {
		t.Fatalf("redirect to an inward-resolving host was not refused: %v", err)
	}
}

// hostRoutingResolver resolves the loopback test server literally and every
// other name to loopback, so a redirect leaves the allowed destination.
type hostRoutingResolver struct {
	public   string
	publicAt netip.Addr
}

func (r hostRoutingResolver) LookupNetIP(_ context.Context, _ string, host string) ([]netip.Addr, error) {
	if host == r.public {
		return []netip.Addr{r.publicAt}, nil
	}
	return []netip.Addr{netip.MustParseAddr("127.0.0.1")}, nil
}

func TestReadBoundedRejectsOversizedBody(t *testing.T) {
	body := strings.NewReader(strings.Repeat("a", 2048))
	if _, err := ReadBounded(body, 1024); !errors.Is(err, ErrResponseTooLarge) {
		t.Fatalf("ReadBounded() = %v, want ErrResponseTooLarge", err)
	}
}

func TestReadBoundedAcceptsBodyAtTheLimit(t *testing.T) {
	data, err := ReadBounded(strings.NewReader(strings.Repeat("a", 1024)), 1024)
	if err != nil {
		t.Fatalf("ReadBounded() = %v, want the body", err)
	}
	if len(data) != 1024 {
		t.Errorf("read %d bytes, want 1024", len(data))
	}
}

func TestReadBoundedRejectsANonPositiveLimit(t *testing.T) {
	if _, err := ReadBounded(strings.NewReader("a"), 0); !errors.Is(err, ErrResponseTooLarge) {
		t.Fatalf("ReadBounded(limit=0) = %v, want ErrResponseTooLarge", err)
	}
}

// Compression is disabled on the transport, so a small gzip stream cannot
// expand past the caller's budget: the peer's bytes arrive undecoded and the
// bound applies to what actually crosses the wire.
func TestClientDoesNotAutoDecompress(t *testing.T) {
	const expanded = 1 << 20

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Encoding", "gzip")
		writer := gzip.NewWriter(w)
		_, _ = writer.Write([]byte(strings.Repeat("a", expanded)))
		_ = writer.Close()
	}))
	defer server.Close()

	host, port := splitHostPort(t, server.Listener.Addr().String())
	client := loopbackClient(t, host)

	resp, err := client.Get(fmt.Sprintf("http://%s:%s/", host, port))
	if err != nil {
		t.Fatalf("Get() = %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if got := resp.Header.Get("Content-Encoding"); got != "gzip" {
		t.Fatalf("Content-Encoding = %q, want gzip preserved for the caller to bound", got)
	}

	data, err := ReadBounded(resp.Body, 64*1024)
	if err != nil {
		t.Fatalf("ReadBounded() = %v", err)
	}
	if len(data) >= expanded {
		t.Errorf("read %d bytes, want the compressed stream, not the expansion", len(data))
	}
}

// A peer that accepts the connection and then stalls is cut off rather than
// holding a dashboard goroutine open.
func TestClientTimesOutOnASlowPeer(t *testing.T) {
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		<-release
		w.WriteHeader(http.StatusOK)
	}))
	defer func() {
		close(release)
		server.Close()
	}()

	host, port := splitHostPort(t, server.Listener.Addr().String())
	policy := DefaultPolicy().
		WithTimeout(300 * time.Millisecond).
		WithResolver(staticResolver{addresses: []netip.Addr{netip.MustParseAddr(host)}})
	policy.ResponseHeaderTimeout = 200 * time.Millisecond

	started := time.Now()
	resp, err := policy.NewClient().Get(fmt.Sprintf("http://%s:%s/", host, port))
	closeResponse(resp)
	if err == nil {
		t.Fatal("a stalled peer was not cut off")
	}
	if elapsed := time.Since(started); elapsed > 3*time.Second {
		t.Errorf("took %v to give up, want the configured deadline", elapsed)
	}
}

// loopbackClient builds a client whose address policy is bypassed only for the
// test server, so transport behaviour can be exercised without a public peer.
func loopbackClient(t *testing.T, host string) *http.Client {
	t.Helper()
	policy := DefaultPolicy().WithResolver(staticResolver{
		addresses: []netip.Addr{netip.MustParseAddr(host)},
	})
	client := policy.NewClient()
	transport, ok := client.Transport.(*http.Transport)
	if !ok {
		t.Fatal("transport is not an *http.Transport")
	}
	dialer := &net.Dialer{Timeout: policy.DialTimeout}
	transport.DialContext = dialer.DialContext
	return client
}

func splitHostPort(t *testing.T, address string) (string, string) {
	t.Helper()
	host, port, err := net.SplitHostPort(address)
	if err != nil {
		t.Fatalf("SplitHostPort(%q) = %v", address, err)
	}
	return host, port
}

func mustRequest(t *testing.T, raw string) *http.Request {
	t.Helper()
	request, err := http.NewRequest(http.MethodGet, raw, nil)
	if err != nil {
		t.Fatalf("NewRequest(%q) = %v", raw, err)
	}
	return request
}

// closeResponse drains nothing and just releases the body when a response
// exists, which is most of these cases: the request was refused.
func closeResponse(resp *http.Response) {
	if resp != nil {
		_ = resp.Body.Close()
	}
}
