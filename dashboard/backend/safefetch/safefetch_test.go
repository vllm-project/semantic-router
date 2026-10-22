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
	"sync"
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
	var firstHopCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/start" {
			firstHopCalls.Add(1)
			http.Redirect(w, r, "http://internal.invalid/admin", http.StatusFound)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	host, port := splitHostPort(t, upstream.Listener.Addr().String())
	client := DefaultPolicy().
		AllowingPrivate(netip.MustParsePrefix("127.0.0.0/8")).
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
	if got := firstHopCalls.Load(); got != 1 {
		t.Fatalf("first hop calls = %d, want the redirect fixture reached once", got)
	}
}

// hostRoutingResolver resolves the explicitly allowed loopback fixture and
// sends the redirect name to a different, disallowed private address.
type hostRoutingResolver struct {
	public   string
	publicAt netip.Addr
}

func (r hostRoutingResolver) LookupNetIP(_ context.Context, _ string, host string) ([]netip.Addr, error) {
	if host == r.public {
		return []netip.Addr{r.publicAt}, nil
	}
	return []netip.Addr{netip.MustParseAddr("10.0.0.1")}, nil
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
	var reached atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		reached.Store(true)
		<-release
		w.WriteHeader(http.StatusOK)
	}))
	defer func() {
		close(release)
		server.Close()
	}()

	host, port := splitHostPort(t, server.Listener.Addr().String())
	policy := DefaultPolicy().
		WithTimeout(300*time.Millisecond).
		WithResolver(staticResolver{addresses: []netip.Addr{netip.MustParseAddr(host)}}).
		AllowingPrivate(netip.MustParsePrefix("127.0.0.0/8"), netip.MustParsePrefix("::1/128"))
	policy.ResponseHeaderTimeout = 200 * time.Millisecond

	started := time.Now()
	resp, err := policy.NewClient().Get(fmt.Sprintf("http://%s:%s/", host, port))
	closeResponse(resp)
	if err == nil {
		t.Fatal("a stalled peer was not cut off")
	}
	if !reached.Load() {
		t.Fatal("request was refused before reaching the server, want the header timeout to be what cuts it off")
	}
	var netErr net.Error
	if !errors.As(err, &netErr) || !netErr.Timeout() {
		t.Errorf("err = %v, want a timeout error", err)
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
	wrapped, ok := client.Transport.(idleClosingTransport)
	if !ok {
		t.Fatal("transport is not an idleClosingTransport")
	}
	dialer := &net.Dialer{Timeout: policy.DialTimeout}
	wrapped.DialContext = dialer.DialContext
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

// Every resolved address is validated, so all of them are safe to dial. Pinning
// the first one drops the fallback the stock dialer gives you, and a host whose
// first record is unreachable then looks dead. This is the ordinary shape of a
// partial IPv6 outage: the AAAA answer is tried first and nothing is listening.
func TestClientFallsBackAcrossValidatedAddresses(t *testing.T) {
	// v4-only server, so the IPv6 loopback answer below refuses immediately.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("reached"))
	}))
	defer server.Close()

	host, port := splitHostPort(t, server.Listener.Addr().String())
	policy := DefaultPolicy().
		WithResolver(staticResolver{addresses: []netip.Addr{
			netip.MustParseAddr("::1"), // first answer, nothing listening
			netip.MustParseAddr(host),  // second answer, the server
		}}).
		AllowingPrivate(netip.MustParsePrefix("127.0.0.0/8"), netip.MustParsePrefix("::1/128"))

	resp, err := policy.NewClient().Get(fmt.Sprintf("http://dual-stack.invalid:%s/", port))
	if err != nil {
		t.Fatalf("Get() = %v, want a fallback to the second validated address", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := ReadBounded(resp.Body, 1024)
	if err != nil {
		t.Fatalf("ReadBounded() = %v", err)
	}
	if string(body) != "reached" {
		t.Errorf("body = %q, want %q", body, "reached")
	}
}

// A refusal on every validated address is still a failure, and the error names
// what was tried rather than being swallowed.
func TestClientFailsWhenEveryValidatedAddressIsUnreachable(t *testing.T) {
	// Bind then release, so the port is closed on both families.
	probe, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	_, port := splitHostPort(t, probe.Addr().String())
	_ = probe.Close()

	policy := DefaultPolicy().
		WithResolver(staticResolver{addresses: []netip.Addr{
			netip.MustParseAddr("::1"),
			netip.MustParseAddr("127.0.0.1"),
		}}).
		AllowingPrivate(netip.MustParsePrefix("127.0.0.0/8"), netip.MustParsePrefix("::1/128"))

	resp, dialErr := policy.NewClient().Get(fmt.Sprintf("http://dead.invalid:%s/", port))
	closeResponse(resp)
	if dialErr == nil {
		t.Fatal("expected an error when no validated address accepts a connection")
	}
}

// closeTrackingConn signals on closed when Close is called, so a test can
// tell whether the real network connection was actually torn down rather
// than left open in the transport's idle pool.
type closeTrackingConn struct {
	net.Conn
	closed chan struct{}
}

func (c *closeTrackingConn) Close() error {
	select {
	case <-c.closed:
	default:
		close(c.closed)
	}
	return c.Conn.Close()
}

// NewClient builds a fresh Transport for every call and nothing else will
// ever reuse it, so a connection left in its idle pool after a successful
// fetch only wastes a socket and a read goroutine until the default 90s idle
// timeout reaps it. The transport must close it as soon as the caller closes
// the response body instead (review on #3617).
func TestClientClosesConnectionAfterBodyClose(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("ok"))
	}))
	defer server.Close()

	host, port := splitHostPort(t, server.Listener.Addr().String())
	closed := make(chan struct{})
	realDial := (&net.Dialer{}).DialContext
	policy := DefaultPolicy().
		WithResolver(staticResolver{addresses: []netip.Addr{netip.MustParseAddr(host)}}).
		AllowingPrivate(netip.MustParsePrefix("127.0.0.0/8"))
	policy.dialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		conn, err := realDial(ctx, network, address)
		if err != nil {
			return nil, err
		}
		return &closeTrackingConn{Conn: conn, closed: closed}, nil
	}

	resp, err := policy.NewClient().Get(fmt.Sprintf("http://server.invalid:%s/", port))
	if err != nil {
		t.Fatalf("Get() = %v", err)
	}
	if _, err := ReadBounded(resp.Body, 1024); err != nil {
		t.Fatalf("ReadBounded() = %v", err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("Body.Close() = %v", err)
	}

	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("connection was still open after the response body was closed, want it swept immediately rather than left for the transport's idle timeout")
	}
}

// hostResolver answers per-hostname, standing in for DNS when a test's
// redirect chain needs different hosts to resolve to different addresses.
type hostResolver map[string][]netip.Addr

func (r hostResolver) LookupNetIP(_ context.Context, _ string, host string) ([]netip.Addr, error) {
	if addrs, ok := r[host]; ok {
		return addrs, nil
	}
	return nil, fmt.Errorf("no such host: %s", host)
}

// A redirect's first hop can already have pooled a connection on this
// transport by the time a later hop fails outright and Client.Do returns an
// error with no response for the caller to close. Nothing else will ever
// reuse this transport either, so that pooled connection must be closed
// right then, not left for the idle timeout (review on #3617).
func TestClientClosesPooledConnectionWhenALaterHopFails(t *testing.T) {
	// Bind then release, so the redirect target refuses the connection.
	probe, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	deadHost, deadPort := splitHostPort(t, probe.Addr().String())
	_ = probe.Close()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, fmt.Sprintf("http://dead.invalid:%s/", deadPort), http.StatusFound)
	}))
	defer server.Close()
	host, port := splitHostPort(t, server.Listener.Addr().String())

	closed := make(chan struct{})
	realDial := (&net.Dialer{}).DialContext
	policy := DefaultPolicy().
		WithResolver(hostResolver{
			"redirector.invalid": {netip.MustParseAddr(host)},
			"dead.invalid":       {netip.MustParseAddr(deadHost)},
		}).
		AllowingPrivate(netip.MustParsePrefix("127.0.0.0/8"))
	policy.dialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		conn, dialErr := realDial(ctx, network, address)
		if dialErr != nil {
			// Only the redirecting server's hop should ever succeed here.
			return nil, dialErr
		}
		return &closeTrackingConn{Conn: conn, closed: closed}, nil
	}

	resp, err := policy.NewClient().Get(fmt.Sprintf("http://redirector.invalid:%s/", port))
	closeResponse(resp)
	if err == nil {
		t.Fatal("expected an error when the redirect target refuses the connection")
	}

	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("the redirecting hop's connection was not closed after the later hop failed")
	}
}

// orderByFamily interleaves the two families so a broken IPv6 path cannot
// consume every attempt before the first IPv4 answer is reached.
func TestOrderByFamilyInterleaves(t *testing.T) {
	v6a, v6b := netip.MustParseAddr("2606:4700::1"), netip.MustParseAddr("2606:4700::2")
	v4a, v4b := netip.MustParseAddr("1.1.1.1"), netip.MustParseAddr("1.0.0.1")

	tests := []struct {
		name    string
		in      []netip.Addr
		network string
		want    []netip.Addr
	}{
		{"v6 first leads with v6", []netip.Addr{v6a, v6b, v4a, v4b}, "tcp", []netip.Addr{v6a, v4a, v6b, v4b}},
		{"v4 first leads with v4", []netip.Addr{v4a, v4b, v6a, v6b}, "tcp", []netip.Addr{v4a, v6a, v4b, v6b}},
		{"single family is untouched", []netip.Addr{v4a, v4b}, "tcp", []netip.Addr{v4a, v4b}},
		{"tcp4 drops v6 answers", []netip.Addr{v6a, v4a}, "tcp4", []netip.Addr{v4a}},
		{"tcp6 drops v4 answers", []netip.Addr{v4a, v6a}, "tcp6", []netip.Addr{v6a}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := orderByFamily(tt.in, tt.network)
			if len(got) != len(tt.want) {
				t.Fatalf("orderByFamily() = %v, want %v", got, tt.want)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Fatalf("orderByFamily() = %v, want %v", got, tt.want)
				}
			}
		})
	}
}

// fakeConn is a net.Conn a stub dialer can hand back without a real socket.
type fakeConn struct{ net.Conn }

func (fakeConn) Close() error { return nil }

// A black-holed first address must not consume the request deadline. This is
// the case a refused address does not cover: a refusal returns in microseconds,
// a black hole returns only when something times out. Walking the candidates
// one at a time means the stalled attempt owns the whole budget and the healthy
// family is never reached (review on #3617).
func TestClientRacesPastAStalledAddress(t *testing.T) {
	const stalled = "[2606:4700::1]:443"

	var dialed []string
	var mu sync.Mutex
	policy := DefaultPolicy().
		WithResolver(staticResolver{addresses: []netip.Addr{
			netip.MustParseAddr("2606:4700::1"), // first family, black holed
			netip.MustParseAddr("1.1.1.1"),      // second family, healthy
		}})
	policy.FallbackDelay = 10 * time.Millisecond
	policy.dialContext = func(ctx context.Context, _, address string) (net.Conn, error) {
		mu.Lock()
		dialed = append(dialed, address)
		mu.Unlock()
		if address == stalled {
			<-ctx.Done() // never connects, never refuses
			return nil, ctx.Err()
		}
		return fakeConn{}, nil
	}

	started := time.Now()
	conn, err := policy.dial(context.Background(), policy.Resolver, policy.dialContext, "tcp", "cloudflare.invalid:443")
	if err != nil {
		t.Fatalf("dial() = %v, want the healthy address to win", err)
	}
	_ = conn.Close()

	// The win must come from the stagger, not from the stalled attempt ending.
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Errorf("took %v, want the fallback to start after FallbackDelay", elapsed)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(dialed) != 2 || dialed[0] != stalled {
		t.Fatalf("dialed = %v, want the stalled address first then the fallback", dialed)
	}
}

// A failure should not wait out the stagger before the next candidate starts.
func TestRaceStartsNextImmediatelyOnFailure(t *testing.T) {
	policy := DefaultPolicy().
		WithResolver(staticResolver{addresses: []netip.Addr{
			netip.MustParseAddr("2606:4700::1"),
			netip.MustParseAddr("1.1.1.1"),
		}})
	policy.FallbackDelay = 30 * time.Second // would dominate if it were waited out
	policy.dialContext = func(_ context.Context, _, address string) (net.Conn, error) {
		if address == "[2606:4700::1]:443" {
			return nil, errors.New("connection refused")
		}
		return fakeConn{}, nil
	}

	started := time.Now()
	conn, err := policy.dial(context.Background(), policy.Resolver, policy.dialContext, "tcp", "cloudflare.invalid:443")
	if err != nil {
		t.Fatalf("dial() = %v", err)
	}
	_ = conn.Close()
	if elapsed := time.Since(started); elapsed > 5*time.Second {
		t.Errorf("took %v, want the next candidate to start as soon as the first failed", elapsed)
	}
}

// trackedConn signals on closed when Close is called, so a test can tell
// whether a connection that landed after the race was over was ever closed.
type trackedConn struct {
	fakeConn
	closed chan struct{}
}

func (c trackedConn) Close() error {
	close(c.closed)
	return c.fakeConn.Close()
}

// Cancelling the caller's context while an attempt is still in flight must
// not leak that attempt's connection if it goes on to succeed anyway: the
// dial loop has already returned by then, so nothing else will ever close it
// (review on #3617).
func TestRaceClosesLateConnAfterCancellation(t *testing.T) {
	policy := DefaultPolicy().
		WithResolver(staticResolver{addresses: []netip.Addr{
			netip.MustParseAddr("1.1.1.1"),
		}})

	dialStarted := make(chan struct{})
	releaseDial := make(chan struct{})
	closed := make(chan struct{})
	policy.dialContext = func(context.Context, string, string) (net.Conn, error) {
		close(dialStarted)
		<-releaseDial
		// Ignores ctx to stand in for the race a real dialer can lose: the
		// handshake finishes right as the caller gives up.
		return trackedConn{fakeConn{}, closed}, nil
	}

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	var dialErr error
	go func() {
		_, dialErr = policy.dial(ctx, policy.Resolver, policy.dialContext, "tcp", "cloudflare.invalid:443")
		close(done)
	}()

	<-dialStarted
	cancel()
	<-done
	if !errors.Is(dialErr, context.Canceled) {
		t.Fatalf("dial() = %v, want context.Canceled", dialErr)
	}

	close(releaseDial)
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("connection that completed after cancellation was never closed, leaking it")
	}
}

// Every candidate failing is still a failure, and the error names them.
func TestRaceReportsEveryFailure(t *testing.T) {
	policy := DefaultPolicy().
		WithResolver(staticResolver{addresses: []netip.Addr{
			netip.MustParseAddr("2606:4700::1"),
			netip.MustParseAddr("1.1.1.1"),
		}})
	policy.FallbackDelay = time.Millisecond
	policy.dialContext = func(_ context.Context, _, address string) (net.Conn, error) {
		return nil, fmt.Errorf("no route to %s", address)
	}

	_, err := policy.dial(context.Background(), policy.Resolver, policy.dialContext, "tcp", "cloudflare.invalid:443")
	if err == nil {
		t.Fatal("expected an error when every candidate fails")
	}
	for _, want := range []string{"2606:4700::1", "1.1.1.1"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error does not name %s: %v", want, err)
		}
	}
}
