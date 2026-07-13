package handlers

import (
	"context"
	"crypto/tls"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type testOutboundResolver func(context.Context, string, string) ([]netip.Addr, error)

func (r testOutboundResolver) LookupNetIP(
	ctx context.Context,
	network string,
	host string,
) ([]netip.Addr, error) {
	return r(ctx, network, host)
}

type controlledTestOutboundClient struct {
	client *http.Client
	do     func(*http.Request) (*http.Response, error)
}

func newControlledTestOutboundClient(client *http.Client) *controlledTestOutboundClient {
	return &controlledTestOutboundClient{client: client}
}

func (c *controlledTestOutboundClient) ValidateURL(_ context.Context, rawURL string) error {
	_, err := parseOutboundHTTPURL(rawURL)
	return err
}

func (c *controlledTestOutboundClient) Do(req *http.Request) (*http.Response, error) {
	if c.do != nil {
		return c.do(req)
	}
	return c.client.Do(req)
}

func newTestOutboundRequest(t *testing.T, rawURL string) *http.Request {
	t.Helper()
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, rawURL, nil)
	if err != nil {
		t.Fatalf("create request: %v", err)
	}
	return req
}

func TestIsPublicOutboundIP(t *testing.T) {
	t.Parallel()

	tests := []struct {
		address string
		want    bool
	}{
		{address: "1.1.1.1", want: true},
		{address: "2606:4700:4700::1111", want: true},
		{address: "::ffff:8.8.8.8", want: true},
		{address: "0.0.0.0", want: false},
		{address: "10.0.0.1", want: false},
		{address: "100.64.0.1", want: false},
		{address: "127.0.0.1", want: false},
		{address: "169.254.169.254", want: false},
		{address: "172.16.0.1", want: false},
		{address: "192.0.2.1", want: false},
		{address: "192.168.1.1", want: false},
		{address: "198.18.0.1", want: false},
		{address: "203.0.113.1", want: false},
		{address: "224.0.0.1", want: false},
		{address: "240.0.0.1", want: false},
		{address: "::", want: false},
		{address: "::1", want: false},
		{address: "::ffff:127.0.0.1", want: false},
		{address: "::ffff:192.168.1.1", want: false},
		{address: "64:ff9b::808:808", want: false},
		{address: "64:ff9b::7f00:1", want: false},
		{address: "64:ff9b:1::c000:201", want: false},
		{address: "100::1", want: false},
		{address: "2001:db8::1", want: false},
		{address: "2002:7f00:1::", want: false},
		{address: "4000::1", want: false},
		{address: "fc00::1", want: false},
		{address: "fe80::1", want: false},
		{address: "fe80::1%lo0", want: false},
		{address: "ff02::1", want: false},
	}

	for _, tt := range tests {
		t.Run(tt.address, func(t *testing.T) {
			t.Parallel()
			address := netip.MustParseAddr(tt.address)
			if got := isPublicOutboundIP(address); got != tt.want {
				t.Fatalf("isPublicOutboundIP(%s) = %v, want %v", address, got, tt.want)
			}
		})
	}
}

func TestPublicOutboundClientRejectsInvalidAndNonPublicURLs(t *testing.T) {
	t.Parallel()

	client := newPublicOutboundHTTPClientWithOptions(time.Second, outboundHTTPClientOptions{
		resolver: testOutboundResolver(func(context.Context, string, string) ([]netip.Addr, error) {
			return []netip.Addr{netip.MustParseAddr("1.1.1.1")}, nil
		}),
	})

	tests := []struct {
		name    string
		url     string
		wantErr error
	}{
		{name: "userinfo", url: "https://user:password@example.com/path", wantErr: errOutboundURLInvalid},
		{name: "non http", url: "file:///etc/passwd", wantErr: errOutboundURLInvalid},
		{name: "missing host", url: "https:///path", wantErr: errOutboundURLInvalid},
		{name: "localhost", url: "http://localhost/admin", wantErr: errOutboundDestinationBlocked},
		{name: "localhost subdomain", url: "http://api.localhost/admin", wantErr: errOutboundDestinationBlocked},
		{name: "metadata address", url: "http://169.254.169.254/latest", wantErr: errOutboundDestinationBlocked},
		{name: "private ipv4", url: "http://10.0.0.1/admin", wantErr: errOutboundDestinationBlocked},
		{name: "loopback ipv6", url: "http://[::1]/admin", wantErr: errOutboundDestinationBlocked},
		{name: "integer ipv4", url: "http://2130706433/admin", wantErr: errOutboundDestinationBlocked},
		{name: "hex ipv4", url: "http://0x7f000001/admin", wantErr: errOutboundDestinationBlocked},
		{name: "short ipv4", url: "http://127.1/admin", wantErr: errOutboundDestinationBlocked},
		{name: "octal ipv4", url: "http://0177.0.0.1/admin", wantErr: errOutboundDestinationBlocked},
		{name: "public literal", url: "https://1.1.1.1/path"},
		{name: "public hostname", url: "https://example.com/path"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := client.ValidateURL(context.Background(), tt.url)
			if tt.wantErr == nil {
				if err != nil {
					t.Fatalf("ValidateURL(%q) error = %v", tt.url, err)
				}
				return
			}
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("ValidateURL(%q) error = %v, want %v", tt.url, err, tt.wantErr)
			}
		})
	}
}

func TestPublicOutboundClientRejectsMixedDNSAnswersBeforeDial(t *testing.T) {
	t.Parallel()

	var dialCalls atomic.Int32
	client := newPublicOutboundHTTPClientWithOptions(time.Second, outboundHTTPClientOptions{
		resolver: testOutboundResolver(func(context.Context, string, string) ([]netip.Addr, error) {
			return []netip.Addr{
				netip.MustParseAddr("1.1.1.1"),
				netip.MustParseAddr("127.0.0.1"),
			}, nil
		}),
		dialContext: func(context.Context, string, string) (net.Conn, error) {
			dialCalls.Add(1)
			return nil, errors.New("unexpected dial")
		},
	})

	req := newTestOutboundRequest(t, "http://mixed.example/path")
	resp, err := client.Do(req)
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if !errors.Is(err, errOutboundDestinationBlocked) {
		t.Fatalf("Do() error = %v, want destination blocked", err)
	}
	if got := dialCalls.Load(); got != 0 {
		t.Fatalf("dial called %d times for a mixed DNS answer", got)
	}
}

func TestPublicOutboundClientDialsOnlyTheValidatedIP(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "ok")
	}))
	defer server.Close()

	var lookups atomic.Int32
	var dialedAddress string
	client := newPublicOutboundHTTPClientWithOptions(2*time.Second, outboundHTTPClientOptions{
		resolver: testOutboundResolver(func(_ context.Context, _ string, host string) ([]netip.Addr, error) {
			if host != "public.example" {
				t.Fatalf("resolved unexpected host %q", host)
			}
			lookups.Add(1)
			return []netip.Addr{netip.MustParseAddr("1.1.1.1")}, nil
		}),
		dialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			dialedAddress = address
			return (&net.Dialer{}).DialContext(ctx, network, server.Listener.Addr().String())
		},
	})

	req := newTestOutboundRequest(t, "http://public.example/resource")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	defer resp.Body.Close()
	if dialedAddress != "1.1.1.1:80" {
		t.Fatalf("dialed address = %q, want validated IP", dialedAddress)
	}
	if got := lookups.Load(); got != 1 {
		t.Fatalf("DNS lookups = %d, want exactly one pinned lookup", got)
	}
}

func TestPublicOutboundClientRejectsRedirectToPrivateDestination(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Redirect(w, &http.Request{}, "https://private.example/admin?token=redirect-secret", http.StatusFound)
	}))
	defer server.Close()

	var dialCalls atomic.Int32
	client := newPublicOutboundHTTPClientWithOptions(2*time.Second, outboundHTTPClientOptions{
		resolver: testOutboundResolver(func(_ context.Context, _ string, host string) ([]netip.Addr, error) {
			switch host {
			case "public.example":
				return []netip.Addr{netip.MustParseAddr("1.1.1.1")}, nil
			case "private.example":
				return []netip.Addr{netip.MustParseAddr("127.0.0.1")}, nil
			default:
				return nil, errors.New("unexpected host")
			}
		}),
		dialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
			dialCalls.Add(1)
			return (&net.Dialer{}).DialContext(ctx, network, server.Listener.Addr().String())
		},
	})

	req := newTestOutboundRequest(t, "http://public.example/start?token=request-secret")
	resp, err := client.Do(req)
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if !errors.Is(err, errOutboundDestinationBlocked) {
		t.Fatalf("Do() error = %v, want destination blocked", err)
	}
	if got := dialCalls.Load(); got != 1 {
		t.Fatalf("dial calls = %d, want redirect rejected before a second dial", got)
	}
	if strings.Contains(err.Error(), "secret") || strings.Contains(err.Error(), "private.example") {
		t.Fatalf("sanitized redirect error leaked target data: %v", err)
	}
}

func TestPublicOutboundClientAllowsAtMostFiveRedirects(t *testing.T) {
	t.Parallel()

	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		step, _ := strconv.Atoi(strings.TrimPrefix(r.URL.Path, "/"))
		http.Redirect(w, r, "/"+strconv.Itoa(step+1), http.StatusFound)
	}))
	defer server.Close()

	client := newPublicOutboundHTTPClientWithOptions(2*time.Second, outboundHTTPClientOptions{
		resolver: testOutboundResolver(func(context.Context, string, string) ([]netip.Addr, error) {
			return []netip.Addr{netip.MustParseAddr("1.1.1.1")}, nil
		}),
		dialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, server.Listener.Addr().String())
		},
	})

	req := newTestOutboundRequest(t, "http://public.example/0")
	resp, err := client.Do(req)
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if !errors.Is(err, errOutboundTooManyRedirects) {
		t.Fatalf("Do() error = %v, want too many redirects", err)
	}
	if got := requests.Load(); got != outboundMaxRedirects+1 {
		t.Fatalf("server requests = %d, want original plus %d redirects", got, outboundMaxRedirects)
	}
}

func TestPublicOutboundClientIgnoresAmbientProxy(t *testing.T) {
	t.Setenv("HTTP_PROXY", "http://proxy.invalid:3128")
	t.Setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
	t.Setenv("ALL_PROXY", "http://proxy.invalid:3128")
	t.Setenv("NO_PROXY", "")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "direct")
	}))
	defer server.Close()

	var dialedAddress string
	client := newPublicOutboundHTTPClientWithOptions(2*time.Second, outboundHTTPClientOptions{
		resolver: testOutboundResolver(func(context.Context, string, string) ([]netip.Addr, error) {
			return []netip.Addr{netip.MustParseAddr("1.1.1.1")}, nil
		}),
		dialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			dialedAddress = address
			return (&net.Dialer{}).DialContext(ctx, network, server.Listener.Addr().String())
		},
	})

	req := newTestOutboundRequest(t, "http://public.example/path")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	defer resp.Body.Close()
	if dialedAddress != "1.1.1.1:80" {
		t.Fatalf("dialed %q; ambient proxy was not disabled", dialedAddress)
	}

	roundTripper := client.client.Transport.(*publicOutboundRoundTripper)
	if roundTripper.transport.Proxy != nil {
		t.Fatal("public outbound transport has a proxy callback")
	}
}

func TestPublicOutboundClientRequiresTLS12OrNewer(t *testing.T) {
	t.Parallel()

	client := newPublicOutboundHTTPClient(time.Second)
	roundTripper := client.client.Transport.(*publicOutboundRoundTripper)
	if got := roundTripper.transport.TLSClientConfig.MinVersion; got < tls.VersionTLS12 {
		t.Fatalf("TLS minimum = %x, want TLS 1.2 or newer", got)
	}
	if got := roundTripper.transport.MaxResponseHeaderBytes; got != outboundMaxResponseHeaderBytes {
		t.Fatalf("response header limit = %d, want %d", got, outboundMaxResponseHeaderBytes)
	}
	if roundTripper.transport.Proxy != nil {
		t.Fatal("public outbound transport inherited an ambient proxy callback")
	}

	defaulted := newPublicOutboundHTTPClient(0)
	if got := defaulted.client.Timeout; got != outboundDefaultTimeout {
		t.Fatalf("zero timeout defaulted to %v, want %v", got, outboundDefaultTimeout)
	}
}

func TestReadBoundedOutboundBodyRejectsMaxPlusOne(t *testing.T) {
	t.Parallel()

	const maxBytes = int64(16)
	exact, err := readBoundedOutboundBody(strings.NewReader(strings.Repeat("a", int(maxBytes))), maxBytes)
	if err != nil || len(exact) != int(maxBytes) {
		t.Fatalf("exact-limit read = %d bytes, err=%v", len(exact), err)
	}

	_, err = readBoundedOutboundBody(strings.NewReader(strings.Repeat("a", int(maxBytes+1))), maxBytes)
	if !errors.Is(err, errOutboundResponseTooLarge) {
		t.Fatalf("max+1 read error = %v, want response too large", err)
	}
}
