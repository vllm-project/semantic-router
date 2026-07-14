package handlers

import (
	"compress/gzip"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"strings"
	"testing"
	"time"
)

func TestPublicOutboundClientBoundsGzipExpandedBody(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Encoding", "gzip")
		compressed := gzip.NewWriter(w)
		_, _ = io.WriteString(compressed, strings.Repeat("x", 1024))
		_ = compressed.Close()
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

	resp, err := client.Do(newTestOutboundRequest(t, "http://public.example/data"))
	if err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	defer resp.Body.Close()
	if _, err := readBoundedOutboundBody(resp.Body, 64); !errors.Is(err, errOutboundResponseTooLarge) {
		t.Fatalf("expanded gzip body error = %v, want response too large", err)
	}
}

func TestPublicOutboundClientTimesOutSlowHeaders(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer server.Close()

	client := newPublicOutboundHTTPClientWithOptions(100*time.Millisecond, outboundHTTPClientOptions{
		resolver: testOutboundResolver(func(context.Context, string, string) ([]netip.Addr, error) {
			return []netip.Addr{netip.MustParseAddr("1.1.1.1")}, nil
		}),
		dialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, server.Listener.Addr().String())
		},
	})

	started := time.Now()
	resp, err := client.Do(newTestOutboundRequest(t, "http://public.example/slow-headers"))
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if !errors.Is(err, errOutboundRequestTimeout) {
		t.Fatalf("slow-header error = %v, want request timeout", err)
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("slow-header request exceeded its budget: %v", elapsed)
	}
}

func TestPublicOutboundClientTimesOutSlowBody(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		<-r.Context().Done()
	}))
	defer server.Close()

	client := newPublicOutboundHTTPClientWithOptions(100*time.Millisecond, outboundHTTPClientOptions{
		resolver: testOutboundResolver(func(context.Context, string, string) ([]netip.Addr, error) {
			return []netip.Addr{netip.MustParseAddr("1.1.1.1")}, nil
		}),
		dialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, server.Listener.Addr().String())
		},
	})

	started := time.Now()
	resp, err := client.Do(newTestOutboundRequest(t, "http://public.example/slow-body"))
	if err != nil {
		t.Fatalf("Do() error before body read = %v", err)
	}
	defer resp.Body.Close()
	if _, err := readBoundedOutboundBody(resp.Body, 64); err == nil {
		t.Fatal("slow body read unexpectedly succeeded")
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("slow-body request exceeded its budget: %v", elapsed)
	}
}

func TestOutboundHandlersRejectAmbiguousAndOversizedRequestBodies(t *testing.T) {
	t.Parallel()
	client := &controlledTestOutboundClient{do: func(*http.Request) (*http.Response, error) {
		return nil, errors.New("outbound request must not run for invalid input")
	}}

	handlers := []struct {
		name    string
		handler http.HandlerFunc
	}{
		{name: "fetch raw", handler: fetchRawHandlerWithClient(client)},
		{name: "open web", handler: openWebHandlerWithClient(client)},
		{name: "web search", handler: webSearchHandlerWithDependencies(newImmediateWebSearchOutbound(client), newRateLimiter())},
		{name: "weather", handler: weatherHandlerWithClient(client, "https://weather.example", "https://weather.example")},
	}

	for _, endpoint := range handlers {
		t.Run(endpoint.name, func(t *testing.T) {
			t.Parallel()
			for _, test := range []struct {
				name       string
				body       string
				wantStatus int
			}{
				{name: "unknown field", body: `{"unexpected":true}`, wantStatus: http.StatusBadRequest},
				{name: "trailing value", body: `{} {}`, wantStatus: http.StatusBadRequest},
				{
					name:       "oversized",
					body:       `{"padding":"` + strings.Repeat("x", outboundMaxRequestBodyBytes) + `"}`,
					wantStatus: http.StatusRequestEntityTooLarge,
				},
			} {
				t.Run(test.name, func(t *testing.T) {
					req := httptest.NewRequest(http.MethodPost, "/api/tools/test", strings.NewReader(test.body))
					resp := httptest.NewRecorder()
					endpoint.handler.ServeHTTP(resp, req)
					if resp.Code != test.wantStatus {
						t.Fatalf("status = %d, want %d: %s", resp.Code, test.wantStatus, resp.Body.String())
					}
				})
			}
		})
	}
}

func TestOutboundURLLengthIsBoundedBeforeResolution(t *testing.T) {
	t.Parallel()
	_, err := parseOutboundHTTPURL("https://example.com/" + strings.Repeat("x", outboundMaxURLBytes))
	if !errors.Is(err, errOutboundURLInvalid) {
		t.Fatalf("oversized URL error = %v, want invalid URL", err)
	}
}
