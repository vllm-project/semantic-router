package handlers

import (
	"context"
	"errors"
	"net"
	"net/netip"
	"sync/atomic"
	"testing"
	"time"
)

func TestPublicOutboundClientRejectsRebindingBetweenValidationAndRequest(t *testing.T) {
	t.Parallel()

	var lookups atomic.Int32
	var dialCalls atomic.Int32
	client := newPublicOutboundHTTPClientWithOptions(time.Second, outboundHTTPClientOptions{
		resolver: testOutboundResolver(func(context.Context, string, string) ([]netip.Addr, error) {
			if lookups.Add(1) == 1 {
				return []netip.Addr{netip.MustParseAddr("1.1.1.1")}, nil
			}
			return []netip.Addr{netip.MustParseAddr("127.0.0.1")}, nil
		}),
		dialContext: func(context.Context, string, string) (net.Conn, error) {
			dialCalls.Add(1)
			return nil, errors.New("unexpected dial")
		},
	})

	const targetURL = "http://rebind.example/data"
	if err := client.ValidateURL(context.Background(), targetURL); err != nil {
		t.Fatalf("initial ValidateURL() error = %v", err)
	}
	resp, err := client.Do(newTestOutboundRequest(t, targetURL))
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if !errors.Is(err, errOutboundDestinationBlocked) {
		t.Fatalf("rebound request error = %v, want destination blocked", err)
	}
	if got := dialCalls.Load(); got != 0 {
		t.Fatalf("dial calls = %d, want rebound destination rejected before dial", got)
	}
}

func TestPublicOutboundClientRejectsHostHeaderOverride(t *testing.T) {
	t.Parallel()

	var dialCalls atomic.Int32
	client := newPublicOutboundHTTPClientWithOptions(time.Second, outboundHTTPClientOptions{
		resolver: testOutboundResolver(func(context.Context, string, string) ([]netip.Addr, error) {
			return []netip.Addr{netip.MustParseAddr("1.1.1.1")}, nil
		}),
		dialContext: func(context.Context, string, string) (net.Conn, error) {
			dialCalls.Add(1)
			return nil, errors.New("unexpected dial")
		},
	})

	req := newTestOutboundRequest(t, "http://public.example/data")
	req.Host = "internal.example"
	resp, err := client.Do(req)
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if !errors.Is(err, errOutboundURLInvalid) {
		t.Fatalf("Host override error = %v, want invalid URL", err)
	}
	if got := dialCalls.Load(); got != 0 {
		t.Fatalf("dial calls = %d, want Host override rejected before dial", got)
	}
}
