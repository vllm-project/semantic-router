package gateway

import (
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"

	"golang.org/x/net/http2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

// heldUpstream signals each call on arrived and answers it once release
// closes.
type heldUpstream struct {
	arrived chan<- struct{}
	release <-chan struct{}
}

func (u heldUpstream) Execute(ctx context.Context, _ *routing.Call, _ string) (*upstream.Result, error) {
	u.arrived <- struct{}{}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-u.release:
		return &upstream.Result{Response: okUpstream(`{"answer":"whole"}`, http.Header{"Content-Type": {"application/json"}})()}, nil
	}
}

// drainTestServer serves a Handler whose upstream holds every call until
// release closes, over protocol, and returns the handler, its server, the
// chat completions URL and a client that speaks protocol.
func drainTestServer(t *testing.T, protocol string, arrived chan<- struct{}, release <-chan struct{}) (*Handler, *http.Server, string, *http.Client) {
	t.Helper()
	handler, err := NewHandler(Options{
		Serving: pinFunc(func() (Serving, func(), error) {
			held := heldUpstream{arrived: arrived, release: release}
			return Serving{Engine: routing.NewEngine(&fakeProcessor{}, routing.DefaultOptions), Upstream: held}, func() {}, nil
		}),
		Listener: "http-8899",
	})
	if err != nil {
		t.Fatal(err)
	}
	opts, scheme := ServerOptions{}, "http"
	var client *http.Client
	switch protocol {
	case "HTTP/1.1":
		client = &http.Client{Transport: &http.Transport{}}
	case "h2c":
		client = &http.Client{Transport: &http2.Transport{AllowHTTP: true, DialTLSContext: func(ctx context.Context, network, addr string, _ *tls.Config) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, addr)
		}}}
	case "HTTP/2 over TLS":
		certFile, keyFile, pool := writeCertificate(t)
		if opts.TLS, err = LoadTLS(certFile, keyFile, nil); err != nil {
			t.Fatal(err)
		}
		scheme = "https"
		client = &http.Client{Transport: &http.Transport{TLSClientConfig: &tls.Config{RootCAs: pool}, ForceAttemptHTTP2: true}}
	default:
		t.Fatalf("unknown protocol %q", protocol)
	}
	ln, err := Listen("127.0.0.1:0", opts)
	if err != nil {
		t.Fatal(err)
	}
	server := NewServer(handler, opts)
	go func() { _ = Serve(server, ln) }()
	t.Cleanup(func() { _ = server.Close() })
	return handler, server, scheme + "://" + ln.Addr().String() + "/v1/chat/completions", client
}

// postForDrain sends a chat completion and returns the status and body the
// client got, or the error.
func postForDrain(client *http.Client, url string) string {
	resp, err := client.Post(url, "application/json", strings.NewReader(`{"model":"m"}`))
	if err != nil {
		return "error: " + err.Error()
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "read error: " + err.Error()
	}
	return fmt.Sprintf("%d %s", resp.StatusCode, body)
}

func waitForDrainSignal(t *testing.T, signal <-chan struct{}, failure string) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(10 * time.Second):
		t.Fatal(failure)
	}
}

// A drain waits for a request in flight on every protocol a listener serves,
// h2c too, whose connection the server's Shutdown neither waits for nor
// closes, and ends as soon as the request does.
func TestDrainWaitsForARequestInFlightOnEveryProtocol(t *testing.T) {
	for _, protocol := range []string{"HTTP/1.1", "h2c", "HTTP/2 over TLS"} {
		t.Run(protocol, func(t *testing.T) {
			arrived, release := make(chan struct{}, 1), make(chan struct{})
			handler, server, url, client := drainTestServer(t, protocol, arrived, release)
			answered := make(chan string, 1)
			go func() { answered <- postForDrain(client, url) }()
			waitForDrainSignal(t, arrived, "the request never reached the upstream")

			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			drained := make(chan int, 1)
			go func() {
				_ = server.Shutdown(ctx)
				drained <- handler.Drain(ctx)
			}()
			select {
			case left := <-drained:
				t.Fatalf("the drain ended with the request in flight, %d left", left)
			case <-time.After(300 * time.Millisecond):
			}
			finished := time.Now()
			close(release)
			select {
			case left := <-drained:
				if left != 0 {
					t.Fatalf("the drain ended with %d requests in flight, want 0", left)
				}
				if waited := time.Since(finished); waited > 2*time.Second {
					t.Fatalf("the drain ended %v after the request finished", waited)
				}
			case <-time.After(10 * time.Second):
				t.Fatal("the drain did not end when the request finished")
			}
			if got := <-answered; got != `200 {"answer":"whole"}` {
				t.Fatalf("the client got %q, want the whole answer", got)
			}
		})
	}
}

// A drain ends at its deadline with requests still in flight, and counts
// them.
func TestDrainEndsAtItsDeadlineAndCountsTheRequestsLeft(t *testing.T) {
	for _, protocol := range []string{"HTTP/1.1", "h2c"} {
		t.Run(protocol, func(t *testing.T) {
			arrived, release := make(chan struct{}, 2), make(chan struct{})
			defer close(release)
			handler, server, url, client := drainTestServer(t, protocol, arrived, release)
			for range 2 {
				go postForDrain(client, url)
			}
			waitForDrainSignal(t, arrived, "the first request never reached the upstream")
			waitForDrainSignal(t, arrived, "the second request never reached the upstream")
			if inFlight := handler.InFlight(); inFlight != 2 {
				t.Fatalf("InFlight() = %d, want 2", inFlight)
			}

			const deadline = 400 * time.Millisecond
			ctx, cancel := context.WithTimeout(context.Background(), deadline)
			defer cancel()
			begun := time.Now()
			_ = server.Shutdown(ctx)
			left := handler.Drain(ctx)
			if waited := time.Since(begun); waited < deadline || waited > deadline+3*time.Second {
				t.Fatalf("the drain ended after %v, want its deadline of %v", waited, deadline)
			}
			if left != 2 {
				t.Fatalf("the drain ended with %d requests in flight, want 2", left)
			}
		})
	}
}

// Shutdown sends a GOAWAY on an h2c connection, so that its client opens no
// new stream on a listener that drains, while its stream in flight finishes.
func TestShutdownSendsAGoAwayOnAnH2CConnection(t *testing.T) {
	arrived, release := make(chan struct{}, 1), make(chan struct{})
	handler, server, url, _ := drainTestServer(t, "h2c", arrived, release)
	conn, err := net.Dial("tcp", strings.TrimSuffix(strings.TrimPrefix(url, "http://"), "/v1/chat/completions"))
	if err != nil {
		t.Fatal(err)
	}
	cc, err := (&http2.Transport{AllowHTTP: true}).NewClientConn(conn)
	if err != nil {
		t.Fatal(err)
	}
	defer cc.Close()
	answered := make(chan string, 1)
	go func() {
		req, reqErr := http.NewRequest(http.MethodPost, url, strings.NewReader(`{"model":"m"}`))
		if reqErr != nil {
			answered <- reqErr.Error()
			return
		}
		resp, reqErr := cc.RoundTrip(req)
		if reqErr != nil {
			answered <- reqErr.Error()
			return
		}
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		answered <- fmt.Sprintf("%d %s", resp.StatusCode, body)
	}()
	waitForDrainSignal(t, arrived, "the request never reached the upstream")

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	drained := make(chan int, 1)
	go func() {
		_ = server.Shutdown(ctx)
		drained <- handler.Drain(ctx)
	}()
	for until := time.Now().Add(5 * time.Second); !cc.State().Closing; time.Sleep(10 * time.Millisecond) {
		if time.Now().After(until) {
			t.Fatal("the h2c connection got no GOAWAY when the server began to shut down")
		}
	}
	close(release)
	if got := <-answered; got != `200 {"answer":"whole"}` {
		t.Fatalf("the stream in flight got %q, want the whole answer", got)
	}
	if left := <-drained; left != 0 {
		t.Fatalf("the drain ended with %d requests in flight, want 0", left)
	}
}
