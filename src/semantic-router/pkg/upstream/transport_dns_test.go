package upstream

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"golang.org/x/net/dns/dnsmessage"
)

func TestDoResolvesSingleBackendHostname(t *testing.T) {
	ipv6, err := net.Listen("tcp6", "[::1]:0")
	if err != nil {
		t.Skipf("IPv6 loopback unavailable: %v", err)
	}
	t.Cleanup(func() { _ = ipv6.Close() })
	port := ipv6.Addr().(*net.TCPAddr).Port
	ipv4, err := net.Listen("tcp4", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		t.Fatal(err)
	}
	for body, listener := range map[string]net.Listener{"ipv4": ipv4, "ipv6": ipv6} {
		server := &httptest.Server{
			Listener: listener,
			Config: &http.Server{
				ReadHeaderTimeout: time.Second,
				Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					_, _ = io.WriteString(w, body)
				}),
			},
		}
		server.Start()
		t.Cleanup(server.Close)
	}

	dialer := backendDNSDialer(t)
	for _, tc := range []struct {
		name     string
		host     string
		ipv4Only bool
		wantCode int
		wantBody string
	}{
		{name: "AAAA only", host: "ipv6-backend.test.", wantCode: 200, wantBody: "ipv6"},
		{name: "A only", host: "ipv4-backend.test.", wantCode: 200, wantBody: "ipv4"},
		{name: "A and AAAA", host: "dual-backend.test.", wantCode: 200, wantBody: "ipv4"},
		{name: "IPv6 literal", host: "[::1]", wantCode: 200, wantBody: "ipv6"},
		{name: "explicit IPv4 only", host: "ipv6-backend.test.", ipv4Only: true, wantCode: 503, wantBody: "no healthy upstream"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			topology := compileYAML(t, fmt.Sprintf(`version: v0.5
providers:
  models:
    - name: model
      backend_refs:
        - {provider: vllm, endpoint: '%s:%d'}
recipes:
  - name: default
`, tc.host, port))
			if tc.ipv4Only {
				topology.Clusters[0].Endpoints[0].IPv4Only = true
			}
			set, err := New(topology, Options{Dial: dialer.DialContext})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = set.Close(context.Background()) })
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			resp, err := set.Do(ctx, post("model"))
			if err != nil {
				t.Fatal(err)
			}
			body, err := readAll(t, resp)
			if err != nil || resp.StatusCode != tc.wantCode || body != tc.wantBody {
				t.Fatalf("response = %d %q (%v), want %d %q", resp.StatusCode, body, err, tc.wantCode, tc.wantBody)
			}
		})
	}
}

// backendDNSDialer makes the real Go resolver query a local DNS server.
func backendDNSDialer(t *testing.T) *net.Dialer {
	t.Helper()
	server, err := net.ListenPacket("udp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	t.Cleanup(func() {
		_ = server.Close()
		<-done
	})
	go func() {
		defer close(done)
		var buf [512]byte
		for {
			n, remote, err := server.ReadFrom(buf[:])
			if err != nil {
				return
			}
			var query dnsmessage.Message
			if unpackErr := query.Unpack(buf[:n]); unpackErr != nil || len(query.Questions) != 1 {
				continue
			}
			question := query.Questions[0]
			name := question.Name.String()
			response := dnsmessage.Message{
				Header:    dnsmessage.Header{ID: query.ID, Response: true, Authoritative: true},
				Questions: query.Questions,
			}
			var answer dnsmessage.ResourceBody
			switch {
			case question.Type == dnsmessage.TypeA && (name == "ipv4-backend.test." || name == "dual-backend.test."):
				answer = &dnsmessage.AResource{A: [4]byte{127, 0, 0, 1}}
			case question.Type == dnsmessage.TypeAAAA && (name == "ipv6-backend.test." || name == "dual-backend.test."):
				answer = &dnsmessage.AAAAResource{AAAA: [16]byte{15: 1}}
			}
			if answer != nil {
				response.Answers = []dnsmessage.Resource{{
					Header: dnsmessage.ResourceHeader{Name: question.Name, Class: question.Class, TTL: 60},
					Body:   answer,
				}}
			}
			packet, err := response.Pack()
			if err != nil {
				t.Errorf("pack DNS response: %v", err)
				return
			}
			_, _ = server.WriteTo(packet, remote)
		}
	}()
	return &net.Dialer{Resolver: &net.Resolver{
		PreferGo: true,
		Dial: func(ctx context.Context, network, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, server.LocalAddr().String())
		},
	}}
}
