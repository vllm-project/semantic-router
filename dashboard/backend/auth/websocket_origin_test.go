package auth

import (
	"crypto/tls"
	"net/http/httptest"
	"testing"
)

func TestValidWebSocketOriginAcceptsStrictSameOrigin(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name           string
		host           string
		origin         string
		tls            bool
		forwardedProto []string
	}{
		{name: "http", host: "play.example.com", origin: "http://play.example.com"},
		{name: "http default port", host: "play.example.com:80", origin: "http://play.example.com"},
		{name: "https TLS", host: "play.example.com", origin: "https://play.example.com", tls: true},
		{name: "https default port", host: "play.example.com:443", origin: "https://play.example.com", tls: true},
		{name: "https reverse proxy", host: "play.example.com", origin: "https://play.example.com", forwardedProto: []string{" HTTPS "}},
		{name: "IPv6 default port", host: "[::1]", origin: "http://[::1]"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			request := httptest.NewRequest("GET", "http://"+tc.host+"/api/openclaw/rooms/room/ws", nil)
			request.Host = tc.host
			request.Header.Set("Origin", tc.origin)
			if tc.tls {
				request.TLS = &tls.ConnectionState{}
			}
			for _, value := range tc.forwardedProto {
				request.Header.Add(forwardedProtoHeader, value)
			}

			if !ValidWebSocketOrigin(request) {
				t.Fatalf("same-origin request was rejected: host=%q origin=%q", tc.host, tc.origin)
			}
		})
	}
}

func TestValidWebSocketOriginRejectsUntrustedOrigins(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name           string
		host           string
		originValues   []string
		tls            bool
		forwardedProto []string
	}{
		{name: "missing origin", host: "play.example.com"},
		{name: "null origin", host: "play.example.com", originValues: []string{"null"}},
		{name: "sibling subdomain", host: "play.example.com", originValues: []string{"https://evil.example.com"}, forwardedProto: []string{"https"}},
		{name: "host mismatch", host: "play.example.com", originValues: []string{"http://other.example.net"}},
		{name: "scheme mismatch", host: "play.example.com", originValues: []string{"https://play.example.com"}},
		{name: "userinfo", host: "play.example.com", originValues: []string{"http://user@play.example.com"}},
		{name: "path", host: "play.example.com", originValues: []string{"http://play.example.com/path"}},
		{name: "query", host: "play.example.com", originValues: []string{"http://play.example.com?x=1"}},
		{name: "fragment", host: "play.example.com", originValues: []string{"http://play.example.com#fragment"}},
		{name: "malformed origin", host: "play.example.com", originValues: []string{"://bad"}},
		{name: "multiple origins", host: "play.example.com", originValues: []string{"http://play.example.com", "http://play.example.com"}},
		{name: "forwarded list", host: "play.example.com", originValues: []string{"https://play.example.com"}, forwardedProto: []string{"https,http"}},
		{name: "repeated forwarded proto", host: "play.example.com", originValues: []string{"https://play.example.com"}, forwardedProto: []string{"https", "https"}},
		{name: "invalid forwarded proto", host: "play.example.com", originValues: []string{"https://play.example.com"}, forwardedProto: []string{"wss"}},
		{name: "forwarded TLS downgrade", host: "play.example.com", originValues: []string{"http://play.example.com"}, tls: true, forwardedProto: []string{"http"}},
		{name: "untrusted forwarded host", host: "internal:8700", originValues: []string{"https://play.example.com"}, forwardedProto: []string{"https"}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			request := httptest.NewRequest("GET", "http://"+tc.host+"/api/openclaw/rooms/room/ws", nil)
			request.Host = tc.host
			for _, value := range tc.originValues {
				request.Header.Add("Origin", value)
			}
			if tc.tls {
				request.TLS = &tls.ConnectionState{}
			}
			for _, value := range tc.forwardedProto {
				request.Header.Add(forwardedProtoHeader, value)
			}
			request.Header.Set("X-Forwarded-Host", "play.example.com")

			if ValidWebSocketOrigin(request) {
				t.Fatalf("untrusted origin was accepted: host=%q origins=%q", tc.host, tc.originValues)
			}
		})
	}
}

func TestIsWebSocketUpgradeRequest(t *testing.T) {
	t.Parallel()

	request := httptest.NewRequest("GET", "http://play.example.com/socket", nil)
	request.Header.Set("Connection", "keep-alive, Upgrade")
	request.Header.Set("Upgrade", "WebSocket")
	if !IsWebSocketUpgradeRequest(request) {
		t.Fatal("valid WebSocket upgrade was not recognized")
	}

	request.Header.Del("Connection")
	if IsWebSocketUpgradeRequest(request) {
		t.Fatal("request without Connection: upgrade was recognized")
	}
}
