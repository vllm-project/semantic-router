package config

import "testing"

func TestLegacyEndpointURLKeepsAFullAddress(t *testing.T) {
	for _, tc := range []struct {
		address  string
		port     int
		protocol string
		want     string
	}{
		{"http://embedder.example.com:9000/v1", 8000, "", "http://embedder.example.com:9000/v1"},
		{"https://embedder.example.com/v1/", 443, "http", "https://embedder.example.com/v1"},
		{"embedder.example.com", 8000, "", "http://embedder.example.com:8000"},
		{"embedder.example.com", 443, "HTTPS", "https://embedder.example.com:443"},
		{"embedder.example.com", 0, "", "http://embedder.example.com"},
		{"::1", 8000, "", "http://[::1]:8000"},
	} {
		if got := LegacyEndpointURL(tc.address, tc.port, tc.protocol); got != tc.want {
			t.Errorf("LegacyEndpointURL(%q, %d, %q) = %q, want %q", tc.address, tc.port, tc.protocol, got, tc.want)
		}
	}
}
