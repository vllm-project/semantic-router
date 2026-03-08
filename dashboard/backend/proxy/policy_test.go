package proxy

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestValidateDashboardOrigin_AllowsSameOrigin(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "http://dashboard.local/embedded/grafana/", nil)
	req.Host = "dashboard.local"
	req.Header.Set("Origin", "http://dashboard.local")

	if err := ValidateDashboardOrigin(req); err != nil {
		t.Fatalf("ValidateDashboardOrigin() error = %v", err)
	}
}

func TestValidateDashboardOrigin_RejectsCrossOrigin(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "http://dashboard.local/embedded/grafana/", nil)
	req.Host = "dashboard.local"
	req.Header.Set("Origin", "https://evil.example")

	if err := ValidateDashboardOrigin(req); err == nil {
		t.Fatal("expected cross-origin request to be rejected")
	}
}

func TestValidateDashboardOrigin_UsesRefererFallback(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "https://dashboard.local/api/router/v1/chat/completions", nil)
	req.Host = "dashboard.local"
	req.Header.Set("X-Forwarded-Proto", "https")
	req.Header.Set("Referer", "https://dashboard.local/playground")

	if err := ValidateDashboardOrigin(req); err != nil {
		t.Fatalf("ValidateDashboardOrigin() error = %v", err)
	}
}

func TestValidateDashboardOrigin_RejectsCrossSiteFetchWithoutOrigin(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "http://dashboard.local/embedded/openclaw/demo/", nil)
	req.Host = "dashboard.local"
	req.Header.Set("Sec-Fetch-Site", "cross-site")

	if err := ValidateDashboardOrigin(req); err == nil {
		t.Fatal("expected cross-site fetch request to be rejected")
	}
}
