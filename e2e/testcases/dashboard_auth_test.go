package testcases

import (
	"net/http"
	"testing"
)

func TestSetDashboardAuthUsesBrowserSessionCookie(t *testing.T) {
	request, err := http.NewRequest(http.MethodGet, "https://dashboard.example.test/api/status", nil)
	if err != nil {
		t.Fatal(err)
	}

	setDashboardAuth(request, "dashboard-session")

	if authorization := request.Header.Get("Authorization"); authorization != "" {
		t.Fatalf("Authorization = %q, want empty", authorization)
	}
	cookie, err := request.Cookie("vsr_session")
	if err != nil {
		t.Fatalf("Dashboard session cookie: %v", err)
	}
	if cookie.Value != "dashboard-session" {
		t.Fatalf("Dashboard session cookie = %q", cookie.Value)
	}
}
