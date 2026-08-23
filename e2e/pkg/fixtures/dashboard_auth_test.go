package fixtures

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestWaitForFirstRouterPublicationUsesBrowserCookieAndManagementMediaType(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/api/router/management/v1/me" {
			t.Fatalf("request path = %q", request.URL.Path)
		}
		if request.Header.Get("Authorization") != "" {
			http.Error(response, "Bearer Dashboard sessions are not accepted", http.StatusUnauthorized)
			return
		}
		cookie, err := request.Cookie(dashboardSessionCookie)
		if err != nil || cookie.Value != "dashboard-session" {
			http.Error(response, "Dashboard session cookie is required", http.StatusUnauthorized)
			return
		}
		if request.Header.Get("Accept") != dashboardManagementV1 {
			http.Error(response, "Management v1 media type is required", http.StatusNotAcceptable)
			return
		}
		response.Header().Set("Content-Type", "application/json")
		_, _ = response.Write([]byte(`{
  "namespaces": [{
    "namespace": {
      "namespaceId": "10000000-0000-4000-8000-000000000001",
      "desiredRevision": 7,
      "appliedRevision": 7
    }
  }]
}`))
	}))
	defer server.Close()

	endpoint := server.URL + "/api/router/management/v1/me"
	bearerResponse, err := DoGETRequestWithHeaders(t.Context(), server.Client(), endpoint, map[string]string{
		"Authorization": "Bearer dashboard-session",
	})
	if err != nil {
		t.Fatal(err)
	}
	if bearerResponse.StatusCode != http.StatusUnauthorized {
		t.Fatalf("Bearer-only status = %d, want %d", bearerResponse.StatusCode, http.StatusUnauthorized)
	}

	cookieOnlyResponse, err := DoGETRequestWithHeaders(t.Context(), server.Client(), endpoint, map[string]string{
		"Cookie": (&http.Cookie{Name: dashboardSessionCookie, Value: "dashboard-session"}).String(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if cookieOnlyResponse.StatusCode != http.StatusNotAcceptable {
		t.Fatalf("cookie-only status = %d, want %d", cookieOnlyResponse.StatusCode, http.StatusNotAcceptable)
	}

	err = WaitForFirstRouterPublication(
		context.Background(), server.Client(), server.URL, "dashboard-session", time.Second, false,
	)
	if err != nil {
		t.Fatalf("WaitForFirstRouterPublication() error = %v", err)
	}
}
