package auth

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestEmbeddedGrafanaQueryCSRFPolicy(t *testing.T) {
	f := newCSRFFixture(t)
	tests := []struct {
		name           string
		method         string
		path           string
		headers        map[string]string
		allowedOrigins []string
		status         int
	}{
		{name: "native embedded query", status: http.StatusNoContent},
		{name: "root API query", path: "/api/ds/query?requestId=Q1", status: http.StatusNoContent},
		{name: "JSON charset", headers: map[string]string{"Content-Type": "application/json; charset=utf-8"}, status: http.StatusNoContent},
		{name: "browser without fetch metadata", headers: map[string]string{"Sec-Fetch-Site": ""}, status: http.StatusNoContent},
		{name: "explicit proxy origin", headers: map[string]string{"Origin": "https://dashboard.example"}, allowedOrigins: []string{"https://dashboard.example"}, status: http.StatusNoContent},
		{name: "TLS terminator preserves host", headers: map[string]string{"Origin": "https://example.com", "X-Forwarded-Proto": "https"}, status: http.StatusNoContent},
		{name: "cross origin", headers: map[string]string{"Origin": "https://other.example"}, status: http.StatusForbidden},
		{name: "untrusted forwarded host", headers: map[string]string{"Origin": "https://other.example", "X-Forwarded-Host": "other.example", "X-Forwarded-Proto": "https"}, status: http.StatusForbidden},
		{name: "same site is insufficient", headers: map[string]string{"Sec-Fetch-Site": "same-site"}, status: http.StatusForbidden},
		{name: "cross site is insufficient", headers: map[string]string{"Sec-Fetch-Site": "cross-site"}, status: http.StatusForbidden},
		{name: "opaque origin", headers: map[string]string{"Origin": "null"}, status: http.StatusForbidden},
		{name: "referer alone is insufficient", headers: map[string]string{"Origin": "", "Referer": "http://example.com/embedded/grafana/"}, status: http.StatusForbidden},
		{name: "plain form", headers: map[string]string{"Content-Type": "application/x-www-form-urlencoded"}, status: http.StatusForbidden},
		{name: "plain text", headers: map[string]string{"Content-Type": "text/plain"}, status: http.StatusForbidden},
		{name: "missing content type", headers: map[string]string{"Content-Type": ""}, status: http.StatusForbidden},
		{name: "invalid explicit token", headers: map[string]string{csrfHeaderName: "invalid"}, status: http.StatusForbidden},
		{name: "query suffix is not allowlisted", path: "/embedded/grafana/api/ds/query/other", status: http.StatusForbidden},
		{name: "Grafana dashboard write still needs token", path: "/embedded/grafana/api/dashboards/db", status: http.StatusForbidden},
		{name: "Grafana datasource write still needs token", path: "/embedded/grafana/api/datasources", status: http.StatusForbidden},
		{name: "Dashboard write still needs token", path: "/api/settings", status: http.StatusForbidden},
		{name: "Jaeger write is not allowlisted", path: "/embedded/jaeger/api/traces", status: http.StatusForbidden},
		{name: "PUT is not query POST", method: http.MethodPut, status: http.StatusForbidden},
		{name: "normal Grafana write with token", path: "/embedded/grafana/api/dashboards/db", headers: map[string]string{csrfHeaderName: f.csrf}, status: http.StatusNoContent},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f.svc.SetAllowedOrigins(tc.allowedOrigins)
			method, path := tc.method, tc.path
			if method == "" {
				method = http.MethodPost
			}
			if path == "" {
				path = "/embedded/grafana/api/ds/query"
			}
			r := httptest.NewRequest(method, path, strings.NewReader(`{"queries":[]}`))
			r.AddCookie(&http.Cookie{Name: authSessionCookieName, Value: f.token})
			r.Header.Set("Origin", "http://example.com")
			r.Header.Set("Content-Type", "application/json")
			r.Header.Set("Sec-Fetch-Site", "same-origin")
			for name, value := range tc.headers {
				r.Header.Set(name, value)
			}
			response, ran := f.serve(t, r)
			if response.Code != tc.status || ran != (tc.status == http.StatusNoContent) {
				t.Fatalf("status=%d handler=%v body=%s, want %d", response.Code, ran, response.Body.String(), tc.status)
			}
		})
	}
}

func TestEmbeddedGrafanaQueryRetainsAuthenticationAndPermissionChecks(t *testing.T) {
	f := newCSRFFixture(t)
	claims, err := f.svc.ParseToken(f.token)
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"/embedded/grafana/api/ds/query", "/api/ds/query"} {
		r := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"queries":[]}`))
		r.Header.Set("Origin", "http://example.com")
		r.Header.Set("Content-Type", "application/json")
		r.Header.Set("Sec-Fetch-Site", "same-origin")
		response, ran := f.serve(t, r)
		if response.Code != http.StatusUnauthorized || ran {
			t.Fatalf("unauthenticated %s: status=%d handler=%v", path, response.Code, ran)
		}
	}
	if _, err := f.svc.store.UpdateUserRoleOrStatus(context.Background(), claims.UserID, RoleRead, ""); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"/embedded/grafana/api/ds/query", "/api/ds/query"} {
		r := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"queries":[]}`))
		r.AddCookie(&http.Cookie{Name: authSessionCookieName, Value: f.token})
		r.Header.Set("Origin", "http://example.com")
		r.Header.Set("Content-Type", "application/json")
		response, ran := f.serve(t, r)
		if response.Code != http.StatusForbidden || ran {
			t.Fatalf("missing logs.read %s: status=%d handler=%v", path, response.Code, ran)
		}
	}
}
