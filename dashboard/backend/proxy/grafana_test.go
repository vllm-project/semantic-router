package proxy

import (
	"compress/gzip"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"testing"
)

func TestGrafanaProxyPreservesPublicOriginAndFixedTarget(t *testing.T) {
	for _, override := range []string{"false", "true"} {
		t.Run("override="+override, func(t *testing.T) {
			t.Setenv("PROXY_OVERRIDE_ORIGIN", override)
			for _, origin := range []string{"", "https://dashboard.example", "https://other.example"} {
				for _, method := range []string{http.MethodGet, http.MethodPost} {
					t.Run(method+"/"+origin, func(t *testing.T) {
						proxy, err := NewGrafanaProxy("http://grafana.internal:3000")
						if err != nil {
							t.Fatal(err)
						}
						request := httptest.NewRequest(method, "https://dashboard.example/embedded/grafana/api/live/ws?format=json", nil)
						if origin != "" {
							request.Header.Set("Origin", origin)
						}
						origins := append([]string(nil), request.Header.Values("Origin")...)
						proxy.Director(request)
						if request.Host != "dashboard.example" || !reflect.DeepEqual(request.Header.Values("Origin"), origins) {
							t.Fatalf("public authority changed: Host=%q Origin=%q", request.Host, request.Header.Values("Origin"))
						}
						want := &url.URL{Scheme: "http", Host: "grafana.internal:3000", Path: "/api/live/ws", RawQuery: "format=json"}
						if request.URL.String() != want.String() {
							t.Fatalf("upstream URL=%s, want %s", request.URL, want)
						}
					})
				}
			}
		})
	}
}

func TestGrafanaProxyInstallsAdapterBeforeApplication(t *testing.T) {
	for _, compressed := range []bool{false, true} {
		t.Run(fmt.Sprint("gzip=", compressed), func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if got := r.Header.Get("Accept-Encoding"); got != "identity" {
					t.Errorf("HTML encoding negotiation = %q", got)
				}
				if r.Header.Get("If-None-Match") != "" || r.Header.Get("If-Modified-Since") != "" {
					t.Error("HTML must be fetched again when the adapter changes")
				}
				w.Header().Set("Content-Type", "text/html; charset=utf-8")
				w.Header().Set("Content-Security-Policy", "script-src 'self'; frame-ancestors 'none'")
				w.Header().Set("ETag", "upstream-html")
				w.Header().Set("Last-Modified", "Sat, 19 Sep 2026 00:00:00 GMT")
				body := `<html><HEAD data-theme="dark"><script src="app.js"></script></HEAD><body>Grafana</body></html>`
				if compressed {
					w.Header().Set("Content-Encoding", "gzip")
					writer := gzip.NewWriter(w)
					_, _ = writer.Write([]byte(body))
					_ = writer.Close()
					return
				}
				_, _ = fmt.Fprint(w, body)
			}))
			defer upstream.Close()
			proxy, err := NewGrafanaProxy(upstream.URL)
			if err != nil {
				t.Fatal(err)
			}
			request := httptest.NewRequest(http.MethodGet, "/embedded/grafana/d/router", nil)
			request.Header.Set("Accept", "text/html")
			request.Header.Set("Accept-Encoding", "gzip")
			request.Header.Set("If-None-Match", "old-html")
			request.Header.Set("If-Modified-Since", "Sat, 19 Sep 2026 00:00:00 GMT")
			response := httptest.NewRecorder()
			proxy.ServeHTTP(response, request)
			body := response.Body.String()
			adapter := strings.Index(body, GrafanaAuthScriptPath)
			if response.Code != http.StatusOK || adapter < 0 || adapter > strings.Index(body, `src="app.js"`) {
				t.Fatalf("adapter must execute first: status=%d body=%s", response.Code, body)
			}
			if response.Header().Get("Content-Length") != fmt.Sprint(len(body)) || response.Header().Get("Content-Encoding") != "" {
				t.Fatal("rewritten HTML has stale length or encoding")
			}
			if response.Header().Get("ETag") != "" || response.Header().Get("Last-Modified") != "" || response.Header().Get("Cache-Control") != "no-store" {
				t.Fatal("rewritten HTML retained upstream cache validators")
			}
			if response.Header().Get("Content-Security-Policy") != "script-src 'self';frame-ancestors 'self'" {
				t.Fatal("adapter must not relax the script policy")
			}
		})
	}
}

func TestGrafanaProxyPreservesAPIAndStaticResponses(t *testing.T) {
	t.Setenv("PROXY_OVERRIDE_ORIGIN", "true")
	for _, contentType := range []string{"application/json", "application/javascript"} {
		t.Run(contentType, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Host != "dashboard.example" || r.Header.Get("Origin") != "https://dashboard.example" {
					http.Error(w, "public authority changed", http.StatusForbidden)
					return
				}
				if r.Header.Get("Accept-Encoding") != "gzip" || r.Header.Get("X-CSRF-Token") != "fixture-csrf" {
					t.Error("non-HTML request headers changed")
				}
				w.Header().Set("Content-Type", contentType)
				_, _ = fmt.Fprint(w, "unchanged response")
			}))
			defer upstream.Close()
			proxy, err := NewGrafanaProxy(upstream.URL)
			if err != nil {
				t.Fatal(err)
			}
			request := httptest.NewRequest(http.MethodPost, "https://dashboard.example/embedded/grafana/api/ds/query", strings.NewReader(`{"queries":[]}`))
			request.Header.Set("Origin", "https://dashboard.example")
			request.Header.Set("Accept-Encoding", "gzip")
			request.Header.Set("X-CSRF-Token", "fixture-csrf")
			response := httptest.NewRecorder()
			proxy.ServeHTTP(response, request)
			if response.Code != http.StatusOK || response.Body.String() != "unchanged response" {
				t.Fatalf("response = %d %s", response.Code, response.Body.String())
			}
		})
	}
}

func TestGrafanaAuthScriptHandler(t *testing.T) {
	for _, method := range []string{http.MethodGet, http.MethodHead, http.MethodPost} {
		response := httptest.NewRecorder()
		GrafanaAuthScriptHandler(response, httptest.NewRequest(method, GrafanaAuthScriptPath, nil))
		if method == http.MethodPost {
			if response.Code != http.StatusMethodNotAllowed {
				t.Fatalf("POST status = %d", response.Code)
			}
			continue
		}
		if response.Code != http.StatusOK || response.Header().Get("Content-Type") != "application/javascript; charset=utf-8" {
			t.Fatalf("%s status = %d", method, response.Code)
		}
		if method == http.MethodGet && response.Body.String() != string(grafanaAuthScript) {
			t.Fatal("script differs from the embedded source")
		}
		if method == http.MethodHead && response.Body.Len() != 0 {
			t.Fatal("HEAD returned a body")
		}
	}
}
