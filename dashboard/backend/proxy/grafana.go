package proxy

import (
	"bytes"
	"compress/gzip"
	_ "embed"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"strings"
)

// GrafanaAuthScriptPath is served through the normal authenticated proxy route.
const GrafanaAuthScriptPath = "/embedded/grafana/_dashboard/auth.js"

//go:embed grafana_auth.js
var grafanaAuthScript []byte

// GrafanaAuthScriptHandler serves the same-origin iframe's CSRF request adapter.
// It contains no session or CSRF token; the browser reads its current CSRF cookie.
func GrafanaAuthScriptHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.Header().Set("Allow", "GET, HEAD")
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/javascript; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Content-Length", fmt.Sprint(len(grafanaAuthScript)))
	if r.Method == http.MethodGet {
		_, _ = w.Write(grafanaAuthScript)
	}
}

// NewGrafanaProxy installs the browser adapter before Grafana starts. Grafana's
// own fetch runs inside its iframe, outside the Dashboard frontend's authFetch.
// Server-side authentication, origin, CSRF and permission checks remain intact.
func NewGrafanaProxy(targetBase string) (*httputil.ReverseProxy, error) {
	proxy, err := NewReverseProxy(targetBase, "/embedded/grafana", false)
	if err != nil {
		return nil, err
	}
	director := proxy.Director
	proxy.Director = func(r *http.Request) {
		director(r)
		if strings.Contains(r.Header.Get("Accept"), "text/html") {
			// Only HTML is rewritten; preserve compression for large static assets.
			r.Header.Set("Accept-Encoding", "identity")
			r.Header.Del("If-None-Match")
			r.Header.Del("If-Modified-Since")
		}
	}
	modifyResponse := proxy.ModifyResponse
	proxy.ModifyResponse = func(resp *http.Response) error {
		if err := modifyResponse(resp); err != nil {
			return err
		}
		if resp.StatusCode != http.StatusOK || resp.Request.Method == http.MethodHead ||
			!strings.Contains(resp.Header.Get("Content-Type"), "text/html") {
			return nil
		}
		defer resp.Body.Close()
		var reader io.Reader = resp.Body
		switch resp.Header.Get("Content-Encoding") {
		case "", "identity":
		case "gzip":
			decoded, err := gzip.NewReader(resp.Body)
			if err != nil {
				return err
			}
			defer decoded.Close()
			reader = decoded
		default:
			return fmt.Errorf("unexpected Grafana HTML content encoding: %s", resp.Header.Get("Content-Encoding"))
		}
		body, err := io.ReadAll(reader)
		if err != nil {
			return err
		}
		// Insert before upstream scripts, without relaxing the upstream script CSP.
		script := []byte(`<script src="` + GrafanaAuthScriptPath + `"></script>`)
		if head := bytes.Index(bytes.ToLower(body), []byte("<head")); head >= 0 {
			if end := bytes.IndexByte(body[head:], '>'); end >= 0 {
				index := head + end + 1
				body = append(append(append([]byte{}, body[:index]...), script...), body[index:]...)
			}
		}
		resp.Body = io.NopCloser(bytes.NewReader(body))
		resp.ContentLength = int64(len(body))
		resp.Header.Set("Content-Length", fmt.Sprint(len(body)))
		resp.Header.Del("ETag")
		resp.Header.Del("Last-Modified")
		resp.Header.Del("Content-Encoding")
		resp.Header.Set("Cache-Control", "no-store")
		return nil
	}
	return proxy, nil
}
